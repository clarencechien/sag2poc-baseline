#!/usr/bin/env python3
"""Analyse a two-expansion B0 run (S3, S4): final scores, per-stage scores, calibration, log-odds re-rank.

Run inside the upstream environment (it re-uses the upstream diversity ranking to rebuild stage results):

  cd vendor/SAG-Benchmark/sag2 && uv run --no-sync --with numpy python ../../../tools/analyze_loop.py \
      --suite ../../../results/modal/s3-b0/run-musique/musique --log ../../../results/modal/s3-b0/judgments-musique.jsonl \
      --index .cache/sag2/musique.sqlite --dataset musique --split dev --wall 2070 --out ../../../results/runs/s3-b0/musique

Outputs in --out: scores_<name>.jsonl for final / h0 / h1 / logodds, run.jsonl, analysis.json.
"""
import argparse
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "SAG-Benchmark" / "sag2" / "scripts"))
from evalkit import aggregate, load_questions, load_scores, load_split, paired_bootstrap, score_run  # noqa: E402
from evalkit.metrics import gold_items  # noqa: E402
from sag2 import rank_dpp  # noqa: E402


def rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True, help="<run>/<upstream dataset name> directory")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-dataset-dir", type=Path, help="dataset dir the index was built from (to open it)")
    parser.add_argument("--upstream-name", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--wall", type=float, required=True, help="benchmark wall seconds for this question set")
    parser.add_argument("--usd-per-hour", type=float, default=1.95)
    parser.add_argument("--compare", nargs="*", default=[])
    parser.add_argument("--stage-compare", nargs="*", default=[], help="h0=<ref> h1=<ref>")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from benchmark_data import DatasetLoader
    from sag2 import DEFAULT_CHUNK_CHARS, CorpusIndex, corpus_fingerprint

    upstream_name = args.upstream_name or args.suite.name
    dataset_dir = args.index_dataset_dir or (ROOT / "vendor/SAG-Benchmark/sag2/dataset")
    index = CorpusIndex(args.index, DatasetLoader(upstream_name, str(dataset_dir)).load_corpus(),
                        corpus_fingerprint(dataset_dir / f"{upstream_name}_corpus.json"), DEFAULT_CHUNK_CHARS)
    idf = index.diversity_idf
    db = sqlite3.connect(args.index)
    chunks = {cid: {"id": cid, "document_id": doc, "content": content, "title": title}
              for cid, doc, title, content in db.execute("SELECT id, document_id, title, content FROM chunks")}
    by_sha = {hashlib.sha1(f"{c['title']}\n{c['content']}".encode()).hexdigest(): c for c in chunks.values()}

    questions = load_questions(args.dataset, ROOT / "data")
    qids = load_split(args.dataset, args.split, ROOT / "data")
    by_question = {questions[q]["question"]: q for q in qids}
    details = {d["dataset_sample_id"]: d for d in rows(args.suite / "details.jsonl")}
    if set(details) != set(qids):
        raise ValueError(f"run covers {len(details)} questions, split has {len(qids)}")

    pid_by_full_doc = {full: str(doc - 1) for doc, full in db.execute("SELECT DISTINCT document_id, full_doc FROM chunks")}

    def docs_top10(ranked_chunks):
        return [str(d - 1) for d in dict.fromkeys(c["document_id"] for c in ranked_chunks)][:10]

    # ---- final and per-stage results rebuilt from the trace -------------------------------------------------------
    runs = {"final": {}, "h0": {}, "h1": {}, "h2": {}}
    stage_missing = {"h1": 0, "h2": 0}
    final_mismatch = 0
    for qid in qids:
        d = details[qid]
        stage_runs = {}
        for step in d["trace"]:
            if step["stage"] != "reranker":
                continue
            ranked = [{**chunks[s["id"]], "jev_score": s["score"]} for s in step["scores"]]
            ranked = rank_dpp(ranked[:10], idf) + ranked[10:]
            stage_runs[f"h{step['hop']}"] = docs_top10(ranked)
        last = None
        for name in ("h0", "h1", "h2"):
            if name in stage_runs:
                last = stage_runs[name]
            else:
                stage_missing[name] = stage_missing.get(name, 0) + 1
            runs[name][qid] = last or []
        runs["final"][qid] = [pid_by_full_doc[text] for text in dict.fromkeys(d["retrieved_docs"])]
        final_mismatch += runs["final"][qid] != runs["h2"][qid]
    scores = {name: score_run(questions, run, qids) for name, run in runs.items()}

    # ---- judgments log: calibration per stage and log-odds re-rank ------------------------------------------------
    gold_sets = {q: set().union(*gold_items(questions[q]["gold_pids"])) for q in qids}
    calib = {name: {"gold": [], "non": []} for name in ("h0", "h1", "h2")}
    latest = {q: {} for q in qids}
    masses, prompt_lengths, unmapped, missing_logp = [], [], 0, 0
    for record in rows(args.log):
        qid = by_question.get(record["original_question"])
        if qid is None:
            unmapped += len(record["p_yes"])
            continue
        stage = "h0" if record["evidence_count"] == 0 else ("h2" if record.get("evidence_has_scores") else "h1")
        masses.extend(record["option_mass"])
        prompt_lengths.extend(record["prompt_tokens"])
        for sha, p, la, lb in zip(record["docs_sha1"], record["p_yes"], record["logp_a"], record["logp_b"]):
            chunk = by_sha.get(sha)
            if chunk is None:
                unmapped += 1
                continue
            if la is None or lb is None:
                missing_logp += 1
                la = la if la is not None else -50.0
                lb = lb if lb is not None else -50.0
            pid = str(chunk["document_id"] - 1)
            calib[stage]["gold" if pid in gold_sets[qid] else "non"].append(p)
            latest[qid][chunk["id"]] = (la - lb, chunk)
    calibration = {}
    for name, groups in calib.items():
        gold, non = np.array(groups["gold"]), np.array(groups["non"])
        calibration[name] = {
            "gold_judged": int(len(gold)), "non_gold_judged": int(len(non)),
            "gold_share_p_ge_0.5": float((gold >= 0.5).mean()) if len(gold) else None,
            "gold_p_median": pct(gold, 50), "gold_p_p25": pct(gold, 25), "gold_p_p75": pct(gold, 75),
            "non_gold_share_p_ge_0.5": float((non >= 0.5).mean()) if len(non) else None,
            "non_gold_count_p_ge_0.5": int((non >= 0.5).sum()), "gold_count_p_ge_0.5": int((gold >= 0.5).sum()),
        }
    logodds_run = {}
    for qid in qids:
        ordered = sorted(latest[qid].values(), key=lambda item: (-item[0], item[1]["id"]))
        logodds_run[qid] = docs_top10([chunk for _, chunk in ordered])
    scores["logodds"] = score_run(questions, logodds_run, qids)

    # ---- usage -----------------------------------------------------------------------------------------------------
    judgments = [details[q]["reranker_usage"]["judgments"] for q in qids]
    tokens = [details[q]["reranker_usage"]["input_tokens"] for q in qids]
    cached = [sum(b.get("cached_input_tokens", 0) for b in details[q]["reranker_usage"]["batches"]) for q in qids]
    masses = np.array(masses)
    usage = {
        "judgments_per_question": {"median": pct(judgments, 50), "p90": pct(judgments, 90), "mean": float(np.mean(judgments))},
        "input_tokens_per_question_mean": float(np.mean(tokens)),
        "cached_share": sum(cached) / sum(tokens),
        "uncached_tokens_per_question": (sum(tokens) - sum(cached)) / len(qids),
        "prefill_tokens_per_s_uncached": (sum(tokens) - sum(cached)) / args.wall,
        "gpu_seconds_per_question": args.wall / len(qids),
        "usd_per_100_questions_warm": args.wall / len(qids) * 100 / 3600 * args.usd_per_hour,
        "max_prompt_tokens": int(max(prompt_lengths)),
        "option_mass": {"judgments": int(len(masses)), "median": pct(masses, 50), "p1": pct(masses, 1),
                        "min": float(masses.min()), "share_below_0.5": float((masses < 0.5).mean())},
    }

    # ---- write -----------------------------------------------------------------------------------------------------
    for name, table in scores.items():
        with open(args.out / f"scores_{name}.jsonl", "w") as stream:
            for qid in qids:
                stream.write(json.dumps({"qid": qid, **table[qid]}) + "\n")
    with open(args.out / "run.jsonl", "w") as stream:
        for qid in qids:
            stream.write(json.dumps({"qid": qid, "retrieved": runs["final"][qid]}) + "\n")
    out = {
        "n": len(qids), "checks": {"final_rebuilt_differs_from_actual": final_mismatch, "stage_missing": stage_missing,
                                   "log_unmapped": unmapped, "log_missing_logprob": missing_logp},
        "aggregate": {name: aggregate(table, qids) for name, table in scores.items()},
        "calibration": calibration, "usage": usage, "paired": {}, "stage_paired": {}, "by_type": {},
    }
    refs = ROOT / "reference" / args.dataset
    for method in args.compare:
        ref = load_scores(refs / f"{method}.jsonl")
        out["paired"][method] = {"reference": aggregate(ref, qids),
                                 **{m: paired_bootstrap(scores["final"], ref, qids, m) for m in ("R@5", "R@10", "AE@10")}}
    for pair in args.stage_compare:
        stage, method = pair.split("=")
        ref = load_scores(refs / f"{method}.jsonl")
        out["stage_paired"][stage] = {"reference_method": method, "reference": aggregate(ref, qids),
                                      **{m: paired_bootstrap(scores[stage], ref, qids, m) for m in ("R@5", "AE@10")}}
    out["logodds_vs_final"] = {m: paired_bootstrap(scores["logodds"], scores["final"], qids, m) for m in ("R@5", "R@10", "AE@10")}
    for qtype in sorted({questions[q]["type"] for q in qids}):
        subset = [q for q in qids if questions[q]["type"] == qtype]
        out["by_type"][qtype] = {"final": aggregate(scores["final"], subset)}
        for method in args.compare:
            out["by_type"][qtype][method] = aggregate(load_scores(refs / f"{method}.jsonl"), subset)
    (args.out / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    index.close()

    def short(a):
        return {k: round(a[k], 2) for k in ("R@1", "R@5", "R@10", "AE@10")}
    print("checks", out["checks"])
    for name, a in out["aggregate"].items():
        print(f"{name:8s}", short(a))
    for method, v in out["paired"].items():
        print("vs", method, short(v["reference"]),
              {m: (round(v[m]["delta"], 2), round(v[m]["ci_low"], 2), round(v[m]["ci_high"], 2)) for m in ("R@5", "R@10", "AE@10")})
    for stage, v in out["stage_paired"].items():
        print(stage, "vs", v["reference_method"], round(v["reference"]["R@5"], 2), round(v["reference"]["AE@10"], 2),
              {m: (round(v[m]["delta"], 2), round(v[m]["ci_low"], 2), round(v[m]["ci_high"], 2)) for m in ("R@5", "AE@10")})
    print("logodds vs final", {m: (round(v["delta"], 2), round(v["ci_low"], 2), round(v["ci_high"], 2)) for m, v in out["logodds_vs_final"].items()})
    print("calibration", json.dumps(calibration))
    print("usage", json.dumps(usage))
    print("by_type", json.dumps({t: {m: round(a["R@5"], 1) for m, a in v.items()} for t, v in out["by_type"].items()}))


if __name__ == "__main__":
    main()
