#!/usr/bin/env python3
"""Analyse one B0 run pulled from the Modal volume (S1 smoke numbers, S2 judge quality).

  python tools/analyze_b0.py --run results/modal/s1-smoke --dataset musique [--hist results/fig/s2] [--compare ...]

Reads <run>/run/<dataset>/details.jsonl, <run>/judgments.jsonl and <run>/_env.json.
Writes <run>/run.jsonl (qid + retrieved pids), <run>/scores.jsonl and <run>/analysis.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evalkit import aggregate, load_corpus, load_questions, load_scores, paired_bootstrap, score_run  # noqa: E402


def rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):  # average ranks over ties
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos, neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pct(values, q):
    return float(np.percentile(values, q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--hist", type=Path, help="prefix for histogram PNGs (hop-0 judgments only)")
    parser.add_argument("--compare", nargs="*", default=[], help="reference methods to pair against, e.g. bm25")
    args = parser.parse_args()

    corpus = load_corpus(args.dataset, ROOT / "data")
    questions = load_questions(args.dataset, ROOT / "data")
    by_text = {f"{r['title'].strip()}\n{r['text']}": pid for pid, r in corpus.items()}
    by_sha = {hashlib.sha1(text.encode()).hexdigest(): pid for text, pid in by_text.items()}
    by_question = {q["question"]: qid for qid, q in questions.items()}

    details = rows(args.run / "run" / args.dataset / "details.jsonl")
    env = json.loads((args.run / "_env.json").read_text())
    qids = [d["dataset_sample_id"] for d in details]
    run = {d["dataset_sample_id"]: [by_text[doc] for doc in d["retrieved_docs"]] for d in details}
    bm25_in_run = {d["dataset_sample_id"]: [by_text[doc] for doc in d["bm25_docs"]] for d in details}
    with open(args.run / "run.jsonl", "w") as stream:
        for qid in qids:
            stream.write(json.dumps({"qid": qid, "retrieved": run[qid]}) + "\n")
    scores = score_run(questions, run, qids)
    with open(args.run / "scores.jsonl", "w") as stream:
        for qid in qids:
            stream.write(json.dumps({"qid": qid, **scores[qid]}) + "\n")
    out = {"run": str(args.run), "n": len(qids), "env": {k: v for k, v in env.items() if k != "freeze"},
           "b0": aggregate(scores, qids), "bm25_inside_run": aggregate(score_run(questions, bm25_in_run, qids), qids)}

    # lexical parity: the first-round BM25 inside the container must equal our local upstream BM25 run
    local = load_scores(ROOT / "reference" / args.dataset / "bm25.jsonl")
    inside = score_run(questions, bm25_in_run, qids)
    out["bm25_parity_mismatches"] = sum(any(abs(inside[q][k] - local[q][k]) > 1e-12 for k in inside[q]) for q in qids)

    # usage per question
    judgments = [d["reranker_usage"]["judgments"] for d in details]
    tokens = [d["reranker_usage"]["input_tokens"] for d in details]
    cached = [sum(b.get("cached_input_tokens", 0) for b in d["reranker_usage"]["batches"]) for d in details]
    total_tokens, total_cached = sum(tokens), sum(cached)
    bench = env["bench_wall_s"]
    usd_hour = env.get("gpu_usd_per_hour") or 1.95
    out["usage"] = {
        "judgments_per_question": {"median": pct(judgments, 50), "p90": pct(judgments, 90), "mean": float(np.mean(judgments))},
        "input_tokens_per_question": {"median": pct(tokens, 50), "mean": float(np.mean(tokens))},
        "cached_share": total_cached / total_tokens,
        "uncached_tokens_per_question": (total_tokens - total_cached) / len(qids),
        "prefill_tokens_per_s_uncached": (total_tokens - total_cached) / bench,
        "prefill_tokens_per_s_all": total_tokens / bench,
        "judgments_per_s": sum(judgments) / bench,
        "gpu_seconds_per_question": bench / len(qids),
        "cold_start_s": env["server_start_s"],
        "bench_wall_s": bench,
        "container_wall_s": env["container_wall_s"],
        "usd_this_run": env["container_wall_s"] / 3600 * usd_hour,
        "usd_per_100_questions_warm": bench / len(qids) * 100 / 3600 * usd_hour,
        "usd_per_100_questions_incl_cold_start": (bench / len(qids) * 100 + env["server_start_s"]) / 3600 * usd_hour,
    }

    # option mass and judge quality from the adapter log
    log = rows(args.run / "judgments.jsonl")
    mass = np.array([m for r in log for m in r["option_mass"]])
    out["option_mass"] = {"judgments": int(len(mass)), "median": pct(mass, 50), "p10": pct(mass, 10), "p1": pct(mass, 1),
                          "min": float(mass.min()), "share_below_0.5": float((mass < 0.5).mean()),
                          "share_below_0.9": float((mass < 0.9).mean())}
    out["max_prompt_tokens"] = int(max(t for r in log for t in r["prompt_tokens"]))
    pos, neg, unmapped, per_question = [], [], 0, []
    for r in log:
        if r["evidence_count"] != 0:
            continue  # first scoring round only: no evidence in the prompt
        qid = by_question.get(r["original_question"])
        if qid is None or qid not in set(qids):
            unmapped += len(r["p_yes"])
            continue
        gold = set(questions[qid]["gold_pids"])
        labels, values = [], []
        for sha, p in zip(r["docs_sha1"], r["p_yes"]):
            pid = by_sha.get(sha)
            if pid is None:
                unmapped += 1
                continue
            labels.append(pid in gold)
            values.append(p)
            (pos if pid in gold else neg).append(p)
        if any(labels) and not all(labels):
            per_question.append(auroc(values, labels))
    if pos and neg:
        out["judge_hop0"] = {
            "gold_candidates": len(pos), "non_gold_candidates": len(neg), "unmapped": unmapped,
            "auroc_pooled": auroc(pos + neg, [True] * len(pos) + [False] * len(neg)),
            "auroc_per_question_mean": float(np.mean(per_question)), "questions_with_gold_candidate": len(per_question),
            "p_yes_gold": {"median": pct(pos, 50), "p25": pct(pos, 25), "p75": pct(pos, 75), "share_ge_0.5": float(np.mean(np.array(pos) >= 0.5))},
            "p_yes_non_gold": {"median": pct(neg, 50), "p75": pct(neg, 75), "p95": pct(neg, 95), "share_ge_0.5": float(np.mean(np.array(neg) >= 0.5))},
        }
        if args.hist:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            args.hist.parent.mkdir(parents=True, exist_ok=True)
            bins = np.linspace(0, 1, 21)
            for name, values, title in (("gold", pos, "Gold evidence passages"), ("non_gold", neg, "Non-evidence passages")):
                fig, ax = plt.subplots(figsize=(6, 3.6))
                ax.hist(values, bins=bins, color="#3b6ea5" if name == "gold" else "#8a8a8a", edgecolor="white")
                ax.set_title(f"{title}: P(yes), first scoring round (n={len(values)})")
                ax.set_xlabel("P(yes) = P(A) / (P(A) + P(B))")
                ax.set_ylabel("candidates")
                if name == "non_gold":
                    ax.set_yscale("log")
                fig.tight_layout()
                fig.savefig(f"{args.hist}_{name}.png", dpi=150)
                plt.close(fig)

    out["paired"] = {}
    for method in args.compare:
        ref = load_scores(ROOT / "reference" / args.dataset / f"{method}.jsonl")
        out["paired"][method] = {"reference": aggregate(ref, qids),
                                 **{m: paired_bootstrap(scores, ref, qids, m) for m in ("R@5", "R@10", "AE@10")}}
    (args.run / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "env"}, indent=1))


if __name__ == "__main__":
    main()
