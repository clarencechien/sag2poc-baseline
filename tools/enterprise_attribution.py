#!/usr/bin/env python3
"""C2 step 1 analysis: B0 vs SAG2+Jev (same loop, different judge) and Taguri vs B0 (same judge, different loop),
document-level R@10 on the questions B0 was run on.

  python tools/enterprise_attribution.py --b0 results/modal/c2-enterprise/results.jsonl --taguri build/tb \
      --bm25-ours results/runs/c2-step1/bm25_ours.jsonl --out results/runs/c2-step1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evalkit import aggregate, load_run, paired_bootstrap, score_run  # noqa: E402

UP = ROOT / "vendor/SAG-Benchmark/sag2/results/enterprise/questions.jsonl"


def rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--taguri", type=Path, required=True)
    parser.add_argument("--bm25-ours", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    upstream = {r["question_id"]: r for r in rows(UP)}
    b0 = {r["question_id"]: r for r in rows(args.b0)}
    ours = {r["question_id"]: r for r in rows(args.bm25_ours)}
    questions = {q: {"gold_pids": r["gold_document_ids"], "type": r["question_type"]} for q, r in upstream.items()}
    qids = [q for q in upstream if q in b0]
    systems = {
        "b0": score_run(questions, {q: b0[q]["retrieved_source_ids"] for q in qids}, qids),
        "b0_first_round_bm25": score_run(questions, {q: b0[q]["bm25_source_ids"] for q in qids}, qids),
        "bm25_our_index": score_run(questions, {q: ours[q]["bm25_source_ids"] for q in qids}, qids),
        "sag2_jev": score_run(questions, {q: upstream[q]["retrieved_source_ids"] for q in qids}, qids),
        "bm25_upstream_record": score_run(questions, {q: upstream[q]["bm25_source_ids"] for q in qids}, qids),
        "taguri_n1": score_run(questions, load_run(args.taguri / "s10/enterprise-N1/run.jsonl"), qids),
        "taguri_n0": score_run(questions, load_run(args.taguri / "s10/enterprise-N0/run.jsonl"), qids),
    }
    ref = ROOT / "reference" / "enterprise"
    ref.mkdir(parents=True, exist_ok=True)
    for name in ("b0", "bm25_our_index"):
        with open(ref / f"{name}.jsonl", "w") as stream:
            for q in qids:
                stream.write(json.dumps({"qid": q, **systems[name][q]}) + "\n")
    groups = {"all": qids}
    for t in dict.fromkeys(questions[q]["type"] for q in qids):
        groups[t] = [q for q in qids if questions[q]["type"] == t]
    pairs = [("b0", "sag2_jev"), ("taguri_n1", "b0"), ("taguri_n1", "sag2_jev"), ("b0", "bm25_our_index"),
             ("taguri_n1", "taguri_n0"), ("sag2_jev", "bm25_upstream_record"), ("bm25_our_index", "bm25_upstream_record")]
    out = {"n": len(qids), "groups": {}}
    for gname, members in groups.items():
        entry = {"n": len(members), "scores": {n: {k: aggregate(t, members)[k] for k in ("R@1", "R@5", "R@10", "AE@10")}
                                                for n, t in systems.items()}, "paired_R@10": {}, "paired_R@5": {}}
        for a, b in pairs:
            entry["paired_R@10"][f"{a} - {b}"] = paired_bootstrap(systems[a], systems[b], members, "R@10")
            entry["paired_R@5"][f"{a} - {b}"] = paired_bootstrap(systems[a], systems[b], members, "R@5")
        out["groups"][gname] = entry

    # judge behaviour per scoring round, document level (a document's score = its best chunk in that round)
    calib = {}
    for q in qids:
        gold = set(questions[q]["gold_pids"])
        for step in b0[q].get("rounds", []):
            best = {}
            for sid, score in step["scores"]:
                best[sid] = max(best.get(sid, 0.0), score)
            bucket = calib.setdefault(f"h{step['hop']}", {"gold": [], "non": [], "gold_total": 0})
            bucket["gold_total"] += len(gold)
            for sid, score in best.items():
                bucket["gold" if sid in gold else "non"].append(score)
    out["calibration"] = {}
    for name, b in sorted(calib.items()):
        gold, non = np.array(b["gold"]), np.array(b["non"])
        out["calibration"][name] = {
            "gold_documents_total": b["gold_total"], "gold_documents_judged": int(len(gold)),
            "gold_share_judged": len(gold) / b["gold_total"],
            "gold_share_p_ge_0.5": float((gold >= 0.5).mean()) if len(gold) else None,
            "gold_p_median": float(np.median(gold)) if len(gold) else None,
            "non_gold_documents_judged": int(len(non)), "non_gold_share_p_ge_0.5": float((non >= 0.5).mean()),
            "non_gold_per_question_p_ge_0.5": float((non >= 0.5).sum() / len(qids)),
        }
    usage = [b0[q]["usage"] for q in qids]
    seconds = [b0[q]["seconds"] for q in qids]
    out["usage"] = {
        "judgments_median": float(np.median([u["judgments"] for u in usage])),
        "judgments_p90": float(np.percentile([u["judgments"] for u in usage], 90)),
        "input_tokens_mean": float(np.mean([u["input_tokens"] for u in usage])),
        "cached_share": sum(u["cached_input_tokens"] for u in usage) / sum(u["input_tokens"] for u in usage),
        "seconds_per_question_median": float(np.median(seconds)), "seconds_per_question_p90": float(np.percentile(seconds, 90)),
        "judge_seconds_per_question_median": float(np.median([u["judge_seconds"] for u in usage])),
        "option_mass_min": min(u["option_mass_min"] for u in usage if u["option_mass_min"] is not None),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "attribution.json").write_text(json.dumps(out, indent=1) + "\n")
    for gname, entry in out["groups"].items():
        print(f"\n[{gname}] n={entry['n']}")
        for n, sc in entry["scores"].items():
            print(f"   {n:22s} R@5 {sc['R@5']:6.2f}  R@10 {sc['R@10']:6.2f}  AE@10 {sc['AE@10']:6.2f}")
        for label, p in entry["paired_R@10"].items():
            p5 = entry["paired_R@5"][label]
            print(f"   {label:40s} R@10 {p['delta']:+6.2f} ({p['ci_low']:+.2f}, {p['ci_high']:+.2f})   R@5 {p5['delta']:+6.2f} ({p5['ci_low']:+.2f}, {p5['ci_high']:+.2f})")
    print("\ncalibration", json.dumps(out["calibration"], indent=1))
    print("usage", json.dumps(out["usage"]))


if __name__ == "__main__":
    main()
