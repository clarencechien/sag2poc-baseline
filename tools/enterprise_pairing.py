#!/usr/bin/env python3
"""C2 step 0 (docs/handoff-b0-attribution.md): pair Taguri's EnterpriseRAG results with the upstream per-question
records, document level, no GPU.

  python tools/enterprise_pairing.py --taguri build/tb --out results/runs/c2-step0

Writes reference/enterprise/{sag2_jev,bm25_upstream}.jsonl (qid + metrics) and <out>/analysis.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evalkit import aggregate, load_run, paired_bootstrap, score_run  # noqa: E402

UP = ROOT / "vendor/SAG-Benchmark/sag2/results/enterprise/questions.jsonl"
TYPES = ["basic", "semantic", "intra_document_reasoning", "project_related", "constrained", "conflicting_info",
         "completeness", "miscellaneous"]


def rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taguri", type=Path, required=True, help="local copy of the needed taguri-b-results files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    upstream = rows(UP)
    questions = {r["qid"]: r for r in rows(args.taguri / "data/enterprise/questions.jsonl")}
    qids = [r["question_id"] for r in upstream]
    assert set(qids) == set(questions) and len(qids) == 470
    for r in upstream:
        assert set(r["gold_document_ids"]) == set(questions[r["question_id"]]["gold_pids"])
        assert r["question_type"] == questions[r["question_id"]]["type"]
    systems = {
        "sag2_jev": score_run(questions, {r["question_id"]: r["retrieved_source_ids"] for r in upstream}, qids),
        "bm25_upstream": score_run(questions, {r["question_id"]: r["bm25_source_ids"] for r in upstream}, qids),
    }
    # evalkit must reproduce the per-question metrics recorded upstream
    mismatch = 0
    for r in upstream:
        for name, key in (("sag2_jev", "sag2"), ("bm25_upstream", "bm25")):
            for k, v in r["metrics"][key].items():
                mismatch += abs(systems[name][r["question_id"]][k.replace("Recall", "R")] - v) > 1e-12
    partial = {}
    for name, sub in (("taguri_n0", "s10/enterprise-N0"), ("taguri_n1", "s10/enterprise-N1"),
                      ("taguri_n1_eg2", "s10/enterprise-N1+EG2")):
        run = load_run(args.taguri / sub / "run.jsonl")
        covered = [q for q in qids if q in run]
        systems[name] = score_run(questions, run, covered)
        partial[name] = len(covered)
    ref = ROOT / "reference" / "enterprise"
    ref.mkdir(parents=True, exist_ok=True)
    for name in ("sag2_jev", "bm25_upstream"):
        with open(ref / f"{name}.jsonl", "w") as stream:
            for q in qids:
                stream.write(json.dumps({"qid": q, **systems[name][q]}) + "\n")
    groups = {"overall": qids, **{t: [q for q in qids if questions[q]["type"] == t] for t in TYPES},
              "project_related+completeness": [q for q in qids if questions[q]["type"] in ("project_related", "completeness")]}
    pairs = [("taguri_n1", "sag2_jev"), ("taguri_n1", "bm25_upstream"), ("taguri_n0", "bm25_upstream"),
             ("taguri_n1", "taguri_n0"), ("sag2_jev", "bm25_upstream"), ("taguri_n1_eg2", "sag2_jev")]
    out = {"recomputation_mismatches_vs_upstream_record": int(mismatch), "coverage": partial, "groups": {}}
    for gname, members in groups.items():
        entry = {"n": len(members), "R@10": {}, "paired": {}}
        for name, table in systems.items():
            have = [q for q in members if q in table]
            if have:
                entry["R@10"][name] = {"n": len(have), "value": aggregate(table, have)["R@10"]}
        for a, b in pairs:
            both = [q for q in members if q in systems[a] and q in systems[b]]
            if len(both) >= 20:
                entry["paired"][f"{a} - {b}"] = paired_bootstrap(systems[a], systems[b], both, "R@10")
        out["groups"][gname] = entry
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print("mismatch vs upstream record:", mismatch, "coverage:", partial)
    for gname, entry in out["groups"].items():
        vals = "  ".join(f"{k}={v['value']:.2f}({v['n']})" for k, v in entry["R@10"].items())
        print(f"\n[{gname}] n={entry['n']}  {vals}")
        for label, p in entry["paired"].items():
            print(f"   {label:32s} n={p['n']:3d}  {p['delta']:+6.2f}  ({p['ci_low']:+.2f}, {p['ci_high']:+.2f})")


if __name__ == "__main__":
    main()
