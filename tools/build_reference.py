#!/usr/bin/env python3
"""Re-score the upstream per-question retrieval records with evalkit and write reference/ (handoff S0 step 8).

  python tools/build_reference.py

Reads retained records under vendor/SAG-Benchmark/sag2/results and our BM25 runs under results/raw/.
reference/<dataset>/<method>.jsonl holds one line per question: qid + metric values only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evalkit import aggregate, load_corpus, load_questions, load_run, load_split, score_run  # noqa: E402

UP = ROOT / "vendor" / "SAG-Benchmark" / "sag2" / "results"
STAGE_KEYS = {"Recall": "R", "AllEvidence": "AE"}


def rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def text_index(dataset):
    return {f"{r['title'].strip()}\n{r['text']}": pid for pid, r in load_corpus(dataset, ROOT / "data").items()}


def run_from_docs(path, dataset):
    by_text = text_index(dataset)
    return {r.get("dataset_sample_id", r.get("id")): [by_text[d] for d in r["retrieved_docs"]] for r in rows(path)}


def run_from_ids(path):
    return {r["dataset_sample_id"]: [str(i - 1) for i in r["retrieved_document_ids"]] for r in rows(path)}


def stage_scores(path, stage):
    out = {}
    for r in rows(path):
        metrics = r["stage_results"][stage]["metrics"]
        out[r["dataset_sample_id"]] = {
            f"{STAGE_KEYS[k.split('@')[0]]}@{k.split('@')[1]}": float(v) for k, v in metrics.items()}
    return out


def main():
    summary = {}
    rerank = json.loads((UP / "rerank/summary.json").read_text())["experiments"]
    plan = {
        "musique": {
            "bm25": ("run", load_run(ROOT / "results/raw/bm25_musique.jsonl"), "pinned upstream code, initial BM25, our run"),
            "sag2_jev": ("run", run_from_docs(UP / "controls/sag2_h2/questions.jsonl", "musique"),
                         "upstream controls/sag2_h2 (SAG2 + Jev, two expansions)"),
            "sag2_jev_h0": ("scores", stage_scores(UP / "controls/sag2_h2/questions.jsonl", "h0"),
                            "upstream controls/sag2_h2 stage h0 (stage metrics as recorded upstream)"),
            "sag2_jev_h1": ("scores", stage_scores(UP / "controls/sag2_h2/questions.jsonl", "h1"),
                            "upstream controls/sag2_h2 stage h1 (stage metrics as recorded upstream)"),
            "bm25_paper_control": ("scores", stage_scores(UP / "controls/sag2_h2/questions.jsonl", "bm25"),
                                   "upstream controls/sag2_h2 stage bm25 (as recorded upstream)"),
        },
        "2wikimultihopqa": {
            "bm25": ("run", load_run(ROOT / "results/raw/bm25_2wikimultihopqa.jsonl"), "pinned upstream code, initial BM25, our run"),
            "sag2_jev": ("run", run_from_docs(UP / "benchmarks/2wikimultihopqa/questions.jsonl", "2wikimultihopqa"),
                         "upstream benchmarks/2wikimultihopqa (SAG2 + Jev, two expansions)"),
        },
    }
    for name, method in (("jina_three", "sag2_jina"), ("jina_single", "bm25_jina_single"),
                         ("qwen_three", "sag2_qwen3rerank_302"), ("qwen_single", "bm25_qwen3rerank_302_single"),
                         ("jina_2wiki", "sag2_jina")):
        entry = rerank[name]
        if entry["dataset"] not in plan:
            continue
        plan[entry["dataset"]][method] = (
            "run", run_from_ids(UP / "rerank" / entry["records"]),
            f"upstream rerank/{entry['records']}: model {entry['model']}, protocol {entry['protocol']}, "
            f"expansions {entry['expansions']}, provider {entry['provider']}")
    for dataset, methods in plan.items():
        questions = load_questions(dataset, ROOT / "data")
        qids = list(questions)
        out = ROOT / "reference" / dataset
        out.mkdir(parents=True, exist_ok=True)
        summary[dataset] = {}
        for method, (kind, payload, source) in methods.items():
            scores = score_run(questions, payload, qids) if kind == "run" else payload
            if set(scores) != set(qids):
                raise ValueError(f"{dataset}/{method}: question set mismatch")
            with open(out / f"{method}.jsonl", "w", encoding="utf-8") as stream:
                for qid in qids:
                    stream.write(json.dumps({"qid": qid, **scores[qid]}) + "\n")
            entry = {"source": source, "rescored_by_evalkit": kind == "run", "all": aggregate(scores, qids)}
            for split in ("dev", "test", "calib", "check"):
                try:
                    entry[split] = aggregate(scores, load_split(dataset, split, ROOT / "data"))
                except KeyError:
                    pass
            summary[dataset][method] = entry
            a = entry["all"]
            print(f"{dataset:16s} {method:28s} n={a['n']} R@5={a['R@5']:.2f} R@10={a['R@10']:.2f} AE@10={a['AE@10']:.2f}"
                  + (f" | dev R@5={entry['dev']['R@5']:.2f}" if "dev" in entry else ""))
    (ROOT / "reference" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
