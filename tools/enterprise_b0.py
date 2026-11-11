#!/usr/bin/env python3
"""B0 on EnterpriseRAG-Bench (C2 step 1, docs/handoff-b0-attribution.md).

The upstream enterprise entry point only talks to the hosted judge. This driver re-uses the upstream pieces
unchanged (read-only index, search loop, the entry point's own search parameters, local-mode scorer) and plugs in
the jevlike backend. Nothing under vendor/ is modified.

Run inside the upstream environment, with PYTHONPATH pointing at adapters/:

  python tools/enterprise_b0.py --dataset-dir <official dir> --index-summary <summary.json> --bm25-only --out <dir>
  python tools/enterprise_b0.py --dataset-dir <official dir> --index-summary <summary.json> \
      --types project_related completeness --model-path <model dir> --out <dir>

Output: <out>/results.jsonl, one line per question, document-level (source ids).
"""
import argparse
import asyncio
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1] / "vendor" / "SAG-Benchmark" / "sag2"
for candidate in (UPSTREAM / "scripts", Path("/opt/upstream/sag2/scripts")):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

from enterprise_index import ReadOnlyIndex  # noqa: E402
from run_enterprise import PARAMETERS  # noqa: E402
from sag2 import search_sag2  # noqa: E402


def top_documents(candidates, k=10):
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate["source_id"], candidate)
    return [c["source_id"] for c in list(unique.values())[:k]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--index-summary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, help="override the documents directory recorded at build time")
    parser.add_argument("--types", nargs="*", help="question types to run; default all labelled questions")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bm25-only", action="store_true", help="first-round lexical retrieval only, no judge")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    build = json.loads(args.index_summary.read_text())
    source_dir = args.source_dir or Path(build["source_directory"])
    questions = [json.loads(line) for line in (args.dataset_dir / "questions.jsonl").read_text().splitlines()]
    selected = [q for q in questions if q["expected_doc_ids"] and (not args.types or q["question_type"] in args.types)]
    if args.limit:
        selected = selected[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "results.jsonl"
    done = {json.loads(line)["question_id"] for line in path.read_text().splitlines()} if path.exists() else set()
    pending = [q for q in selected if q["question_id"] not in done]
    print(f"{len(selected)} questions selected, {len(done)} already done, {len(pending)} to run", flush=True)

    started = time.perf_counter()
    master = ReadOnlyIndex(Path(build["index_path"]), build["index_signature"], source_dir)
    shared = ((master.frequencies, master.total, master.document_count), master.lexical_statistics)
    print(f"index ready in {time.perf_counter() - started:.1f}s: {master.document_count} documents, {master.total} chunks",
          flush=True)

    engine = None
    if not args.bm25_only:
        from jevlike_backend import JevlikeReranker
        from local_reranker import LocalScorer

        engine = JevlikeReranker(args.model_path, batch_size=args.batch_size)

    write_lock = threading.Lock()
    local = threading.local()

    def thread_index():
        if not hasattr(local, "index"):
            local.index = ReadOnlyIndex(Path(build["index_path"]), build["index_signature"], source_dir,
                                        shared[0], lexical_statistics=shared[1])
        return local.index

    def run(question):
        index = thread_index()
        t0 = time.perf_counter()
        row = {"question_id": question["question_id"], "question_type": question["question_type"],
               "gold_document_ids": sorted(question["expected_doc_ids"])}
        if args.bm25_only:
            expression = index.match_query(question["question"])
            hits = index.retrieve(expression, set(), PARAMETERS["candidate_k"]) if expression else []
            row["bm25_source_ids"] = top_documents(hits)
        else:
            scorer = LocalScorer(engine, args.out / "requests" / question["question_id"], "sag2")
            result = asyncio.run(search_sag2(index, scorer, question["question"], **PARAMETERS))
            usage = result["jev_usage"]
            row["retrieved_source_ids"] = top_documents(result["candidates"])
            row["bm25_source_ids"] = top_documents(result["baseline"])
            # per scoring round: the documents judged and their scores (document level, best chunk first)
            chunk_ids = {s["id"] for step in result["trace"] if step["stage"] == "jev" for s in step["scores"]}
            source_of = {cid: sid for cid, sid in index.db.execute(
                "SELECT id, source_id FROM chunks WHERE id IN (SELECT value FROM json_each(?))",
                (json.dumps(sorted(chunk_ids)),))}
            row["rounds"] = [{"hop": step["hop"], "scores": [[source_of[s["id"]], s["score"]] for s in step["scores"]]}
                             for step in result["trace"] if step["stage"] == "jev"]
            row["usage"] = {
                "judgments": usage["judgments"], "input_tokens": usage["input_tokens"],
                "cached_input_tokens": sum(b.get("cached_input_tokens", 0) for b in usage["batches"]),
                "judge_seconds": round(sum(b.get("judge_seconds", 0) for b in usage["batches"]), 2),
                "option_mass_min": min((b["option_mass_min"] for b in usage["batches"] if b.get("option_mass_min") is not None),
                                       default=None),
            }
        row["seconds"] = round(time.perf_counter() - t0, 2)
        with write_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        return row

    finished = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for row in pool.map(run, pending):
            finished += 1
            if finished % 10 == 0 or finished == len(pending):
                print(f"done {finished}/{len(pending)}, {time.perf_counter() - started:.1f}s", flush=True)
    master.close()
    print(json.dumps({"questions": len(selected), "wall_seconds": round(time.perf_counter() - started, 1)}))


if __name__ == "__main__":
    main()
