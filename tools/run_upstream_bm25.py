#!/usr/bin/env python3
"""Pure upstream BM25 (the initial lexical retrieval of the pinned SAG2 code; no model calls).

Run inside the upstream environment:
  cd vendor/SAG-Benchmark/sag2 && uv run python ../../../tools/run_upstream_bm25.py \
      --dataset musique --out ../../../results/raw/bm25_musique.jsonl

Writes a run file {"qid", "retrieved": [pid, ...]} with pid = 0-based corpus row (upstream document_id − 1).
"""
import argparse
import json
import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1] / "vendor" / "SAG-Benchmark" / "sag2"
sys.path.insert(0, str(UPSTREAM / "scripts"))

from benchmark_data import DatasetLoader  # noqa: E402
from sag2 import DEFAULT_CHUNK_CHARS, CorpusIndex, corpus_fingerprint  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=UPSTREAM / "dataset")
    parser.add_argument("--cache-dir", type=Path, default=UPSTREAM / ".cache" / "sag2")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    loader = DatasetLoader(args.dataset, str(args.dataset_dir))
    index = CorpusIndex(args.cache_dir / f"{args.dataset}.sqlite", loader.load_corpus(),
                        corpus_fingerprint(args.dataset_dir / f"{args.dataset}_corpus.json"), DEFAULT_CHUNK_CHARS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for record in loader.get_question_records():
            expression = index.match_query(record["question"])
            hits = index.retrieve(expression, set(), args.candidate_k) if expression else []
            documents = list(dict.fromkeys(hit["document_id"] for hit in hits))[: args.top_k]
            stream.write(json.dumps({"qid": record["id"], "retrieved": [str(d - 1) for d in documents]}) + "\n")
    index.close()


if __name__ == "__main__":
    main()
