#!/usr/bin/env python3
"""Write a normalised dataset (corpus.jsonl / questions.jsonl / splits.json) in the HippoRAG-style layout the
upstream entry point reads. Format conversion only; file names use a dataset name the entry point accepts.

  python tools/convert_to_upstream.py --src build/tb/data/musique-zhtw2 --split dev_zh --out build/up_zhtw
"""
import argparse
import json
from pathlib import Path


def rows(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    corpus = rows(args.src / "corpus.jsonl")
    by_pid = {row["pid"]: row for row in corpus}
    keys = [f"{row['title'].strip()}\n{row['text']}" for row in corpus]
    questions = {row["qid"]: row for row in rows(args.src / "questions.jsonl")}
    qids = json.loads((args.src / "splits.json").read_text())[args.split]
    # Identical passages share one text key upstream. That is harmless unless one of them is gold.
    seen, duplicated = {}, set()
    for row, key in zip(corpus, keys):
        if key in seen:
            duplicated.update((row["pid"], seen[key]))
        seen.setdefault(key, row["pid"])
    gold = {pid for qid in qids for item in questions[qid]["gold_pids"] for pid in ([item] if isinstance(item, str) else item)}
    if duplicated & gold:
        raise ValueError(f"duplicate passages include gold pids: {sorted(duplicated & gold)}")
    if duplicated:
        print(f"note: {len(duplicated)} passages have identical text (none is gold): {sorted(duplicated)}")
    out = []
    for qid in qids:
        flat = [pid for item in questions[qid]["gold_pids"] for pid in ([item] if isinstance(item, str) else item)]
        out.append({"id": qid, "question": questions[qid]["question"], "answerable": True,
                    "paragraphs": [{"title": by_pid[pid]["title"].strip(), "paragraph_text": by_pid[pid]["text"],
                                    "is_supporting": True} for pid in dict.fromkeys(flat)]})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "musique.json").write_text(json.dumps(out, ensure_ascii=False))
    (args.out / "musique_corpus.json").write_text(
        json.dumps([{"title": row["title"], "text": row["text"]} for row in corpus], ensure_ascii=False))
    print(args.out, len(corpus), "passages", len(out), "questions")


if __name__ == "__main__":
    main()
