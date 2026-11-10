#!/usr/bin/env python3
"""Convert the HippoRAG 2 release files (MuSiQue, 2WikiMultiHopQA) to the normalised layout (handoff §5.1)
and write the fixed splits (handoff §5.3).

  python tools/convert_hipporag.py --src vendor/SAG-Benchmark/sag2/dataset --out data

pid = 0-based row index of the passage in <name>_corpus.json, as a string.
Gold evidence: MuSiQue paragraphs flagged is_supporting; 2Wiki context entries whose title is in supporting_facts
(sentences joined by one space). A gold passage is matched to the corpus on the exact "title\\ntext" string.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

SEED = 20261110
TOLERANCE = 0.05  # handoff §5.3: each class within 5 percentage points of the whole


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(title, text):
    return f"{title.strip()}\n{text}"


def convert(name, src):
    corpus = json.loads((src / f"{name}_corpus.json").read_text(encoding="utf-8"))
    samples = json.loads((src / f"{name}.json").read_text(encoding="utf-8"))
    by_key = {}
    for i, row in enumerate(corpus):
        by_key.setdefault(key(row["title"], row["text"]), str(i))
    if len(by_key) != len(corpus):
        raise ValueError(f"{name}: duplicate passages in corpus")
    questions, unmatched = [], 0
    for sample in samples:
        if name == "musique":
            gold = [key(p["title"], p.get("paragraph_text") or p.get("text") or "")
                    for p in sample["paragraphs"] if p["is_supporting"]]
            qtype = sample["id"].split("hop")[0] + "hop"
        else:
            titles = {fact[0] for fact in sample["supporting_facts"]}
            gold = [key(title, " ".join(sentences)) for title, sentences in sample["context"] if title in titles]
            qtype = sample["type"]
        gold = list(dict.fromkeys(gold))
        unmatched += sum(g not in by_key for g in gold)
        pids = [by_key[g] for g in gold if g in by_key]
        if not pids:
            raise ValueError(f"{name}: question without gold evidence")
        questions.append({"qid": sample.get("_id", sample.get("id")), "question": sample["question"].strip(),
                          "gold_pids": pids, "type": qtype})
    return corpus, questions, unmatched


def decide_split(questions, n):
    """First n if every type's share is within TOLERANCE of the whole; else stratified sample with SEED."""
    whole = collections.Counter(q["type"] for q in questions)
    head = collections.Counter(q["type"] for q in questions[:n])
    gaps = {t: head[t] / n - whole[t] / len(questions) for t in whole}
    report = {"n": n, "whole": dict(whole), "first_n": dict(head),
              "gap_pp": {t: round(100 * g, 2) for t, g in gaps.items()}}
    if all(abs(g) < TOLERANCE for g in gaps.values()):
        return [q["qid"] for q in questions[:n]], {**report, "method": "first_n"}
    import random
    rng = random.Random(SEED)
    chosen = set()
    quota = {t: round(n * whole[t] / len(questions)) for t in whole}
    for t in sorted(quota):
        pool = [q["qid"] for q in questions if q["type"] == t]
        chosen.update(rng.sample(pool, quota[t]))
    ordered = [q["qid"] for q in questions if q["qid"] in chosen]
    return ordered, {**report, "method": f"stratified_seed_{SEED}", "quota": quota}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    manifest = {}
    for name, n_dev in (("musique", 300), ("2wikimultihopqa", 100)):
        corpus, questions, unmatched = convert(name, args.src)
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "corpus.jsonl",
                    [{"pid": str(i), "title": r["title"], "text": r["text"]} for i, r in enumerate(corpus)])
        write_jsonl(out / "questions.jsonl", questions)
        dev, report = decide_split(questions, n_dev)
        splits = {"dev": dev}
        if name == "musique":
            dev_set = set(dev)
            splits.update({"calib": dev[:150], "check": dev[150:], "dev_zh_source": dev[:100],
                           "test": [q["qid"] for q in questions if q["qid"] not in dev_set]})
        (out / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        manifest[name] = {
            "source": "osunlp/HippoRAG_2@5ec05b38deecc3318bb432c69865959c56058990",
            "source_sha256": {f"{name}.json": sha256(args.src / f"{name}.json"),
                              f"{name}_corpus.json": sha256(args.src / f"{name}_corpus.json")},
            "passages": len(corpus), "questions": len(questions), "gold_not_in_corpus": unmatched,
            "types": dict(collections.Counter(q["type"] for q in questions)),
            "split_sizes": {k: len(v) for k, v in splits.items()}, "split_decision": report,
        }
        print(name, json.dumps(manifest[name]["split_decision"]), manifest[name]["split_sizes"], "unmatched", unmatched)
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
