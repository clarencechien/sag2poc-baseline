#!/usr/bin/env python3
"""Convert MultiHop-RAG to the normalised layout (handoff S4) and to the HippoRAG-style files the upstream
benchmark entry point reads.

  python tools/convert_multihop_rag.py --src <dir with MultiHopRAG.json, corpus.json> \
      --out data/multihop-rag --upstream-out build/mhr_upstream

Source: Hugging Face yixuantt/MultiHopRAG @ 71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82 (ODC-BY).

Chunking: each article is split at paragraph boundaries (blank lines) into passages of at most 1,200 characters;
a single paragraph longer than that is split at sentence ends, then hard-split as a last resort.
Gold: for every evidence sentence, the passage(s) of its article that contain the sentence after whitespace
normalisation. If none does (e.g. the sentence straddles two passages), the evidence falls back to
"any passage of that article" and the question is counted in MANIFEST.json.
null_query questions (no answer in the corpus) are excluded.
"""
import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

SEED = 20261110
MAX_CHARS = 1200
SAMPLE = 300
REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def split_long(paragraph):
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > MAX_CHARS:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:MAX_CHARS])
            sentence = sentence[MAX_CHARS:]
        if current and len(current) + 1 + len(sentence) > MAX_CHARS:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


def chunk(body):
    units = []
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = paragraph.strip()
        if paragraph:
            units.extend([paragraph] if len(paragraph) <= MAX_CHARS else split_long(paragraph))
    passages, current = [], ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > MAX_CHARS:
            passages.append(current)
            current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        passages.append(current)
    return passages


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/multihop-rag"))
    parser.add_argument("--upstream-out", type=Path, default=Path("build/mhr_upstream"))
    args = parser.parse_args()
    articles = json.loads((args.src / "corpus.json").read_text(encoding="utf-8"))
    queries = json.loads((args.src / "MultiHopRAG.json").read_text(encoding="utf-8"))

    corpus, by_url, seen = [], {}, set()
    for article in articles:
        title = article["title"].strip()
        for text in chunk(article["body"]):
            if (title, text) in seen:
                continue
            seen.add((title, text))
            pid = str(len(corpus))
            corpus.append({"pid": pid, "title": title, "text": text})
            by_url.setdefault(article["url"], []).append(pid)
    normalised = {row["pid"]: norm(row["text"]) for row in corpus}

    questions, evidence_total, evidence_fallback, questions_with_fallback = [], 0, 0, 0
    for i, query in enumerate(queries):
        if query["question_type"] == "null_query":
            continue
        gold, any_fallback = [], False
        for evidence in query["evidence_list"]:
            evidence_total += 1
            pids = by_url[evidence["url"]]
            fact = norm(evidence["fact"])
            hits = [pid for pid in pids if fact in normalised[pid]]
            if not hits:
                evidence_fallback += 1
                any_fallback = True
                hits = pids
            item = hits[0] if len(hits) == 1 else hits
            if item not in gold:
                gold.append(item)
        questions_with_fallback += any_fallback
        questions.append({"qid": f"mhr-{i:04d}", "question": query["query"].strip(), "gold_pids": gold,
                          "type": query["question_type"]})

    counts = collections.Counter(q["type"] for q in questions)
    exact = {t: SAMPLE * n / len(questions) for t, n in counts.items()}
    quota = {t: int(v) for t, v in exact.items()}
    for t in sorted(exact, key=lambda t: (-(exact[t] - quota[t]), t))[: SAMPLE - sum(quota.values())]:
        quota[t] += 1
    rng = random.Random(SEED)
    chosen = set()
    for t in sorted(quota):
        chosen.update(rng.sample([q["qid"] for q in questions if q["type"] == t], quota[t]))
    sample = [q["qid"] for q in questions if q["qid"] in chosen]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "corpus.jsonl", corpus)
    write_jsonl(args.out / "questions.jsonl", questions)
    (args.out / "splits.json").write_text(json.dumps({"sample300": sample}, indent=1) + "\n")
    by_qid = {q["qid"]: q for q in questions}
    sample_fallback = sum(any(isinstance(g, list) and len(g) > 3 for g in by_qid[q]["gold_pids"]) for q in sample)
    lengths = [len(row["text"]) for row in corpus]
    manifest = {
        "source": f"yixuantt/MultiHopRAG@{REVISION}", "license": "ODC-BY",
        "source_sha256": {name: hashlib.sha256((args.src / name).read_bytes()).hexdigest()
                          for name in ("MultiHopRAG.json", "corpus.json")},
        "articles": len(articles), "passages": len(corpus), "max_passage_chars": max(lengths),
        "mean_passage_chars": round(sum(lengths) / len(lengths), 1),
        "questions_excluding_null": len(questions), "null_query_excluded": len(queries) - len(questions),
        "types": dict(counts), "sample_seed": SEED, "sample_quota": quota,
        "evidence_total": evidence_total, "evidence_not_matched": evidence_fallback,
        "evidence_not_matched_share": round(evidence_fallback / evidence_total, 4),
        "questions_with_unmatched_evidence": questions_with_fallback,
    }
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))

    # HippoRAG-style copy for the upstream entry point (format conversion only): the files are named after
    # a dataset the entry point accepts and are kept in their own directory with their own index cache.
    args.upstream_out.mkdir(parents=True, exist_ok=True)
    texts = {row["pid"]: row for row in corpus}
    upstream_questions = []
    for qid in sample:
        flat = [pid for item in by_qid[qid]["gold_pids"] for pid in ([item] if isinstance(item, str) else item)]
        upstream_questions.append({
            "id": qid, "question": by_qid[qid]["question"], "answerable": True,
            "paragraphs": [{"title": texts[pid]["title"], "paragraph_text": texts[pid]["text"], "is_supporting": True}
                           for pid in dict.fromkeys(flat)]})
    (args.upstream_out / "musique.json").write_text(json.dumps(upstream_questions, ensure_ascii=False))
    (args.upstream_out / "musique_corpus.json").write_text(
        json.dumps([{"title": row["title"], "text": row["text"]} for row in corpus], ensure_ascii=False))


if __name__ == "__main__":
    main()
