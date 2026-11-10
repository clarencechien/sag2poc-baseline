"""Loaders for the normalised dataset layout (handoff §5.1).

data/<dataset>/corpus.jsonl     {"pid", "title", "text"}
data/<dataset>/questions.jsonl  {"qid", "question", "gold_pids", "type"}
                                (a gold_pids entry is a pid, or a list of pids meaning "any one of these")
data/<dataset>/splits.json      {"<split>": [qid, ...]}

A *run* file is JSONL with one line per question: {"qid", "retrieved": [pid, ...]} in rank order.
A *scores* file is JSONL with one line per question: {"qid", "R@1", ..., "AE@10"}.
"""
import json
from pathlib import Path


def _jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _dataset_dir(dataset, root="data"):
    path = Path(dataset)
    return path if path.is_dir() else Path(root) / dataset


def load_corpus(dataset, root="data"):
    """-> {pid: {"pid", "title", "text"}} in file order."""
    return {row["pid"]: row for row in _jsonl(_dataset_dir(dataset, root) / "corpus.jsonl")}


def load_questions(dataset, root="data"):
    """-> {qid: {"qid", "question", "gold_pids", "type"}} in file order."""
    rows = _jsonl(_dataset_dir(dataset, root) / "questions.jsonl")
    out = {row["qid"]: row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"duplicate qid in {dataset}")
    return out


def load_split(dataset, split, root="data"):
    """-> [qid, ...]; split "all" means every question in file order."""
    if split == "all":
        return list(load_questions(dataset, root))
    splits = json.loads((_dataset_dir(dataset, root) / "splits.json").read_text(encoding="utf-8"))
    if split not in splits:
        raise KeyError(f"{dataset} has no split {split!r}; available: {sorted(splits)}")
    return list(splits[split])


def load_run(path):
    """-> {qid: [pid, ...]}"""
    rows = _jsonl(path)
    out = {row["qid"]: list(row["retrieved"]) for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"duplicate qid in run {path}")
    return out


def load_scores(path):
    """-> {qid: {"R@1": ..., "AE@10": ...}}"""
    return {row["qid"]: {k: v for k, v in row.items() if k != "qid"} for row in _jsonl(path)}
