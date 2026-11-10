"""R@k and AE@k (handoff §5.2).

R@k   For one question: |gold ∩ first k de-duplicated results| / |gold|. Reported as the mean over questions.
AE@k  For one question: 1 if the first k de-duplicated results contain every gold item, else 0.
      Reported as the fraction of questions.

De-duplication keeps the first occurrence of each pid and happens before the cut at k.
A gold item is normally one pid. It may also be a list of pids, meaning "any one of these passages".
"""
K_LIST = (1, 2, 5, 10)


def dedupe(items):
    return list(dict.fromkeys(items))


def gold_items(gold):
    """Each gold item is one piece of evidence: a pid, or a list of pids any one of which counts
    (used when an evidence sentence cannot be pinned to a single passage, see MultiHop-RAG)."""
    items = []
    for item in gold:
        alternatives = frozenset([item] if isinstance(item, str) else item)
        if not alternatives:
            raise ValueError("empty gold item")
        if alternatives not in items:
            items.append(alternatives)
    return items


def per_question(gold, retrieved, k_list=K_LIST):
    items = gold_items(gold)
    if not items:
        raise ValueError("question has no gold evidence")
    ranked = dedupe(retrieved)
    out = {}
    for k in k_list:
        top = set(ranked[:k])
        found = sum(1 for alternatives in items if alternatives & top)
        out[f"R@{k}"] = found / len(items)
        out[f"AE@{k}"] = float(found == len(items))
    return out


def score_run(questions, run, qids=None, k_list=K_LIST):
    """questions: {qid: {"gold_pids": [...]}}; run: {qid: [pid, ...]} -> {qid: metrics}.

    Every requested qid must be present in the run: a missing question is an error, not a zero.
    """
    qids = list(questions) if qids is None else list(qids)
    missing = [q for q in qids if q not in run]
    if missing:
        raise KeyError(f"run is missing {len(missing)} question(s), e.g. {missing[:3]}")
    return {q: per_question(questions[q]["gold_pids"], run[q], k_list) for q in qids}


def aggregate(scores, qids=None, percent=True):
    """Mean of each metric over qids -> {"n": int, "R@1": ..., ...}."""
    qids = list(scores) if qids is None else list(qids)
    if not qids:
        raise ValueError("no questions to aggregate")
    keys = list(scores[qids[0]])
    scale = 100.0 if percent else 1.0
    return {"n": len(qids), **{k: scale * sum(scores[q][k] for q in qids) / len(qids) for k in keys}}
