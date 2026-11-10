"""R@k and AE@k (handoff §5.2).

R@k   For one question: |gold ∩ first k de-duplicated results| / |gold|. Reported as the mean over questions.
AE@k  For one question: 1 if the first k de-duplicated results contain every gold item, else 0.
      Reported as the fraction of questions.

De-duplication keeps the first occurrence of each pid and happens before the cut at k.
"""
K_LIST = (1, 2, 5, 10)


def dedupe(items):
    return list(dict.fromkeys(items))


def per_question(gold, retrieved, k_list=K_LIST):
    gold = set(gold)
    if not gold:
        raise ValueError("question has no gold evidence")
    ranked = dedupe(retrieved)
    out = {}
    for k in k_list:
        found = gold.intersection(ranked[:k])
        out[f"R@{k}"] = len(found) / len(gold)
        out[f"AE@{k}"] = float(found == gold)
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
