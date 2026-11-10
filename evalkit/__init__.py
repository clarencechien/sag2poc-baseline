"""evalkit: retrieval metrics for SAG2POC (handoff §5.1 / §5.2).

Written from the metric definitions only; imports nothing from any upstream benchmark code.
"""
from .bootstrap import paired_bootstrap
from .data import load_corpus, load_questions, load_run, load_scores, load_split
from .metrics import K_LIST, aggregate, per_question, score_run

__all__ = ["K_LIST", "aggregate", "load_corpus", "load_questions", "load_run", "load_scores", "load_split",
           "paired_bootstrap", "per_question", "score_run"]
