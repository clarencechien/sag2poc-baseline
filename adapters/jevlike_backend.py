"""jevlike backend for the upstream local-rerank interface (handoff §6.2).

Reads the first-token probability of an option letter from Gemma 4 26B-A4B served by SGLang:

  score       = P(A) / (P(A) + P(B))      A = yes, B = no
  option mass = P(A) + P(B)               ~1.0 when the template is healthy

One candidate per forward pass (never packed). The prompt is ordered

  template head -> original question -> evidence so far -> candidate passage -> question and options

so that every candidate of a scoring round shares one token prefix and SGLang's radix cache can reuse it.
The template is jevlike's GEMMA4_TEMPLATE_NOTHINK_BOS, unchanged: literal <bos>, thinking disabled through the
empty thought channel. Prompts are sent as input_ids.

The class mirrors what the upstream LocalScorer needs from a reranker: `config`, `lock`, `model`,
and `rank(query, documents, instruction) -> {"model", "results", "usage"}`.
"""
import hashlib
import json
import math
import os
import threading
import time
import urllib.request
from pathlib import Path

# --- jevlike decide/prompt.py, copied verbatim (the template must not change) ---------------------------------------
SYSTEM_EN = "You are a decision engine. Answer with exactly one letter."
SYS_MARK, USR_MARK = "§SYSMARK§", "§USRMARK§"
GEMMA4_TEMPLATE_NOTHINK = ("<|turn>system\n" + SYS_MARK + "<turn|>\n<|turn>user\n" + USR_MARK
                           + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>")
GEMMA4_TEMPLATE_NOTHINK_BOS = "<bos>" + GEMMA4_TEMPLATE_NOTHINK
# --------------------------------------------------------------------------------------------------------------------

LETTERS = ("A", "B")
OPTIONS = "A. Yes\nB. No"
ANSWER_LINE = "Answer with the letter of the correct option only."


def build_user_text(state: str, document: str, instruction: str) -> str:
    """New question type "evidence": state first, candidate next, the affirmative question and options last."""
    return f"[State]\n{state}\n\ncandidate:\n{document}\n\n[Question] {instruction}\n{OPTIONS}\n\n{ANSWER_LINE}"


def build_prompt(state: str, document: str, instruction: str) -> str:
    return (GEMMA4_TEMPLATE_NOTHINK_BOS.replace(SYS_MARK, SYSTEM_EN)
            .replace(USR_MARK, build_user_text(state, document, instruction)))


class _NoLock:
    """The model lives in the SGLang server, which batches concurrent requests itself."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class JevlikeReranker:
    model = "jevlike"
    backend = "sglang"

    def __init__(self, path, base_url=None, *, batch_size=None, context_length=None, log_path=None,
                 timeout=900):
        from tokenizers import Tokenizer

        self.path = Path(path).resolve(strict=True)
        self.base_url = (base_url or os.environ.get("JEVLIKE_URL") or "http://127.0.0.1:30000").rstrip("/")
        self.batch_size = int(batch_size or os.environ.get("JEVLIKE_BATCH", 64))
        self.context_length = int(context_length or os.environ.get("JEVLIKE_CONTEXT", 16384))
        self.timeout = timeout
        self.lock = _NoLock()
        self.device = "sglang"
        self.tokenizer = Tokenizer.from_file(str(self.path / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.letter_ids = {}
        for letter in LETTERS:  # label self-check: one token, round-trips to itself
            ids = self.tokenizer.encode(letter, add_special_tokens=False).ids
            if len(ids) != 1 or self.tokenizer.decode(ids) != letter:
                raise ValueError(f"option letter {letter!r} is not a single round-trippable token: {ids}")
            self.letter_ids[letter] = ids[0]
        probe = self.encode(build_prompt("state", "document", "question?"))
        if self.bos_id is None or probe[0] != self.bos_id or probe.count(self.bos_id) != 1:
            raise ValueError("prompt does not start with exactly one <bos>")
        self._log_path = Path(log_path or os.environ.get("JEVLIKE_LOG", "")) if (log_path or os.environ.get("JEVLIKE_LOG")) else None
        self._log_lock = threading.Lock()
        tokenizer_sha = hashlib.sha256((self.path / "tokenizer.json").read_bytes()).hexdigest()
        self.config = {
            "model": "jevlike",
            "backend": "sglang",
            "served_model": self.path.name,
            "readout": "first-token option letter, A=yes B=no, score=P(A)/(P(A)+P(B))",
            "template": "GEMMA4_TEMPLATE_NOTHINK_BOS",
            "template_sha256": hashlib.sha256(GEMMA4_TEMPLATE_NOTHINK_BOS.encode()).hexdigest(),
            "user_text_sha256": hashlib.sha256(build_user_text("{state}", "{document}", "{instruction}").encode()).hexdigest(),
            "tokenizer_sha256": tokenizer_sha,
            "letter_token_ids": self.letter_ids,
            "one_candidate_per_forward_pass": True,
            "source_truncation": False,
        }

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _generate(self, batch: list[list[int]]) -> list[dict]:
        ids = [self.letter_ids[letter] for letter in LETTERS]
        body = {
            "input_ids": batch,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": 5,
            "return_text_in_logprobs": True,
            "token_ids_logprob": [ids] * len(batch),
        }
        request = urllib.request.Request(f"{self.base_url}/generate", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read())
        return data if isinstance(data, list) else [data]

    def _read(self, item: dict) -> dict:
        meta = item.get("meta_info", {})
        exact = (meta.get("output_token_ids_logprobs") or [[]])[0]
        by_id = {int(entry[1]): float(entry[0]) for entry in exact}
        p = {letter: math.exp(by_id[tid]) if tid in by_id else 0.0 for letter, tid in self.letter_ids.items()}
        mass = p["A"] + p["B"]
        top = (meta.get("output_top_logprobs") or [[]])[0]
        return {
            "logp_a": by_id.get(self.letter_ids["A"]),
            "logp_b": by_id.get(self.letter_ids["B"]),
            "p_yes": p["A"] / mass if mass > 0 else 0.0,
            "option_mass": mass,
            "prompt_tokens": int(meta.get("prompt_tokens") or 0),
            "cached_tokens": int(meta.get("cached_tokens") or 0),
            "first_token": (top[0][2] if top and len(top[0]) > 2 else item.get("text")),
        }

    def rank(self, query: str, documents: list[str], instruction: str) -> dict:
        started = time.perf_counter()
        prompts = [self.encode(build_prompt(query, document, instruction)) for document in documents]
        longest = max(map(len, prompts), default=0)
        if longest + 1 > self.context_length:
            raise ValueError(
                f"jevlike prompt exceeds the model context ({longest} tokens); source was not truncated")
        reads: list[dict] = []
        if prompts:
            # Warm the shared prefix with one candidate, then send the rest in batches.
            reads.extend(self._read(item) for item in self._generate(prompts[:1]))
            for start in range(1, len(prompts), self.batch_size):
                reads.extend(self._read(item) for item in self._generate(prompts[start:start + self.batch_size]))
        if len(reads) != len(documents):
            raise ValueError("SGLang returned a different number of results than candidates")
        seconds = time.perf_counter() - started
        rows = [{"index": i, "relevance_score": read["p_yes"], "option_mass": read["option_mass"],
                 "logp_a": read["logp_a"], "logp_b": read["logp_b"],
                 "prompt_tokens": read["prompt_tokens"], "cached_tokens": read["cached_tokens"],
                 "first_token": read["first_token"]} for i, read in enumerate(reads)]
        usage = {
            "input_tokens": sum(read["prompt_tokens"] for read in reads),
            "output_tokens": len(reads),
            "cached_input_tokens": sum(read["cached_tokens"] for read in reads),
            "option_mass_min": min((read["option_mass"] for read in reads), default=None),
            "option_mass_below_0.5": sum(read["option_mass"] < 0.5 for read in reads),
            "judge_seconds": round(seconds, 3),
        }
        if self._log_path is not None:
            record = {
                "state_sha1": hashlib.sha1(query.encode()).hexdigest(),
                "state_chars": len(query),
                "seconds": round(seconds, 3),
                "docs_sha1": [hashlib.sha1(document.encode()).hexdigest() for document in documents],
                "logp_a": [read["logp_a"] for read in reads],
                "logp_b": [read["logp_b"] for read in reads],
                "p_yes": [read["p_yes"] for read in reads],
                "option_mass": [round(read["option_mass"], 6) for read in reads],
                "prompt_tokens": [read["prompt_tokens"] for read in reads],
                "cached_tokens": [read["cached_tokens"] for read in reads],
            }
            try:
                state = json.loads(query)
                record["original_question"] = state["original_question"]
                record["evidence_count"] = len(state["previously_retrieved_evidence"])
                record["evidence_has_scores"] = any(
                    "previous_jev_score" in item for item in state["previously_retrieved_evidence"])
            except (ValueError, KeyError, TypeError):
                record["original_question"] = query
                record["evidence_count"] = 0
            with self._log_lock, self._log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"model": "local/jevlike/sglang", "results": rows, "usage": usage}
