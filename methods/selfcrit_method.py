"""
Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models (SelfCrit)
Paper: https://arxiv.org/pdf/2510.09259
Github Implementation:  https://github.com/yongding-tao/RL-Data-Contamination

Design:
Self-Critique detects RL-phase contamination by exploiting policy collapse:
after RLVR training, the model converges to a narrow high-reward reasoning path
for samples it was trained on.  When asked to produce a *different* solution
conditioned on its first answer, contaminated samples fail to deviate — their
token-level entropy sequences remain highly similar to the initial response.
Clean samples show greater divergence.

Detection score:
  score = penalized_cosine_similarity(E1, E2)
        = cosine(pad(E1), pad(E2)) × (min(|E1|, |E2|) / max(|E1|, |E2|))

where E1, E2 are token-level entropy sequences of the initial and critique
responses respectively.  Higher score → more likely contaminated (member).
"""

from __future__ import annotations

import copy
import math
import torch
import numpy as np
from typing import Any

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_selfcrit_question
from model_utils import ModelBundle
from progress_utils import progress


# ─── self-critique prompt ──
_CRITIQUE_INSTRUCTION = (
    "\nA possible answer is provided below (it may or may not be correct). "
    "Please provide a response that follows a different reasoning path or "
    "provides an alternative solution:\n---\n{response}\n---\n"
    "Please now provide your new, different response:"
)


# ─── sampling params wrapper ──
class _SamplingParams:
    def __init__(self, max_tokens: int, batch_size: int, progress_desc: str):
        self.max_tokens = max_tokens
        self.temperature = 0.0        # greedy — paper and code both require this
        self.batch_size = batch_size
        self.progress_desc = progress_desc
        self.seed = 42


# ─── penalized cosine similarity ─────────────────────────────────────────────
def _penalized_cosine_similarity(e1: list[float], e2: list[float]) -> float:
    """
    Zero-pads the shorter sequence to the length of the longer, then:
      score = cosine(pad_e1, pad_e2) × (min_len / max_len)
    """
    if not e1 or not e2:
        return 0.0
    len1, len2 = len(e1), len(e2)
    max_len = max(len1, len2)
    a = np.zeros(max_len, dtype=np.float64)
    b = np.zeros(max_len, dtype=np.float64)
    a[:len1] = e1
    b[:len2] = e2
    dot = float(np.dot(a, b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    cosine_sim = dot / (na * nb)
    length_penalty = min(len1, len2) / max_len
    return float(cosine_sim * length_penalty)


# ─── chat-template helpers ────────────────────────────────────────────────────
def _apply_chat_template(tokenizer, user_content: str) -> str:
    """
    Wrap user_content in a single-turn chat template, returning a decoded string
    that is suitable for the generate() path in HFGenerationAdapter.
    """
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _truncate_response(tokenizer, response_text: str, max_response_tokens: int) -> str:
    """
    Official code truncates the initial response before embedding it in the
    critique prompt when the model context window would otherwise overflow.
    """
    if max_response_tokens <= 0:
        return response_text
    ids = tokenizer.encode(response_text, add_special_tokens=False)
    if len(ids) <= max_response_tokens:
        return response_text
    truncated_ids = ids[:max_response_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=False)


# ─── SelfCritMethod ─────────────────────────────────────────────────────────────

class SelfCritMethod(BaseMethod):
    """
    Self-Critique contamination detector for RL post-training.

    Applicable stages: RLVR (primary), DPO (secondary).
    NOT applicable to: pretraining, midtraining, SFT.

    Score convention: HIGHER score = more likely contaminated.
    This matches the official code's get_direction() = 1.
    """
    name = "selfcrit"

    def __init__(
        self,
        batch_size:   int = 1,
        max_tokens:   int = 1024,
        max_model_len: int = 4096,
        **kwargs,
    ):
        self.batch_size    = batch_size
        self.max_tokens    = max_tokens
        # If the model has a known context window, store it for truncation check.
        # Pass explicitly; otherwise the method skips truncation (safe default).
        self.max_model_len = max_model_len

    # ── main run ──────────────────────────────────────────────────────────────
    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        tokenizer = model_bundle.tokenizer

        # ── 0. Prepare records ───────────────────────────────────────────────
        prepared: list[dict] = []
        for idx, record in enumerate(dataset.records):
            question = extract_selfcrit_question(record, dataset.data_type)
            if not question:
                continue
            prepared.append({
                "idx":      idx,
                "id":       extract_sample_id(record, fallback=idx),
                "question": question,
            })

        if not prepared:
            return self._empty_result(dataset, len(dataset.records))

        # ── 1. Phase 1 — initial greedy responses ────────────────────────────
        # Apply chat template: wrap each question as a user message.
        initial_prompts = [
            _apply_chat_template(tokenizer, item["question"])
            for item in prepared
        ]

        sp1 = _SamplingParams(
            max_tokens    = self.max_tokens,
            batch_size    = self.batch_size,
            progress_desc = f"selfcrit initial [{dataset.name}]",
        )
        initial_outputs = model_bundle.llm_adapter.generate(initial_prompts, sp1)
        # initial_outputs[i] = {"text": str, "generated_token_entropies": list[float]}

        # ── 2. Phase 2 — self-critique greedy responses ──────────────────────
        critique_prompts: list[str] = []
        for i, item in enumerate(prepared):
            out = initial_outputs[i]
            initial_text = out["text"]

            # Context-window safety: truncate initial response if needed
            # (mirrors generate_full_data.py: max_response_len = max_model_len - template_tokens - 50)
            if self.max_model_len is not None:
                template_len = len(tokenizer.encode(
                    _apply_chat_template(tokenizer, item["question"]),
                    add_special_tokens=False,
                ))
                # max_resp_toks = self.max_model_len - template_len - 50
                max_resp_toks = self.max_model_len - template_len - self.max_tokens - 50
                initial_text = _truncate_response(tokenizer, initial_text, max_resp_toks)

            critique_question = (
                item["question"]
                + _CRITIQUE_INSTRUCTION.format(response=initial_text)
            )
            critique_prompts.append(
                _apply_chat_template(tokenizer, critique_question)
            )

        sp2 = _SamplingParams(
            max_tokens    = self.max_tokens,
            batch_size    = self.batch_size,
            progress_desc = f"selfcrit critique [{dataset.name}]",
        )
        critique_outputs = model_bundle.llm_adapter.generate(critique_prompts, sp2)

        # ── 3. Compute scores ────────────────────────────────────────────────
        samples: list[dict] = []
        score_values: list[float] = []

        for i, item in enumerate(prepared):
            e1 = initial_outputs[i]["generated_token_entropies"]
            e2 = critique_outputs[i]["generated_token_entropies"]
            score = _penalized_cosine_similarity(e1, e2) if (e1 and e2) else float("nan")
            score_values.append(score)
            samples.append({
                "record_index":           item["idx"],
                "sample_id":              item["id"],
                "text":                   item["question"],
                "initial_response":       initial_outputs[i]["text"],
                "critique_response":      critique_outputs[i]["text"],
                "initial_entropy_len":    len(e1),
                "critique_entropy_len":   len(e2),
                "self_critique_score":    score,
            })

        finite_scores = [s for s in score_values if math.isfinite(s)]
        mean_score = sum(finite_scores) / len(finite_scores) if finite_scores else float("nan")

        return {
            "method":             self.name,
            "dataset":            dataset.name,
            "dataset_path":       dataset.path,
            "dataset_data_type":  dataset.data_type,
            "max_tokens":         self.max_tokens,
            "num_total_records":  len(dataset.records),
            "num_scored_records": len(samples),
            "num_skipped_records": len(dataset.records) - len(samples),
            "summary": {
                "mean_self_critique_score": mean_score,
            },
            "samples": samples,
        }

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _empty_result(dataset: DatasetBundle, total: int) -> dict[str, Any]:
        return {
            "method":             "selfcrit",
            "dataset":            dataset.name,
            "dataset_path":       dataset.path,
            "dataset_data_type":  dataset.data_type,
            "num_total_records":  total,
            "num_scored_records": 0,
            "num_skipped_records": total,
            "summary":            {},
            "samples":            [],
        }


def build_method(batch_size: int = 1, max_tokens: int = 1024, **kwargs) -> BaseMethod:
    return SelfCritMethod(batch_size=batch_size, max_tokens=max_tokens, **kwargs)