"""
Standard MIA baselines:

1. Perplexity (Loss): Mean Log-Likelihood 
2. Zlib: Sum Log-Likelihood / compressed length 
3. Lowercase: Mean LL difference (original - lowercase) 
"""

from __future__ import annotations

import zlib
import math
import torch
from typing import Any
from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle
from progress_utils import progress


def _safe_mean(values: list[float]) -> float:
    finite_values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite_values:
        return float("nan")
    return sum(finite_values) / len(finite_values)


class LossZlibLowercaseMethod(BaseMethod):
    name = "loss_zlib_lowercase"

    def __init__(self, batch_size: int = 3, max_tokens: int = 1024, **kwargs):
        self.batch_size = batch_size
        self.max_tokens = max_tokens

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        # -------------------------
        # 1. Prepare data
        # -------------------------
        prepared = []
        for idx, record in enumerate(dataset.records):
            sample_id = extract_sample_id(record, fallback=idx)
            text_content = extract_text(record)
            if text_content:
                prepared.append({
                    "id": sample_id,
                    "idx": idx,
                    "text": text_content
                })

        samples = []
        loss_values = []
        zlib_values = []
        lowercase_values = []

        # -------------------------
        # 2. Batch processing
        # -------------------------
        for start in progress(
            range(0, len(prepared), self.batch_size),
            desc=f"{self.name} [{dataset.name}]",
            unit="batch",
            leave=False,
        ):
            chunk = prepared[start: start + self.batch_size]

            # Interleave original + lowercase
            combined_texts = []
            for c in chunk:
                combined_texts.append(c["text"])
                combined_texts.append(c["text"].lower())

            full_logits, target_ids, _ = model_bundle.llm_adapter.score_plain_batch(
                texts=combined_texts,
                batch_size=len(combined_texts),
            )

            # -------------------------
            # 3. Compute metrics
            # -------------------------
            for i in range(len(chunk)):
                orig_logits = full_logits[i * 2]
                orig_ids = target_ids[i * 2]

                low_logits = full_logits[i * 2 + 1]
                low_ids = target_ids[i * 2 + 1]

                if orig_ids.numel() == 0 or low_ids.numel() == 0:
                    continue

                # --- Original LL ---
                o_log_probs = torch.log_softmax(orig_logits, dim=-1)
                o_token_lp = o_log_probs.gather(-1, orig_ids.unsqueeze(-1)).squeeze(-1)

                ll_sum_orig = float(o_token_lp.sum().item())
                mean_ll_orig = float(o_token_lp.mean().item())

                # --- Lowercase LL ---
                l_log_probs = torch.log_softmax(low_logits, dim=-1)
                l_token_lp = l_log_probs.gather(-1, low_ids.unsqueeze(-1)).squeeze(-1)

                mean_ll_low = float(l_token_lp.mean().item())

                # =========================
                # Metrics (ALL: higher = member)
                # =========================

                # 1. Loss (mean LL)
                loss_score = mean_ll_orig

                # 2. Zlib
                raw_text = chunk[i]["text"]
                z_bytes = len(zlib.compress(raw_text.encode("utf-8")))
                z_score = ll_sum_orig / z_bytes if z_bytes > 0 else float("nan")

                # 3. Lowercase (correct formulation)
                lowercase_score = mean_ll_orig - mean_ll_low

                # Collect
                loss_values.append(loss_score)
                zlib_values.append(z_score)
                lowercase_values.append(lowercase_score)

                samples.append({
                    "record_index": chunk[i]["idx"],
                    "sample_id": chunk[i]["id"],
                    "token_count": int(orig_ids.numel()),
                    "loss_score": loss_score,
                    "zlib_score": z_score,
                    "lowercase_score": lowercase_score,
                    "text": raw_text,
                })

        # -------------------------
        # 4. Return results
        # -------------------------
        return {
            "method": self.name,
            "dataset": dataset.name,
            "num_total_records": len(dataset.records),
            "num_scored_records": len(samples),
            "summary": {
                "mean_loss_score": _safe_mean(loss_values),
                "mean_zlib_score": _safe_mean(zlib_values),
                "mean_lowercase_score": _safe_mean(lowercase_values),
            },
            "samples": samples,
        }


def build_method(batch_size: int = 1, max_tokens: int = 1024, **kwargs) -> BaseMethod:
    return LossZlibLowercaseMethod(batch_size=batch_size, max_tokens=max_tokens)