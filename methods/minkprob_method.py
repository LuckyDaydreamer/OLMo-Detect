'''
Detecting Pretraining Data from Large Language Models (MinK)
Paper: https://arxiv.org/pdf/2310.16789
Github Implementation: https://github.com/swj0419/detect-pretrain-code

Min-K%++: Improved Baseline for Detecting Pre-Training Data from Large Language Models (MinK++)
Paper: https://arxiv.org/pdf/2404.02936
Github Implementation: https://github.com/zjysteven/mink-plus-plus
'''

from __future__ import annotations

from base_method import BaseMethod
from data_utils import (
    DatasetBundle,
    extract_sample_id,
    extract_text,
)
from model_utils import ModelBundle
from progress_utils import progress
import math
import torch
import os


MATCHED_CONFIG = {
    "OLMo2-1B-Instruct": {"mink": 0.05, "minkpp": 0.7},
    "OLMo2-7B-Instruct": {"mink": 0.1, "minkpp": 0.5},
    "OLMo2-13B-Instruct": {"mink": 0.1, "minkpp": 0.5},
    "OLMo2-32B-Instruct": {"mink": 0.2, "minkpp": 0.5},
}

SHIFTED_CONFIG = {
    "OLMo2-1B-Instruct": {"mink": 0.4, "minkpp": 0.5}, 
    "OLMo2-7B-Instruct": {"mink": 0.3, "minkpp": 0.5},
    "OLMo2-13B-Instruct": {"mink": 0.2, "minkpp": 0.5},
    "OLMo2-32B-Instruct": {"mink": 0.4, "minkpp": 0.5}, 
}


def get_optimal_k(model_path: str, dataset_path: str) -> tuple[float, float]:
    model_name = os.path.basename(model_path.rstrip('/'))
    file_name = os.path.basename(dataset_path).lower()
    
    if "shifted" in file_name:
        config_dict = SHIFTED_CONFIG
    else:
        config_dict = MATCHED_CONFIG
    
    # Retrieve config for the specific model size
    config = config_dict.get(model_name, {"mink": 0.2, "minkpp": 0.2}) # Fallback to 0.2
    print("mink: {}, mink++: {}".format(config["mink"], config["minkpp"]))
    return config["mink"], config["minkpp"]


def mink_prob_score(full_logits, target_ids, k_percent):
    """
    MinK. 
    """
    if target_ids.numel() == 0:
        return float("nan")
    if not (0 < k_percent <= 1.0):
        raise ValueError("k_percent must be in (0, 1].")
    log_probs = torch.log_softmax(full_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    k = max(1, int(k_percent * token_log_probs.numel()))
    worst_log_probs, _ = torch.topk(token_log_probs, k, largest=False)
    mean_logprob = worst_log_probs.mean()
    return float(mean_logprob)


def mink_pp_score(full_logits, target_ids, k_percent):
    """
    MinK++.
    """
    if target_ids.numel() == 0:
        return float("nan")
    if not (0 < k_percent <= 1.0):
        raise ValueError("k_percent must be in (0, 1].")
    log_probs = torch.log_softmax(full_logits, dim=-1)
    probs = log_probs.exp()

    # The expectation of the next token log-prob under the model distribution
    mu = (probs * log_probs).sum(dim=-1)  # [T]

    # Get the SD
    second_moment = (probs * (log_probs ** 2)).sum(dim=-1)
    sigma = (second_moment - (mu ** 2)).clamp(min=1e-9).sqrt()

    # Target token log-probs
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_scores = (token_log_probs - mu) / (sigma + 1e-9)

    k = max(1, int(k_percent * token_scores.numel()))
    worst_scores, _ = torch.topk(token_scores, k, largest=False)
    return float(worst_scores.mean())


def _safe_mean(values: list[float]) -> float:
    finite_values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite_values:
        return float("nan")
    return sum(finite_values) / len(finite_values)


def _cache_key(dataset: DatasetBundle, k_mink: float, k_minkpp: float, batch_size: int) -> str:
    return (
        f"mink_scores::{dataset.path}::{dataset.data_type}"
        f"::records={len(dataset.records)}::k_m={k_mink}::k_pp={k_minkpp}::b={batch_size}"
    )


def compute_mink_scores(
    model_bundle: ModelBundle,
    dataset: DatasetBundle,
    k_mink: float = 0.2,
    k_minkpp: float = 0.2,
    batch_size: int = 3,
) -> dict:
    key = _cache_key(dataset, k_mink=k_mink, k_minkpp=k_minkpp, batch_size=batch_size)
    if key in model_bundle.runtime_cache:
        return model_bundle.runtime_cache[key]

    result = _compute_paragraph_scores(
        model_bundle=model_bundle,
        dataset=dataset,
        k_mink=k_mink,
        k_minkpp=k_minkpp,
        batch_size=batch_size,
    )

    model_bundle.runtime_cache[key] = result
    return result


def _compute_paragraph_scores(
    model_bundle: ModelBundle,
    dataset: DatasetBundle,
    k_mink: float,
    k_minkpp: float,
    batch_size: int,
) -> dict:
    prepared: list[tuple[int, str, str]] = []
    for idx, record in enumerate(dataset.records):
        text = extract_text(record)
        if text:
            sample_id = extract_sample_id(record, fallback=idx)
            prepared.append((idx, sample_id, text))

    if not prepared:
        return {
            "dataset": dataset.name,
            "samples": [],
            "summary": {},
        }
    
    samples: list[dict] = []
    min_k_logprob_values: list[float] = []
    min_kpp_values: list[float] = []

    for start in progress(
        range(0, len(prepared), batch_size),
        desc=f"minkprob [{dataset.name}] plain",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        chunk = prepared[start : start + batch_size]
        texts = [item[2] for item in chunk]

        full_logits, target_ids, _ = model_bundle.llm_adapter.score_plain_batch(
            texts=texts,
            batch_size=batch_size,
        )

        for i, (record_index, sample_id, text) in enumerate(chunk):
            logits = full_logits[i]
            ids = target_ids[i]
            min_k_logprob = float(mink_prob_score(logits, ids, k_percent=k_mink))
            min_kpp = float(mink_pp_score(logits, ids, k_percent=k_minkpp))

            min_k_logprob_values.append(min_k_logprob)
            min_kpp_values.append(min_kpp)

            samples.append({
                "record_index": record_index,
                "sample_id": sample_id,
                "text": text,
                "token_count": int(ids.numel()),
                "min_k_prob_mean_logprob": min_k_logprob,
                "min_k_pp_score": min_kpp,
            })

    return {
        "dataset": dataset.name,
        "dataset_data_type": dataset.data_type,
        "input_mode": "plain",
        "k_mink": k_mink,
        "k_minkpp": k_minkpp,
        "batch_size": batch_size,
        "num_total_records": len(dataset.records),
        "num_scored_records": len(samples),
        "num_skipped_records": len(dataset.records) - len(samples),
        "summary": {
            "mean_min_k_prob_logprob": _safe_mean(min_k_logprob_values),
            "mean_min_k_pp_score": _safe_mean(min_kpp_values),
        },
        "samples": samples,
    }



def _sample_base(sample: dict) -> dict:
    base = {"record_index": sample["record_index"], "sample_id": sample["sample_id"]}
    for key in ["question", "answer", "answer_token_count", "text", "token_count"]:
        if key in sample:
            base[key] = sample[key]
    return base


class MinkProbMethod(BaseMethod):
    name = "minkprob"

    def __init__(self, k_percent: float = 0.2, batch_size: int = 1):
        self.k_percent = k_percent
        self.batch_size = batch_size

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict:
        k_mink, k_minkpp = get_optimal_k(model_bundle.model_path, dataset.path)
        
        shared = compute_mink_scores(
            model_bundle=model_bundle,
            dataset=dataset,
            k_mink=k_mink,
            k_minkpp=k_minkpp,
            batch_size=self.batch_size,
        )

        samples = []
        for sample in shared["samples"]:
            row = _sample_base(sample)
            row["min_k_prob_mean_logprob"] = sample["min_k_prob_mean_logprob"]
            row["min_k_pp_score"] = sample["min_k_pp_score"]
            samples.append(row)

        return {
            "method": self.name,
            "dataset": shared["dataset"],
            "dataset_data_type": shared["dataset_data_type"],
            "input_mode": shared["input_mode"],
            "k_mink_used": k_mink,
            "k_minkpp_used": k_minkpp,
            "batch_size": shared["batch_size"],
            "num_total_records": shared["num_total_records"],
            "num_scored_records": shared["num_scored_records"],
            "summary": shared["summary"],
            "samples": samples,
        }


def build_method(k_percent: float = 0.2, batch_size: int = 1) -> BaseMethod:
    return MinkProbMethod(k_percent=k_percent, batch_size=batch_size)