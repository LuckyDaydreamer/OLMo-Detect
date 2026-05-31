"""
Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method (DCPDD)
Paper: https://aclanthology.org/anthology-files/anthology-files/pdf/emnlp/2024.emnlp-main.300.pdf
Github Implementation: https://github.com/zhang-wei-chao/DC-PDD 
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from base_method import BaseMethod
from data_utils import (
    DatasetBundle,
    extract_sample_id,
    extract_text,
)
from model_utils import ModelBundle
from progress_utils import progress


DEFAULT_OLMO_MODEL_PATHS = [
    "olmo_models/OLMo2-1B-Instruct",
    "olmo_models/OLMo2-7B-Instruct",
    "olmo_models/OLMo2-13B-Instruct",
    "olmo_models/OLMo2-32B-Instruct",
]
FIXED_REFERENCE_JSON_PATH = Path(__file__).resolve().parent / "dcpdd_data" / "c4_token_occurrence_OLMo2-Instruct.json"


def _c4_arrow_files(c4_dir: str | Path) -> list[Path]:
    path = Path(c4_dir)
    return sorted(path.glob("*.arrow"))


def _output_path_for_model(output_dir: str | Path, model_name: str) -> Path:
    return Path(output_dir) / f"c4_token_occurrence_{model_name}.json"


def _safe_mean(values: list[float]) -> float:
    finite_values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite_values:
        return float("nan")
    return float(sum(finite_values) / len(finite_values))


# ─── token frequency building ───────────────────────

def build_token_frequency_for_tokenizer(
    tokenizer: Any,
    c4_dir: str | Path = "data/c4",
    max_tokens: int = 1024,
    c4_batch_size: int = 512,
) -> tuple[dict[str, int], int, int]:
    """
    Count token occurrences on C4.  Matches repo com_fre_dis.py:
      input_ids = tok.encode(text)[:max_tok]
      for token_id in input_ids: fre_dis[token_id] += 1

    Returns:
      token_frequency_map : {str(token_id): count}  (sparse, non-zero only)
      vocab_size           : tokenizer vocab size
      num_processed_examples : rows processed
    """
    arrow_files = _c4_arrow_files(c4_dir)
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in c4 directory: {c4_dir}")

    vocab_size = len(tokenizer)
    token_frequency = [0] * vocab_size
    num_processed_examples = 0

    for arrow_path in progress(
        arrow_files,
        desc="dcpdd c4 files",
        unit="file",
        dynamic_ncols=True,
    ):
        shard = Dataset.from_file(str(arrow_path))
        for batch in progress(
            shard.iter(batch_size=c4_batch_size),
            desc=f"dcpdd token freq [{arrow_path.name}]",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ):
            texts = [t for t in batch.get("text", []) if isinstance(t, str) and t]
            for text in texts:
                num_processed_examples += 1
                token_ids = tokenizer.encode(text)[:max_tokens]
                for token_id in token_ids:
                    if 0 <= token_id < vocab_size:
                        token_frequency[token_id] += 1

    token_frequency_map = {
        str(token_id): int(count)
        for token_id, count in enumerate(token_frequency)
        if count != 0
    }
    return token_frequency_map, vocab_size, num_processed_examples


def export_reference_frequency_for_models(
    model_paths: list[str],
    c4_dir: str | Path = "data/c4",
    output_dir: str | Path = "data",
    max_tokens: int = 1024,
    c4_batch_size: int = 512,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    frequency_maps: dict[str, dict[str, int]] = {}

    for model_path in model_paths:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_name = Path(model_path).name

        token_frequency_map, vocab_size, num_processed_examples = build_token_frequency_for_tokenizer(
            tokenizer=tokenizer,
            c4_dir=c4_dir,
            max_tokens=max_tokens,
            c4_batch_size=c4_batch_size,
        )

        output_path = _output_path_for_model(out_dir, model_name)
        payload = {
            "model_name":                   model_name,
            "model_path":                   model_path,
            "tokenizer_name_or_path":       getattr(tokenizer, "name_or_path", model_path),
            "c4_dir":                       str(c4_dir),
            "max_tokens_per_sample":        max_tokens,
            "vocab_size":                   vocab_size,
            "num_processed_examples":       num_processed_examples,
            "num_non_zero_token_ids":       len(token_frequency_map),
            "num_total_token_occurrences":  int(sum(token_frequency_map.values())),
            "token_frequency":              token_frequency_map,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        files.append({
            "model_name":                  model_name,
            "output_json_path":            str(output_path),
            "num_non_zero_token_ids":      len(token_frequency_map),
            "num_total_token_occurrences": int(sum(token_frequency_map.values())),
        })
        frequency_maps[model_name] = token_frequency_map

    base_model = files[0]["model_name"] if files else None
    base_map   = frequency_maps.get(base_model or "", {})

    per_model_equal_to_base: dict[str, bool] = {}
    all_equal = True
    for model_name, token_map in frequency_maps.items():
        is_equal = token_map == base_map
        per_model_equal_to_base[model_name] = is_equal
        all_equal = all_equal and is_equal

    comparison = {
        "base_model":                base_model,
        "all_equal":                 all_equal,
        "per_model_equal_to_base":   per_model_equal_to_base,
        "checked_model_count":       len(files),
    }

    comparison_path = out_dir / "c4_token_occurrence_olmo_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    return {
        "files":                  files,
        "comparison":             comparison,
        "comparison_json_path":   str(comparison_path),
    }


def export_reference_frequency_for_olmo_models(
    c4_dir:       str | Path = "data/c4",
    output_dir:   str | Path = "data",
    max_tokens:   int        = 1024,
    c4_batch_size: int       = 512,
) -> dict[str, Any]:
    return export_reference_frequency_for_models(
        model_paths   = DEFAULT_OLMO_MODEL_PATHS,
        c4_dir        = c4_dir,
        output_dir    = output_dir,
        max_tokens    = max_tokens,
        c4_batch_size = c4_batch_size,
    )


# ─── hardcoded alpha configs ───────────────

MATCHED_CONFIG = {
    "OLMo2-1B-Instruct":  {"alpha": 0.001},
    "OLMo2-7B-Instruct":  {"alpha": 0.2},
    "OLMo2-13B-Instruct": {"alpha": 0.05},
    "OLMo2-32B-Instruct": {"alpha": 0.2},
}

SHIFTED_CONFIG = {
    "OLMo2-1B-Instruct":  {"alpha": 0.5},
    "OLMo2-7B-Instruct":  {"alpha": 0.5},
    "OLMo2-13B-Instruct": {"alpha": 0.5},
    "OLMo2-32B-Instruct": {"alpha": 0.5},
}


def get_optimal_alpha(model_path: str, dataset_path: str) -> float:
    model_name = os.path.basename(model_path.rstrip("/"))
    file_name  = os.path.basename(dataset_path).lower()

    config_dict = SHIFTED_CONFIG if "shifted" in file_name else MATCHED_CONFIG
    config      = config_dict.get(model_name, {"alpha": 0.01})   # fallback
    alpha       = config["alpha"]
    print(f"[dcpdd] alpha: {alpha}")
    return alpha


# ─── DCPDDMethod ──────────────────────────────────────────────────────────────

class DCPDDMethod(BaseMethod):
    name = "dcpdd"

    def __init__(
        self,
        c4_dir:        str   = "data/c4",
        output_dir:    str   = "data",
        max_tokens:    int   = 1024,
        c4_batch_size: int   = 512,
        batch_size:    int   = 3,
        max_cha:       int   = 512,
        lang:          str   = "en",
        alpha:         float = 0.01,   # fallback; overridden by get_optimal_alpha
    ):
        self.c4_dir        = c4_dir
        self.output_dir    = output_dir
        self.max_tokens    = max_tokens
        self.c4_batch_size = c4_batch_size
        self.batch_size    = batch_size
        self.max_cha       = max_cha
        self.lang          = lang
        self.alpha         = alpha

    def _reference_path(self, model_name: str) -> Path:
        return FIXED_REFERENCE_JSON_PATH

    def _load_or_create_reference_payload(
        self, model_bundle: ModelBundle
    ) -> dict[str, Any]:
        ref_path = self._reference_path(model_bundle.model_name)
        if ref_path.exists():
            with ref_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and isinstance(payload.get("token_frequency"), dict):
                return payload

        token_frequency_map, vocab_size, num_processed_examples = \
            build_token_frequency_for_tokenizer(
                tokenizer   = model_bundle.tokenizer,
                c4_dir      = self.c4_dir,
                max_tokens  = self.max_tokens,
                c4_batch_size = self.c4_batch_size,
            )
        payload = {
            "model_name":                  model_bundle.model_name,
            "model_path":                  model_bundle.model_path,
            "tokenizer_name_or_path":      getattr(model_bundle.tokenizer, "name_or_path",
                                                   model_bundle.model_path),
            "c4_dir":                      str(self.c4_dir),
            "max_tokens_per_sample":       self.max_tokens,
            "vocab_size":                  vocab_size,
            "num_processed_examples":      num_processed_examples,
            "num_non_zero_token_ids":      len(token_frequency_map),
            "num_total_token_occurrences": int(sum(token_frequency_map.values())),
            "token_frequency":             token_frequency_map,
        }
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with ref_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    @staticmethod
    def _build_smoothed_freq_array(
        token_frequency: dict[str, int],
        vocab_size:      int,
    ) -> np.ndarray:
        """
        The sparse dict is expanded to a full dense array first, then Laplace
        smoothing is applied once.  Result is a (vocab_size,) float64 array of
        smoothed token probabilities.
        """
        fre_dis_npy = np.zeros(vocab_size, dtype=np.float64)
        for token_id_str, count in token_frequency.items():
            idx = int(token_id_str)
            if 0 <= idx < vocab_size:
                fre_dis_npy[idx] = float(count)

        # Laplace smoothing — identical to repo formula
        total       = fre_dis_npy.sum()
        fre_dis_smo = (fre_dis_npy + 1.0) / (total + float(vocab_size))
        return fre_dis_smo   # (vocab_size,) float64, all entries > 0

    def _prepare_text(self, record: dict[str, Any], data_type: str) -> str | None:
        text = extract_text(record)
        if not text:
            return None
        # if self.lang == "cn":
        #     import jieba   # type: ignore
        #     text = "".join(jieba.lcut(text)[: self.max_cha])
        # else:
        #     text = " ".join(text.split()[: self.max_cha])
        return text.strip() or None

    def _compute_dc_pdd(
        self,
        input_ids:    list[int],
        tar_prob:     list[float],
        fre_dis_smo:  np.ndarray,
        alpha:        float | None = None,   # overrides self.alpha when set
    ) -> float:
        """
        Matches repo cal_met():
            probs   = np.exp(tar_prob)
            indexes = first occurrence of each token id
            x_pro   = probs[indexes]
            x_fre   = fre_dis_smo[input_ids[indexes]]   ← direct dense indexing
            ce      = x_pro * log(1 / x_fre)
            ce[ce > a] = a
            score   = -np.mean(ce)                       ← HIGHER for members

        Note: repo uses a list for seen-token tracking (O(n²)); we use a set (O(n)).
        Functionally identical — this is an efficiency improvement, not a change.
        """
        n = min(len(input_ids), len(tar_prob))
        if n <= 0:
            return float("nan")

        ids_arr  = np.array(input_ids[:n], dtype=np.int64)
        tar_arr  = np.array(tar_prob[:n],  dtype=np.float64)
        probs    = np.exp(tar_arr)   # convert log-probs → probs

        # First-occurrence indices (deduplicate token ids, keep first position)
        first_indexes: list[int] = []
        seen: set[int] = set()
        for idx, token_id in enumerate(ids_arr.tolist()):
            if token_id not in seen:
                seen.add(token_id)
                first_indexes.append(idx)

        if not first_indexes:
            return float("nan")

        fi    = np.array(first_indexes, dtype=np.int64)
        a     = alpha if alpha is not None else self.alpha

        x_pro = probs[fi]                   # model probabilities at first-occ positions
        x_fre = fre_dis_smo[ids_arr[fi]]   # ← FIX 1: direct dense numpy indexing

        ce = x_pro * np.log(1.0 / x_fre)
        ce = np.minimum(ce, a)              # equivalent to ce[ce > a] = a

        # Repo stores pred["DC-PDD"] = -np.mean(ce), but sweep() calls
        # roc_curve(labels, -score) which effectively uses +np.mean(ce).
        # Our AUC script calls roc_auc_score(y_true, scores) with no negation,
        # so we must return +mean(ce) — higher = more likely contaminated (member).
        return float(np.mean(ce))

    def run(
        self, model_bundle: ModelBundle, dataset: DatasetBundle
    ) -> dict[str, Any]:
        reference_payload = self._load_or_create_reference_payload(model_bundle)
        token_frequency   = reference_payload.get("token_frequency", {})
        vocab_size        = int(reference_payload.get("vocab_size", len(model_bundle.tokenizer)))

        # Build smoothed dense array ONCE
        fre_dis_smo = self._build_smoothed_freq_array(token_frequency, vocab_size)

        # Resolve alpha from hardcoded config
        alpha = get_optimal_alpha(model_bundle.model_path, dataset.path)

        prepared: list[dict[str, Any]] = []
        for idx, record in enumerate(dataset.records):
            text = self._prepare_text(record, dataset.data_type)
            if not text:
                continue
            prepared.append({
                "record_index": idx,
                "sample_id":    extract_sample_id(record, fallback=idx),
                "label":        record.get("label"),
                "text":         text,
            })

        if not prepared:
            return {
                "method":                          self.name,
                "dataset":                         dataset.name,
                "dataset_path":                    dataset.path,
                "dataset_data_type":               dataset.data_type,
                "alpha":                           self.alpha,
                "reference_frequency_json_path":   str(self._reference_path(model_bundle.model_name)),
                "num_total_records":               len(dataset.records),
                "num_scored_records":              0,
                "num_skipped_records":             len(dataset.records),
                "summary":                         {},
                "samples":                         [],
            }

        texts     = [p["text"] for p in prepared]
        tar_stats = model_bundle.llm_adapter.token_logprob_distribution_batch(
            texts         = texts,
            batch_size    = self.batch_size,
            max_tokens    = self.max_tokens,
            progress_desc = f"dcpdd tar prob [{dataset.name}]",
        )

        def _stats_or_empty(stats: list[dict[str, Any]], i: int) -> dict[str, Any]:
            if i < len(stats):
                return stats[i]
            return {"input_ids": [], "token_log_probs": [], "mu": [], "sigma": []}

        samples:        list[dict[str, Any]] = []
        dc_pdd_values:  list[float]          = []

        for i, item in enumerate(
            progress(
                prepared,
                desc         = f"dcpdd score [{dataset.name}]",
                unit         = "sample",
                dynamic_ncols = True,
                leave        = False,
            )
        ):
            tar        = _stats_or_empty(tar_stats, i)
            input_ids  = [int(x) for x in tar.get("input_ids", [])]
            tar_prob   = [float(x) for x in tar.get("token_log_probs", [])]

            dc_pdd_score = self._compute_dc_pdd(
                input_ids   = input_ids,
                tar_prob    = tar_prob,
                fre_dis_smo = fre_dis_smo,
                alpha       = alpha,
            )
            dc_pdd_values.append(dc_pdd_score)

            samples.append({
                "record_index": item["record_index"],
                "sample_id":    item["sample_id"],
                "label":        item["label"],
                "text":         item["text"],
                "token_count":  min(len(input_ids), len(tar_prob)),
                "dc_pdd_score": dc_pdd_score,
            })

        return {
            "method":                          self.name,
            "dataset":                         dataset.name,
            "dataset_path":                    dataset.path,
            "dataset_data_type":               dataset.data_type,
            "alpha_used":                      alpha,
            "max_tokens":                      self.max_tokens,
            "reference_frequency_json_path":   str(self._reference_path(model_bundle.model_name)),
            "num_total_records":               len(dataset.records),
            "num_scored_records":              len(samples),
            "num_skipped_records":             len(dataset.records) - len(samples),
            "summary": {
                "mean_dc_pdd_score": _safe_mean(dc_pdd_values),
            },
            "samples": samples,
        }


def build_method(
    c4_dir:       str = "data/c4",
    output_dir:   str = "data",
    max_tokens:   int = 1024,
    c4_batch_size: int = 512,
    batch_size:   int = 3,
) -> BaseMethod:
    return DCPDDMethod(
        c4_dir        = c4_dir,
        output_dir    = output_dir,
        max_tokens    = max_tokens,
        c4_batch_size = c4_batch_size,
        batch_size    = batch_size,
    )


# ─── alpha tuning helpers ──────────────

ALPHA_SEARCH_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def compute_dc_pdd_scores_from_cache(
    cached_samples: list[dict],   # list of {"input_ids": [...], "tar_prob": [...]}
    fre_dis_smo:    np.ndarray,
    alpha:          float,
) -> list[float]:
    """
    Compute dc_pdd scores for a given alpha over pre-cached
    (input_ids, tar_prob) pairs.  Zero model calls.
    """
    scores = []
    for s in cached_samples:
        input_ids = s["input_ids"]
        tar_prob  = s["tar_prob"]
        n = min(len(input_ids), len(tar_prob))
        if n <= 0:
            scores.append(float("nan"))
            continue
        ids_arr = np.array(input_ids[:n], dtype=np.int64)
        probs   = np.exp(np.array(tar_prob[:n], dtype=np.float64))

        first_indexes: list[int] = []
        seen: set[int] = set()
        for idx, token_id in enumerate(ids_arr.tolist()):
            if token_id not in seen:
                seen.add(token_id)
                first_indexes.append(idx)

        if not first_indexes:
            scores.append(float("nan"))
            continue

        fi    = np.array(first_indexes, dtype=np.int64)
        x_pro = probs[fi]
        x_fre = fre_dis_smo[ids_arr[fi]]
        ce    = x_pro * np.log(1.0 / x_fre)
        ce    = np.minimum(ce, alpha)
        scores.append(float(np.mean(ce)))
    return scores


def score_dataset_for_tuning(
    model_bundle: "ModelBundle",
    dataset_path: str,
    fre_dis_smo:  np.ndarray,
    max_cha:      int   = 512,
    lang:         str   = "en",
    max_tokens:   int   = 1024,
    batch_size:   int   = 3,
) -> list[dict]:
    """
    Run model ONCE on a dataset; return list of
    {"input_ids": [...], "tar_prob": [...]} for alpha sweeping.
    """
    from data_utils import load_dataset as _load_dataset, extract_text as _extract_text

    dataset = _load_dataset(dataset_path)
    texts = []
    for record in dataset.records:
        text = _extract_text(record)
        if not text:
            continue
        if lang == "cn":
            import jieba
            text = "".join(jieba.lcut(text)[:max_cha])
        else:
            text = " ".join(text.split()[:max_cha])
        text = text.strip()
        if text:
            texts.append(text)

    print(f"  Scoring {len(texts)} samples from {dataset_path}")
    tar_stats = model_bundle.llm_adapter.token_logprob_distribution_batch(
        texts=texts, batch_size=batch_size, max_tokens=max_tokens
    )

    cached = []
    for s in tar_stats:
        cached.append({
            "input_ids": [int(x) for x in s.get("input_ids", [])],
            "tar_prob":  [float(x) for x in s.get("token_log_probs", [])],
        })
    return cached
