"""
Time Travel in LLMs: Tracing Data Contamination in Large Language Models (Guided Instruction)

Paper: https://arxiv.org/abs/2308.08493
Github Implementation:  https://github.com/shahriargolchin/time-travel-in-llms

Algorithm (per-instance):
Given a paragraph x:
  1. Sentence-split x; sample a split point p ∈ [min_p, max_p] of the sentence
     count. Let `first_piece` = sentences[:p], `second_piece` = sentences[p:].
  2. Build TWO prompts using `first_piece`:
        - GUIDED:  cites the source corpus (e.g. "DCLM", "peS2o") and asks
          the model to reproduce the SECOND PIECE EXACTLY as in the original.
        - GENERAL: asks the model to continue coherently, no source cited.
  3. Generate one greedy completion for each prompt → `guided_completion`,
     `general_completion`.
  4. Per-instance scores:
        rouge_guided   = ROUGE-L F1 (guided_completion,  second_piece)
        rouge_general  = ROUGE-L F1 (general_completion, second_piece)
        rouge_delta    = rouge_guided - rouge_general
        gpt4_icl       = P(Yes) from GPT-4 first-token logprobs on the paper's
                         few-shot ICL prompt with (second_piece, guided_completion)
  5. Return the chosen score_type as the contamination score (higher = more
     likely contaminated). Which variant wins on AUROC is tuned on dev.
"""

from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Callable

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle
from progress_utils import progress
from guided_instruction_helper.openai_judge import OpenAIJudge


# ─── tuned hyper-parameters (post-tuning; update via guided_instruction_tune.py) ─

# `score_type` ∈ {
#     "rouge_guided", "rouge_delta",
#     "bleurt_guided", "bleurt_delta",   (require BLEURT install)
#     "gpt4_icl",                         (requires OPENAI_API_KEY)
# }

MATCHED_CONFIG: dict[str, dict[str, Any]] = {
    "OLMo2-1B-Instruct":  {"score_type": "rouge_delta",  "min_p": 40, "max_p": 70},
    "OLMo2-7B-Instruct":  {"score_type": "rouge_delta",  "min_p": 40, "max_p": 70},
    "OLMo2-13B-Instruct": {"score_type": "rouge_guided", "min_p": 40, "max_p": 70},
    "OLMo2-32B-Instruct": {"score_type": "rouge_guided", "min_p": 40, "max_p": 70},
}

SHIFTED_CONFIG: dict[str, dict[str, Any]] = {
    "OLMo2-1B-Instruct":  {"score_type": "rouge_delta",  "min_p": 40, "max_p": 70},
    "OLMo2-7B-Instruct":  {"score_type": "rouge_delta",  "min_p": 40, "max_p": 70},
    "OLMo2-13B-Instruct": {"score_type": "rouge_guided", "min_p": 40, "max_p": 70},
    "OLMo2-32B-Instruct": {"score_type": "rouge_guided", "min_p": 40, "max_p": 70},
}

_FALLBACK_CONFIG = {"score_type": "rouge_guided", "min_p": 40, "max_p": 70}


def get_optimal_params(model_path: str, dataset_path: str) -> dict[str, Any]:
    model_name = os.path.basename(model_path.rstrip("/"))
    file_name  = os.path.basename(dataset_path).lower()
    config_dict = SHIFTED_CONFIG if "shifted" in file_name else MATCHED_CONFIG
    config = config_dict.get(model_name, _FALLBACK_CONFIG)
    print(f"[guided_instruction] score_type={config['score_type']} "
          f"min_p={config['min_p']} max_p={config['max_p']}")
    return config


# ─── source-name extraction (paper's `dataset_name` slot, adapted to OLMo) ───
_SOURCE_NAME_MAP: dict[str, str] = {
    "dclm":          "DCLM",
    "openwebmath":   "OpenWebMath",
    "pes2o":         "peS2o",
    "starcoder":     "StarCoder",
    "math-mix":      "OLMo2 math midtraining mix",
    "stackexchange": "Stack Exchange",
    "dpo":           "OLMo2 DPO training data",
    "rlvr":          "OLMo2 RLVR training data",
    "sft":           "OLMo2 SFT training data",
}
_DEFAULT_SOURCE_HINT = "the OLMo2 training corpus"


def extract_source_hint(dataset_path: str) -> str:
    """Walk path tokens; return first known source name. Fall back to generic."""
    parts = re.split(r"[/\\]", dataset_path.lower())
    for part in parts:
        if part in _SOURCE_NAME_MAP:
            return _SOURCE_NAME_MAP[part]
    return _DEFAULT_SOURCE_HINT


# ─── prompts  ─────────
_GUIDED_PROMPT = """INSTRUCTION:
You are provided with the FIRST PIECE of a passage from {source}.
Finish the SECOND PIECE of the passage as EXACTLY appeared in the original.
ONLY rely on the original form of the passage to finish the SECOND PIECE.

FIRST PIECE:
{first_piece}

SECOND PIECE:
"""

_GENERAL_PROMPT = """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a coherent passage.

FIRST PIECE:
{first_piece}

SECOND PIECE:
"""


def build_guided_prompt(first_piece: str, source: str) -> str:
    return _GUIDED_PROMPT.format(source=source, first_piece=first_piece)


def build_general_prompt(first_piece: str) -> str:
    return _GENERAL_PROMPT.format(first_piece=first_piece)


# ─── sentence tokeniser (NLTK punkt → regex fallback) ────────────────────────
_NLTK_AVAILABLE = False
try:
    import nltk  # type: ignore
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize  # type: ignore
    try:
        nltk.data.find("tokenizers/punkt")
        _NLTK_AVAILABLE = True
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            _NLTK_AVAILABLE = True
        except Exception:
            _NLTK_AVAILABLE = False
except ImportError:
    _NLTK_AVAILABLE = False

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z\"\'])")


def sent_tokenize_safe(text: str) -> list[str]:
    if _NLTK_AVAILABLE:
        try:
            return _nltk_sent_tokenize(text)
        except Exception:
            pass
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return parts if parts else [text]


# ─── random text split ─────────────

def _split_randomly(item_list: list[str], min_p: float, max_p: float, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    split_point = round(
        rng.uniform(min_p / 100.0 * len(item_list), max_p / 100.0 * len(item_list))
    )
    split_point = max(1, min(split_point, len(item_list) - 1)) if len(item_list) > 1 else 1
    first_piece  = " ".join(item_list[:split_point])
    second_piece = " ".join(item_list[split_point:])
    return first_piece, second_piece


def split_text_randomly(text: str, min_p: float = 40.0, max_p: float = 70.0, seed: int = 42) -> tuple[str, str]:
    sentences = sent_tokenize_safe(text)
    if len(sentences) > 1:
        return _split_randomly(sentences, min_p, max_p, seed)
    words = text.split(" ")
    return _split_randomly(words, min_p, max_p, seed)


# ─── ROUGE-L F1 ───────────────────────────────────────
_ROUGE_FN: Callable[[str, str], float] | None = None


def _build_rouge_fn() -> Callable[[str, str], float]:
    # (1) rouge_score
    try:
        from rouge_score import rouge_scorer  # type: ignore
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        def _fn(reference: str, candidate: str) -> float:
            if not reference or not candidate:
                return 0.0
            return float(scorer.score(reference, candidate)["rougeL"].fmeasure)
        return _fn
    except ImportError:
        pass
    # (2) HF evaluate
    try:
        import evaluate  # type: ignore
        _hf = evaluate.load("rouge")
        def _fn(reference: str, candidate: str) -> float:
            if not reference or not candidate:
                return 0.0
            out = _hf.compute(
                references=[reference], predictions=[candidate],
                use_aggregator=False, use_stemmer=True,
            )
            return float(out["rougeL"][0])
        return _fn
    except ImportError:
        pass
    # (3) pure-Python LCS-F1 fallback
    def _tokenise(s: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9]+", s.lower())

    def _lcs_length(a: list[str], b: list[str]) -> int:
        if not a or not b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        prev = [0] * (len(b) + 1)
        for x in a:
            curr = [0] * (len(b) + 1)
            for j, y in enumerate(b, start=1):
                if x == y:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[-1]

    def _fn(reference: str, candidate: str) -> float:
        ref = _tokenise(reference)
        cnd = _tokenise(candidate)
        if not ref or not cnd:
            return 0.0
        l = _lcs_length(ref, cnd)
        if l == 0:
            return 0.0
        precision = l / len(cnd)
        recall    = l / len(ref)
        return 2.0 * precision * recall / (precision + recall)
    return _fn


def compute_rouge_l(reference: str, candidate: str) -> float:
    global _ROUGE_FN
    if _ROUGE_FN is None:
        _ROUGE_FN = _build_rouge_fn()
    return _ROUGE_FN(reference, candidate)


# ─── BLEURT ───────────────────────────────────────
_BLEURT_FN: Callable[[str, str], float] | None = None
_BLEURT_TRIED = False


def compute_bleurt(reference: str, candidate: str, checkpoint: str | None = None) -> float:
    global _BLEURT_FN, _BLEURT_TRIED
    if _BLEURT_FN is None and not _BLEURT_TRIED:
        _BLEURT_TRIED = True
        ckpt = checkpoint or os.environ.get("BLEURT_CHECKPOINT")
        if not ckpt:
            return 0.0
        try:
            from bleurt import score as bleurt_score  # type: ignore
            scorer = bleurt_score.BleurtScorer(ckpt)
            def _fn(reference: str, candidate: str) -> float:
                if not reference or not candidate:
                    return 0.0
                return float(scorer.score(references=[reference], candidates=[candidate])[0])
            _BLEURT_FN = _fn
        except Exception:
            return 0.0
    if _BLEURT_FN is None:
        return 0.0
    return _BLEURT_FN(reference, candidate)


# ─── per-dataset generation (cached on model_bundle) ─────────────────────────

@dataclass
class _SamplingParams:
    max_tokens: int
    temperature: float
    batch_size: int
    progress_desc: str
    seed: int


def _completions_cache_key(
    dataset: DatasetBundle,
    min_p: float,
    max_p: float,
    max_new_tokens: int,
    max_tokens: int,
    seed: int,
    source: str,
) -> str:
    return (
        f"guided_inst_completions::{dataset.path}::{dataset.data_type}"
        f"::min_p={min_p}::max_p={max_p}::mnt={max_new_tokens}"
        f"::mt={max_tokens}::seed={seed}::src={source}"
    )


def compute_completions(
    model_bundle: ModelBundle,
    dataset: DatasetBundle,
    min_p: float = 40.0,
    max_p: float = 70.0,
    max_new_tokens: int = 256,
    max_tokens: int = 4096,
    batch_size: int = 1,
    seed: int = 42,
    source_hint: str | None = None,
) -> dict[str, Any]:
    if source_hint is None:
        source_hint = extract_source_hint(dataset.path)

    key = _completions_cache_key(
        dataset, min_p=min_p, max_p=max_p,
        max_new_tokens=max_new_tokens, max_tokens=max_tokens,
        seed=seed, source=source_hint,
    )
    cached = model_bundle.runtime_cache.get(key)
    if cached is not None:
        return cached

    prepared: list[dict[str, Any]] = []
    skipped = 0
    for idx, record in enumerate(dataset.records):
        text = extract_text(record)
        if not text:
            skipped += 1
            continue
        first_piece, second_piece = split_text_randomly(
            text, min_p=min_p, max_p=max_p, seed=seed,
        )
        if not first_piece.strip() or not second_piece.strip():
            skipped += 1
            continue
        prepared.append({
            "record_index": idx,
            "sample_id":    extract_sample_id(record, fallback=idx),
            "label":        record.get("label"),
            "first_piece":  first_piece,
            "second_piece": second_piece,
            "guided_prompt":  build_guided_prompt(first_piece, source_hint),
            "general_prompt": build_general_prompt(first_piece),
        })

    tokenizer = model_bundle.tokenizer
    prompt_token_budget = max(64, int(max_tokens) - int(max_new_tokens))

    def _truncate_prompt(prompt: str) -> str:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) <= prompt_token_budget:
            return prompt
        ids = ids[:prompt_token_budget]
        return tokenizer.decode(ids, skip_special_tokens=True)

    guided_prompts  = [_truncate_prompt(item["guided_prompt"])  for item in prepared]
    general_prompts = [_truncate_prompt(item["general_prompt"]) for item in prepared]

    adapter = model_bundle.llm_adapter

    sp_guided = _SamplingParams(
        max_tokens=int(max_new_tokens), temperature=0.0,
        batch_size=int(batch_size),
        progress_desc=f"guided gen [{dataset.name}]", seed=int(seed),
    )
    guided_outputs: list[dict[str, Any]] = []
    if guided_prompts:
        guided_outputs = adapter.generate(prompts=guided_prompts, sampling_params=sp_guided)

    sp_general = _SamplingParams(
        max_tokens=int(max_new_tokens), temperature=0.0,
        batch_size=int(batch_size),
        progress_desc=f"general gen [{dataset.name}]", seed=int(seed),
    )
    general_outputs: list[dict[str, Any]] = []
    if general_prompts:
        general_outputs = adapter.generate(prompts=general_prompts, sampling_params=sp_general)

    if len(guided_outputs) != len(prepared) or len(general_outputs) != len(prepared):
        raise RuntimeError(
            f"Generation count mismatch: prepared={len(prepared)} "
            f"guided={len(guided_outputs)} general={len(general_outputs)}"
        )

    out_samples: list[dict[str, Any]] = []
    for item, g_out, n_out in zip(prepared, guided_outputs, general_outputs):
        out_samples.append({
            "record_index":         item["record_index"],
            "sample_id":            item["sample_id"],
            "label":                item["label"],
            "first_piece":          item["first_piece"],
            "second_piece":         item["second_piece"],
            "guided_completion":    str(g_out.get("text", "")),
            "general_completion":   str(n_out.get("text", "")),
        })

    result = {
        "dataset":           dataset.name,
        "dataset_path":      dataset.path,
        "dataset_data_type": dataset.data_type,
        "min_p":             min_p,
        "max_p":             max_p,
        "max_new_tokens":    max_new_tokens,
        "max_tokens":        max_tokens,
        "seed":              seed,
        "source_hint":       source_hint,
        "skipped":           skipped,
        "samples":           out_samples,
    }
    model_bundle.runtime_cache[key] = result
    return result


# ─── GPT-4 ICL judge pass (cached on model_bundle + on disk) ─────────────────

def _icl_cache_key(
    completion_data: dict[str, Any],
    judge_model: str,
) -> str:
    return (
        f"guided_inst_icl::{completion_data['dataset_path']}"
        f"::min_p={completion_data['min_p']}"
        f"::max_p={completion_data['max_p']}"
        f"::mnt={completion_data['max_new_tokens']}"
        f"::seed={completion_data['seed']}"
        f"::src={completion_data['source_hint']}"
        f"::judge={judge_model}"
    )


def compute_icl_scores(
    model_bundle: ModelBundle,
    completion_data: dict[str, Any],
    judge_model: str = "gpt-4o-mini-2024-07-18",
    judge_cache_dir: str | None = ".openai_judge_cache",
    judge_workers: int = 8,
) -> dict[int, float]:
    """
    Run the GPT-4 ICL judge over (second_piece, guided_completion) for every
    cached record. Returns {record_index: P(Yes)}.

    Two caches:
      * model_bundle.runtime_cache (in-memory, per-process)
      * disk cache inside the OpenAIJudge (per (ref, cand, judge_model))
    """
    in_memory_key = _icl_cache_key(completion_data, judge_model)
    cached = model_bundle.runtime_cache.get(in_memory_key)
    if cached is not None:
        return cached

    judge = OpenAIJudge(
        model       = judge_model,
        cache_dir   = judge_cache_dir,
        num_workers = judge_workers,
    )

    samples = completion_data["samples"]
    pairs = [
        (item["record_index"], item["second_piece"], item["guided_completion"])
        for item in samples
    ]

    progress_iter = progress(
        range(len(pairs)),
        desc=f"icl judge [{completion_data['dataset']}]",
        unit="judgment",
        dynamic_ncols=True,
        leave=False,
    )
    progress_iter_iter = iter(progress_iter)

    def _on_progress() -> None:
        try:
            next(progress_iter_iter)
        except StopIteration:
            pass

    judge_results = judge.judge_pairs(pairs, on_progress=_on_progress)

    # Drain any leftover progress ticks (defensive).
    for _ in progress_iter_iter:
        pass

    scores: dict[int, float] = {
        rid: float(r.get("p_yes", float("nan")))
        for rid, r in judge_results.items()
    }
    model_bundle.runtime_cache[in_memory_key] = scores
    return scores


# ─── per-record scoring ─────────

VALID_SCORE_TYPES = (
    "rouge_guided", "rouge_general", "rouge_delta",
    "bleurt_guided", "bleurt_general", "bleurt_delta",
    "gpt4_icl",
)


def score_record(
    completion_record: dict[str, Any],
    score_type: str,
    icl_score: float | None = None,
    use_bleurt: bool = False,
    bleurt_checkpoint: str | None = None,
) -> dict[str, float]:
    """
    Compute all relevant metrics on one cached record and return a dict with
    the chosen `score_type` selected as `score`.

    For score_type == "gpt4_icl", `icl_score` (P(Yes) from the judge) MUST
    be supplied by the caller — the judge runs as a separate cached pass.
    """
    if score_type not in VALID_SCORE_TYPES:
        raise ValueError(
            f"Unknown score_type={score_type!r}. "
            f"Choose from {sorted(VALID_SCORE_TYPES)}."
        )

    ref = completion_record["second_piece"]
    g   = completion_record["guided_completion"]
    n   = completion_record["general_completion"]

    rouge_guided  = compute_rouge_l(ref, g)
    rouge_general = compute_rouge_l(ref, n)
    rouge_delta   = rouge_guided - rouge_general

    bleurt_guided  = float("nan")
    bleurt_general = float("nan")
    bleurt_delta   = float("nan")
    if use_bleurt or score_type.startswith("bleurt"):
        bleurt_guided  = compute_bleurt(ref, g, bleurt_checkpoint)
        bleurt_general = compute_bleurt(ref, n, bleurt_checkpoint)
        bleurt_delta   = bleurt_guided - bleurt_general

    gpt4_icl_value = float("nan") if icl_score is None else float(icl_score)

    table = {
        "rouge_guided":   rouge_guided,
        "rouge_general":  rouge_general,
        "rouge_delta":    rouge_delta,
        "bleurt_guided":  bleurt_guided,
        "bleurt_general": bleurt_general,
        "bleurt_delta":   bleurt_delta,
        "gpt4_icl":       gpt4_icl_value,
    }
    score = table[score_type]
    return {
        "rouge_guided":   float(rouge_guided),
        "rouge_general":  float(rouge_general),
        "rouge_delta":    float(rouge_delta),
        "bleurt_guided":  float(bleurt_guided),
        "bleurt_general": float(bleurt_general),
        "bleurt_delta":   float(bleurt_delta),
        "gpt4_icl":       float(gpt4_icl_value),
        "score":          float(score),
    }


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


# ─── BaseMethod ──────────────────────────────────────────────────────────────

class GuidedInstructionMethod(BaseMethod):
    """
    Guided-Instruction contamination method (Algorithms 1 & 2, kept
    per-instance so AUROC can be computed directly).

    Score convention: higher = more likely contaminated.

    Tuned per (model, matched/shifted) — read from MATCHED_CONFIG / SHIFTED_CONFIG
    via `get_optimal_params()`:
        score_type, min_p, max_p

    Fixed (paper defaults):
        seed = 42, temperature = 0 (greedy), max_new_tokens = 256
        judge_model = gpt-4o-mini-2024-07-18, judge_cache_dir = .openai_judge_cache
    """
    name = "guided_instruction"

    def __init__(
        self,
        max_new_tokens:    int  = 256,
        seed:              int  = 42,
        max_tokens:        int  = 4096,    # framework wildcard
        batch_size:        int  = 1,       # framework wildcard
        # BLEURT (optional)
        use_bleurt:        bool = False,
        bleurt_checkpoint: str | None = None,
        # GPT-4 ICL judge (used only when score_type == "gpt4_icl")
        judge_model:       str  = "gpt-4o-mini-2024-07-18",
        judge_cache_dir:   str | None = ".openai_judge_cache",
        judge_workers:     int  = 8,
    ):
        self.max_new_tokens    = int(max_new_tokens)
        self.seed              = int(seed)
        self.max_tokens        = int(max_tokens)
        self.batch_size        = int(batch_size)
        self.use_bleurt        = bool(use_bleurt)
        self.bleurt_checkpoint = bleurt_checkpoint
        self.judge_model       = str(judge_model)
        self.judge_cache_dir   = judge_cache_dir
        self.judge_workers     = int(judge_workers)

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        cfg        = get_optimal_params(model_bundle.model_path, dataset.path)
        score_type = str(cfg["score_type"])
        min_p      = float(cfg["min_p"])
        max_p      = float(cfg["max_p"])

        # 1. Generate (or fetch from cache) guided + general completions.
        comp_data = compute_completions(
            model_bundle    = model_bundle,
            dataset         = dataset,
            min_p           = min_p,
            max_p           = max_p,
            max_new_tokens  = self.max_new_tokens,
            max_tokens      = self.max_tokens,
            batch_size      = self.batch_size,
            seed            = self.seed,
        )

        # 2. If gpt4_icl is the chosen score, run the judge pass (cached).
        icl_scores: dict[int, float] = {}
        if score_type == "gpt4_icl":
            icl_scores = compute_icl_scores(
                model_bundle     = model_bundle,
                completion_data  = comp_data,
                judge_model      = self.judge_model,
                judge_cache_dir  = self.judge_cache_dir,
                judge_workers    = self.judge_workers,
            )

        # 3. Score every record (pure-Python on cached strings + judgments).
        out_samples: list[dict[str, Any]] = []
        score_values:        list[float] = []
        rouge_guided_values: list[float] = []
        rouge_general_values: list[float] = []
        gpt4_icl_values:     list[float] = []
        for item in comp_data["samples"]:
            scored = score_record(
                item,
                score_type        = score_type,
                icl_score         = icl_scores.get(item["record_index"]),
                use_bleurt        = self.use_bleurt,
                bleurt_checkpoint = self.bleurt_checkpoint,
            )
            score_values.append(scored["score"])
            rouge_guided_values.append(scored["rouge_guided"])
            rouge_general_values.append(scored["rouge_general"])
            gpt4_icl_values.append(scored["gpt4_icl"])

            out_samples.append({
                "record_index":       item["record_index"],
                "sample_id":          item["sample_id"],
                "label":              item["label"],
                "first_piece":        item["first_piece"],
                "second_piece":       item["second_piece"],
                "guided_completion":  item["guided_completion"],
                "general_completion": item["general_completion"],
                "rouge_guided":       scored["rouge_guided"],
                "rouge_general":      scored["rouge_general"],
                "rouge_delta":        scored["rouge_delta"],
                "bleurt_guided":      scored["bleurt_guided"],
                "bleurt_general":     scored["bleurt_general"],
                "bleurt_delta":       scored["bleurt_delta"],
                "gpt4_icl":           scored["gpt4_icl"],
                "score":              scored["score"],
            })

        return {
            "method":              self.name,
            "dataset":             dataset.name,
            "dataset_path":        dataset.path,
            "dataset_data_type":   dataset.data_type,
            "score_type_used":     score_type,
            "min_p_used":          min_p,
            "max_p_used":          max_p,
            "max_new_tokens":      self.max_new_tokens,
            "max_tokens":          self.max_tokens,
            "seed":                self.seed,
            "source_hint":         comp_data["source_hint"],
            "judge_model":         self.judge_model if score_type == "gpt4_icl" else None,
            "num_total_records":   len(dataset.records),
            "num_scored_records":  len(out_samples),
            "num_skipped_records": comp_data["skipped"],
            "summary": {
                "mean_score":         _safe_mean(score_values),
                "mean_rouge_guided":  _safe_mean(rouge_guided_values),
                "mean_rouge_general": _safe_mean(rouge_general_values),
                "mean_gpt4_icl":      _safe_mean(gpt4_icl_values),
            },
            "samples": out_samples,
        }


def build_method(
    max_new_tokens:    int  = 256,
    seed:              int  = 42,
    max_tokens:        int  = 4096,
    batch_size:        int  = 1,
    use_bleurt:        bool = False,
    bleurt_checkpoint: str | None = None,
    judge_model:       str  = "gpt-4o-mini-2024-07-18",
    judge_cache_dir:   str | None = ".openai_judge_cache",
    judge_workers:     int  = 8,
    **_,
) -> BaseMethod:
    return GuidedInstructionMethod(
        max_new_tokens    = max_new_tokens,
        seed              = seed,
        max_tokens        = max_tokens,
        batch_size        = batch_size,
        use_bleurt        = use_bleurt,
        bleurt_checkpoint = bleurt_checkpoint,
        judge_model       = judge_model,
        judge_cache_dir   = judge_cache_dir,
        judge_workers     = judge_workers,
    )