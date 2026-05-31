from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


QUESTION_PATTERN = re.compile(r"Question:\s*(.*?)(?:\nReasoning:|\nAnswer:|$)", re.DOTALL)
ANSWER_PATTERN = re.compile(r"Answer:\s*(.*)$", re.DOTALL)

DATA_TYPE_ALIASES = {
    "qa":        "qa",
    "paragraph": "paragraph",
    "auto":      "auto",
    "dpo":       "dpo",
}

# Modified priority: check for chosen/rejected keywords first to treat them as paragraphs
DATASET_TYPE_BY_KEYWORD = {
    "benchmark_dev": "paragraph",
    "pretraining":  "paragraph",
    "midtraining":  "paragraph",
    "chosen":       "paragraph",
    "rejected":     "paragraph",
    "dpo":          "dpo",
    "posttraining": "paragraph",
}


@dataclass
class DatasetBundle:
    path: str
    name: str
    data_type: str
    records: list[dict[str, Any]]


def _dataset_name_from_path(path: str) -> str:
    stem = Path(path).stem
    # Result directories are named after the dataset stem. Drop a trailing
    # "_eval" so output dirs are e.g. "<domain>" rather than "<domain>_eval".
    if stem.endswith("_eval"):
        stem = stem[: -len("_eval")]
    return stem


def normalize_data_type(data_type: str | None) -> str:
    if data_type is None:
        return "auto"
    key = data_type.strip().lower()
    if not key:
        return "auto"
    return DATA_TYPE_ALIASES.get(key, key)


def is_paragraph_data_type(data_type: str | None) -> bool:
    return normalize_data_type(data_type) == "paragraph"


def is_qa_data_type(data_type: str | None) -> bool:
    return normalize_data_type(data_type) == "qa"


def is_dpo_data_type(data_type: str | None) -> bool:
    return normalize_data_type(data_type) == "dpo"


def infer_dataset_data_type(path: str) -> str:
    lowered = path.lower()
    for keyword, data_type in DATASET_TYPE_BY_KEYWORD.items():
        if keyword in lowered:
            return data_type
    raise ValueError(
        f"No hardcoded data type rule for dataset path: {path}. "
        "Add an entry to DATASET_TYPE_BY_KEYWORD in data_utils.py."
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_num}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)
            else:
                records.append({"value": value})
    return records


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if isinstance(value, list):
        items = value
        records: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                records.append(item)
            else:
                records.append({"value": item})
        return records
    if isinstance(value, dict):
        return [value]
    return [{"value": value}]


def load_dataset(path: str) -> DatasetBundle:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    data_type = infer_dataset_data_type(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        records = _load_jsonl(p)
    elif suffix == ".json":
        records = _load_json(p)
    else:
        try:
            records = _load_jsonl(p)
        except ValueError:
            records = _load_json(p)
    return DatasetBundle(
        path=str(p),
        name=_dataset_name_from_path(str(p)),
        data_type=normalize_data_type(data_type),
        records=records,
    )


def load_datasets(paths: list[str]) -> list[DatasetBundle]:
    return [load_dataset(path) for path in paths]


def extract_sample_id(record: dict[str, Any], fallback: int) -> str:
    raw_id = record.get("id")
    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()
    return f"sample_{fallback}"


def _from_metadata(record: dict[str, Any], key: str) -> str | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def extract_text(record: dict[str, Any]) -> str | None:
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    value = record.get("value")
    if isinstance(value, str) and value.strip():
        return value.strip()

    from_meta = _from_metadata(record, "text")
    if from_meta:
        return from_meta
    return None


def extract_selfcrit_question(record: dict[str, Any], data_type: str) -> str | None:
    """
    Extract the question to feed to the model for Self-Critique (RL-MIA).
    DPO uses data["prompt"]; everything else uses data["text"].
    """
    if is_dpo_data_type(data_type):
        prompt = record.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
        return None
    return extract_text(record)


def extract_question(record: dict[str, Any], data_type: str = "auto") -> str | None:
    normalized = normalize_data_type(data_type)
    if is_paragraph_data_type(normalized):
        return extract_text(record)

    from_meta = _from_metadata(record, "question")
    if from_meta:
        return from_meta
    text = extract_text(record)
    if text is None:
        return None
    match = QUESTION_PATTERN.search(text)
    if not match:
        if normalized == "auto":
            return text
        return None
    question = match.group(1).strip()
    return question or None


def extract_answer(record: dict[str, Any], data_type: str = "auto") -> str | None:
    normalized = normalize_data_type(data_type)
    if is_paragraph_data_type(normalized):
        return extract_text(record)

    from_meta = _from_metadata(record, "answer")
    if from_meta:
        return from_meta
    text = extract_text(record)
    if text is None:
        return None
    match = ANSWER_PATTERN.search(text)
    if not match:
        if normalized == "auto":
            return text.strip() or None
        return None
    answer = match.group(1).strip()
    return answer or None