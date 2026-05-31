from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from data_utils import load_datasets
from method_loader import load_methods
from model_utils import load_model_bundle
from progress_utils import progress


def _safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("_") or "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return None
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class DetectorConfig:
    data_paths: list[str]
    model_path: str
    method_names: list[str]
    output_dir: str = "results"
    seed: int = 42
    max_tokens: int = 1024
    batch_size: int = 3
    uncon_dev_path: str | None = None


class Detector:
    def __init__(self, config: DetectorConfig):
        self.config = config

    def run(self) -> dict[str, Any]:
        model_bundle = load_model_bundle(
            model_path=self.config.model_path,
            seed=self.config.seed,
        )
        datasets = load_datasets(self.config.data_paths)
        # methods = load_methods(
        #     self.config.method_names,
        #     method_kwargs={
        #         "*": {
        #             "batch_size": self.config.batch_size,
        #             "max_tokens": self.config.max_tokens,
        #         },
        #     },
        # )

        wildcard_kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.uncon_dev_path is not None:
            wildcard_kwargs["uncon_dev_path"] = self.config.uncon_dev_path

        methods = load_methods(
            self.config.method_names,
            method_kwargs={"*": wildcard_kwargs},
        )

        output_root = Path(self.config.output_dir)
        all_results: list[dict[str, Any]] = []
        manifest_paths: list[str] = []

        for dataset in progress(
            datasets,
            desc="Datasets",
            unit="dataset",
            dynamic_ncols=True,
        ):
            dataset_dir = output_root / _safe_filename(dataset.name)
            model_output_dir = dataset_dir / _safe_filename(model_bundle.model_name)
            model_output_dir.mkdir(parents=True, exist_ok=True)

            manifest: dict[str, Any] = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(self.config),
                "model_name": model_bundle.model_name,
                "dataset_name": dataset.name,
                "dataset_path": dataset.path,
                "dataset_data_type": dataset.data_type,
                "results": [],
            }

            for method in progress(
                methods,
                desc=f"Methods [{dataset.name}]",
                unit="method",
                dynamic_ncols=True,
                leave=False,
            ):
                start = time.time()
                method_result = method.run(model_bundle=model_bundle, dataset=dataset)
                elapsed_seconds = time.time() - start

                # Methods may emit alternate-view payloads via "_sidecar_outputs":
                # a {filename_stem: method_result_like} mapping. Each gets written
                # alongside the primary file. The key is stripped from the primary
                # payload before serialization.
                sidecar_outputs: dict = {}
                if isinstance(method_result, dict) and "_sidecar_outputs" in method_result:
                    sidecar_outputs = method_result.pop("_sidecar_outputs") or {}

                output_payload = {
                    "model_name": model_bundle.model_name,
                    "model_path": model_bundle.model_path,
                    "dataset_name": dataset.name,
                    "dataset_path": dataset.path,
                    "dataset_data_type": dataset.data_type,
                    "method_name": method.name,
                    "elapsed_seconds": elapsed_seconds,
                    "result": method_result,
                }
                output_payload = _json_safe(output_payload)

                output_path = model_output_dir / f"{_safe_filename(method.name)}.json"
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(output_payload, f, ensure_ascii=False, indent=2, allow_nan=False)

                for sidecar_stem, sidecar_result in sidecar_outputs.items():
                    sidecar_payload = _json_safe({
                        "model_name": model_bundle.model_name,
                        "model_path": model_bundle.model_path,
                        "dataset_name": dataset.name,
                        "dataset_path": dataset.path,
                        "dataset_data_type": dataset.data_type,
                        "method_name": sidecar_stem,
                        "elapsed_seconds": elapsed_seconds,
                        "result": sidecar_result,
                    })
                    sidecar_path = model_output_dir / f"{_safe_filename(sidecar_stem)}.json"
                    with sidecar_path.open("w", encoding="utf-8") as f:
                        json.dump(sidecar_payload, f, ensure_ascii=False, indent=2, allow_nan=False)

                row = {
                    "dataset_name": dataset.name,
                    "dataset_path": dataset.path,
                    "dataset_data_type": dataset.data_type,
                    "method_name": method.name,
                    "output_path": str(output_path),
                    "elapsed_seconds": elapsed_seconds,
                }
                manifest["results"].append(row)
                all_results.append(row)
            
            method_suffix = _safe_filename(methods[0].name) if methods else "default"
            manifest_path = model_output_dir / f"run_manifest_{method_suffix}.json"

            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(_json_safe(manifest), f, ensure_ascii=False, indent=2, allow_nan=False)
            manifest_paths.append(str(manifest_path))

        return {
            "model_name": model_bundle.model_name,
            "output_dir": str(output_root),
            "manifest_path": manifest_paths[0] if len(manifest_paths) == 1 else None,
            "manifest_paths": manifest_paths,
            "results": all_results,
        }
