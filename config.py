"""Path configuration.

Everything resolves relative to the repository root, so a fresh clone runs without
editing any file. Override any entry with the matching environment variable to keep
large artefacts (model weights, scores) outside the repo.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _path(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, default)).expanduser()


# benchmark ships with the repo
BENCHMARK = _path("OLMO_DETECT_BENCH", REPO / "benchmark")

# per-record scores: results/<leaf>/<model>/<method>.json, plus results/emmia/<tag>/
RESULTS = _path("OLMO_DETECT_RESULTS", REPO / "results")

# leave-one-domain-out scores for the three supervised detectors, one dir each
LODO_MIA_TUNER = _path("OLMO_DETECT_LODO_MT", REPO / "results_lodo/mia_tuner")
LODO_FSD = _path("OLMO_DETECT_LODO_FSD", REPO / "results_lodo/fsd")
LODO_CAMIA = _path("OLMO_DETECT_LODO_CAMIA", REPO / "results_lodo/camia_lr")

# model weights: downloaded from the HuggingFace Hub
MODELS = _path("OLMO_DETECT_MODELS", REPO / "olmo_models")

LOGS = _path("OLMO_DETECT_LOGS", REPO / "analysis/logs")
