from __future__ import annotations

import argparse
import json
from typing import Any

from detector import Detector, DetectorConfig

'''
python run_detection.py \
  --model olmo_models/OLMo2-1B-Instruct \
  --data \
    ./benchmark/midtraining/math-mix/gsm8k/contaminated/contaminated_midtraining_math-mix_gsm8k.jsonl \
    ./benchmark/midtraining/math-mix/gsm8k/uncontaminated/uncontaminated_midtraining_math-mix_gsm8k.jsonl \
  --method DCPDD \
  --batch-size 6 \
  --max-tokens 4096 \
  --output-dir results
'''

def main(
    data: list[str],
    model: str,
    method: list[str],
    output_dir: str = "results",
    seed: int = 42,
    max_tokens: int = 1024,
    batch_size: int = 3,
    uncon_dev_path: str | None = None,
) -> dict[str, Any]:
    '''
    Data: List of dataset paths
    Model: Path to model directory
    Method: List of method names (e.g., selfcrit, minkprob)
    Output_dir: Directory for output result files
    Uncon_dev_path: Path to uncontaminated dev JSONL (required by RECALL and calibrated methods)
    '''
    config = DetectorConfig(
        data_paths=data,
        model_path=model,
        method_names=method,
        output_dir=output_dir,
        seed=seed,
        max_tokens=max_tokens,
        batch_size=batch_size,
        uncon_dev_path=uncon_dev_path,
    )
    detector = Detector(config)
    return detector.run()


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run contamination detection for one model across methods and datasets."
    )
    parser.add_argument("--model", required=True, help="Path to model directory or model ID.")
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="One or more dataset paths (.jsonl or .json).",
    )
    parser.add_argument(
        "--method",
        required=True,
        nargs="+",
        help="One or more method names (e.g., selfcrit minkprob).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output result files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Inference batch size for methods that support batching (e.g., minkprob, selfcrit).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens for generation-based methods (e.g., selfcrit).",
    )
    parser.add_argument(
        "--uncon-dev-path",
        type=str,
        default=None,
        help="Path to the uncontaminated dev JSONL file. "
             "Required by methods that need a reference distribution "
             "(e.g. recall, calibrated_loss_zlib_lowercase).",
    )
    return parser


if __name__ == "__main__":
    parser = _build_cli()
    args = parser.parse_args()
    result = main(
        data=args.data,
        model=args.model,
        method=args.method,
        output_dir=args.output_dir,
        seed=args.seed,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        uncon_dev_path=args.uncon_dev_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))