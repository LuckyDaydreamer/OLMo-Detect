<p align="center">
  <img src="assets/logo.svg" alt="OLMo-Detect" width="460">
</p>

<p align="center">
  <em>A Multi-Stage Benchmark for Verbatim Contamination Detection in Large Language Models</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/stages-3-4F8DFD"> 
  <img src="https://img.shields.io/badge/domains-9-4F8DFD"> 
  <img src="https://img.shields.io/badge/models-OLMo%202%20(1B%E2%80%9332B)-7C5CFC"> 
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-1b2330"></a>
</p>

---

This repository contains the data and code for the EMNLP 2026 submission *OLMo-Detect: A Multi-Stage Benchmark for Verbatim Contamination Detection in Large Language Models.*


## Setup

```bash
curl -L https://anonymous.4open.science/api/repo/OLMo-Detect-3E76/zip -o OLMo-Detect.zip
unzip OLMo-Detect.zip -d OLMo-Detect && cd OLMo-Detect
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Evaluation also needs the four OLMo 2 Instruct checkpoints, placed under `olmo_models/` with these exact directory names:

| Local directory | Hugging Face model |
|-----------------|--------------------|
| `olmo_models/OLMo2-1B-Instruct`  | [allenai/OLMo-2-0425-1B-Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) |
| `olmo_models/OLMo2-7B-Instruct`  | [allenai/OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) |
| `olmo_models/OLMo2-13B-Instruct` | [allenai/OLMo-2-1124-13B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct) |
| `olmo_models/OLMo2-32B-Instruct` | [allenai/OLMo-2-0325-32B-Instruct](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct) |

```bash
hf download allenai/OLMo-2-0425-1B-Instruct  --local-dir olmo_models/OLMo2-1B-Instruct
hf download allenai/OLMo-2-1124-7B-Instruct  --local-dir olmo_models/OLMo2-7B-Instruct
hf download allenai/OLMo-2-1124-13B-Instruct --local-dir olmo_models/OLMo2-13B-Instruct
hf download allenai/OLMo-2-0325-32B-Instruct --local-dir olmo_models/OLMo2-32B-Instruct
```


## Benchmark

### Overview
Built upon the OLMo 2 training pipeline, OLMo-Detect comprises nine domains across all three stages of modern LLM training: pre-training (DCLM-Baseline, peS2o, OpenWebMath, and StarCoder), mid-training (GSM8K and StackExchange), and post-training (SFT, DPO, and RLVR). The repository is organized as follows:

```
OLMo-Detect/
├── benchmark/                         # benchmark data — 9 domains × 3 stages
│   └── <stage>/<domain>/
│       ├── contaminated/
│       │   ├── matched/<split>/       # contaminated_<stage>_<domain>_<split>_matched.jsonl
│       │   └── shifted/<split>/       # contaminated_<stage>_<domain>_<split>_shifted.jsonl
│       └── uncontaminated/<split>/    # uncontaminated_<stage>_<domain>_<split>.jsonl
├── results/                           # released per-record detection scores (every method × model size)
├── methods/                           # detection-method implementations (one module per method)
├── utils/                             # data-processing code (13-gram filtering, sampling, IFEval constraints)
├── infini_gram/, infini_gram_mini/    # vendored infini-gram (used by utils/)
├── run_all.slurm, run_all.py          # Step 1: run a method across all domains
├── evaluate.py                        # Step 2: AUC / TPR@5%FPR (all-stage / per-stage / per-domain / per-subset)
├── run_detection.py                   # score one (model, file, method); called by run_all
├── detector.py, model_utils.py, data_utils.py, base_method.py, method_loader.py   # core engine
├── requirements.txt
└── .gitignore
```

- `<split>`: `dev` (hyperparameter tuning) or `test` (evaluation).
- `matched`: `OLMo-Detect`, where contaminated and uncontaminated splits are explicitly aligned along up to three axes: text quality, temporal range, and lexical similarity.
- `shifted`: `OLMo-Detect (Shifted)`, where contaminated splits are sampled without distributional alignment to their uncontaminated counterparts.
- **DPO** additionally has separate `chosen` and `rejected` files (e.g. `..._matched_chosen_test.jsonl` and `..._matched_rejected_test.jsonl`), since its two scored inputs are evaluated independently.


### Data Format
Every instance has a **`text`** field, which is the input scored by detection methods (for DPO, the scored inputs are `chosen_text` and `rejected_text` instead). Within a domain, contaminated and uncontaminated instances share the same fields; uncontaminated instances additionally carry per-model **`<size>_13-gram_overlap_score`** fields (the 13-gram overlap with the OLMo 2 corpus). Beyond `text`, each domain keeps its source-native metadata:

| Domain | Domain-specific fields |
|--------|------------------------|
| **DCLM-Baseline** | `url`, `metadata` (WARC headers), `language_id_whole_page_fasttext`, fastText quality scores, n-gram / word counts |
| **OpenWebMath** | `url`, `created`, `metadata` (math-extraction info) |
| **peS2o** | `id`, `added`, `created` |
| **StarCoder** | `id`, `max_stars_repo_path`, `max_stars_repo_name`, `max_stars_count`, `top1_word_freq`, `top2_word_freq` |
| **GSM8K** | `id`, `source`, `added`, `created`, `metadata` (original question/answer) |
| **Stack Exchange** | `created`, `question_score`, `answer_score` |
| **SFT** | `messages` (chat turns; `text` is the rendered chat-templated string) |
| **DPO** | `prompt`, `chosen`, `rejected`, `chosen_text`, `rejected_text`, `chosen_model`, `rejected_model`, `chosen_rating`, `rejected_rating` |
| **RLVR** | `messages`, `ground_truth`, `dataset`, `constraint_type`, `constraint`; `text_no_cot` (text with the shared CoT prefix removed) |


## Infini-Gram Indexed OLMo 2 Corpus
The original OLMo 2 data is available from the following sources:

- **Pre-training:** [olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124)
- **Mid-training:** [dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)
- **Post-training:**
  - **SFT:**
    - 7B and 13B: [tulu-3-sft-olmo-2-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture)
    - 1B and 32B: [tulu-3-sft-olmo-2-mixture-0225](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225)
  - **DPO:**
    - 1B: [olmo-2-0425-1b-preference-mix](https://huggingface.co/datasets/allenai/olmo-2-0425-1b-preference-mix)
    - 7B: [olmo-2-1124-7b-preference-mix](https://huggingface.co/datasets/allenai/olmo-2-1124-7b-preference-mix)
    - 13B: [olmo-2-1124-13b-preference-mix](https://huggingface.co/datasets/allenai/olmo-2-1124-13b-preference-mix)
    - 32B: [olmo-2-0325-32b-preference-mix](https://huggingface.co/datasets/allenai/olmo-2-0325-32b-preference-mix)
  - **RLVR:** [RLVR-GSM-MATH-IF-Mixed-Constraints](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints)

We employ Infini-gram, a suffix-array–based indexing system, to index the full OLMo 2 training corpus across all four model sizes and all three training stages. The complete indexed corpus (~12.3 TB) is available for download (link withheld for anonymity and will be released after the anonymity period).


## Data Processing Code
The `utils/` directory contains the data processing code:
- `13gram_filtering.py`: filters uncontaminated instances against the infini-gram–indexed OLMo 2 corpus, keeping only those whose 13-gram overlap stays below the 20% threshold. For DCLM-Baseline, StarCoder, and Stack Exchange, it also applies text quality boundary check. The DCLM quality filter needs the fastText model from [mlfoundations/fasttext-oh-eli5](https://huggingface.co/mlfoundations/fasttext-oh-eli5), downloaded into `utils/fasttext_dir/`.
- `sample_contaminated_candidates.py`: samples the contaminated candidate pool via boundary sampling for quality-filtered domains (DCLM-Baseline, StarCoder, and Stack Exchange), retaining instances whose text quality scores sit just above OLMo 2's filtering threshold.
- `simulated_annealing_sampling.py`: selects the final contaminated split from the candidate pool via constrained simulated annealing, matching the uncontaminated split's instance count and token count while minimizing lexical (and, where dates are available, temporal) distribution mismatch.
- `ifeval_constraint.py`: augments RLVR/IFEval prompts with verifiable instruction-following constraints, replicating the OLMo 2 setup exactly.

### Source Data
The source files used to sample each domain are detailed below:

| Domain | Uncontaminated | Contaminated |
|--------|----------------|--------------|
| **DCLM-Baseline** | DCLM-Pool [global-shard_01-local-shard_0, global-shard_05-local-shard_6, global-shard_07-local-shard_3](https://data.commoncrawl.org/contrib/datacomp/DCLM-refinedweb/index.html) | olmo-mix-1124 DCLM-Baseline [global-shard_01-local-shard_0](https://huggingface.co/datasets/allenai/olmo-mix-1124/tree/main/data/dclm/raw/hero-run-fasttext_for_HF/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/global-shard_01_of_10/local-shard_0_of_10), [global-shard_05-local-shard_6](https://huggingface.co/datasets/allenai/olmo-mix-1124/tree/main/data/dclm/raw/hero-run-fasttext_for_HF/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/global-shard_05_of_10/local-shard_6_of_10), [global-shard_07-local-shard_3](https://huggingface.co/datasets/allenai/olmo-mix-1124/tree/main/data/dclm/raw/hero-run-fasttext_for_HF/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/global-shard_07_of_10/local-shard_3_of_10) |
| **peS2o** | [validation-00.jsonl](https://huggingface.co/datasets/allenai/peS2o/blob/main/data/v2/validation-00000-of-00002.json.gz), [validation-01.jsonl](https://huggingface.co/datasets/allenai/peS2o/blob/main/data/v2/validation-00001-of-00002.json.gz) | [olmo-mix-1124 peS2o](https://huggingface.co/datasets/allenai/olmo-mix-1124/tree/main/data/pes2o) |
| **OpenWebMath** | [test.jsonl](https://huggingface.co/datasets/EleutherAI/proof-pile-2/tree/main/open-web-math/test) | [olmo-mix-1124 OpenWebMath](https://huggingface.co/datasets/allenai/olmo-mix-1124/tree/main/data/open-web-math/train) |
| **StarCoder** | Original StarCoder [Assembly](https://huggingface.co/datasets/bigcode/starcoderdata/tree/main/assembly) and [Java](https://huggingface.co/datasets/bigcode/starcoderdata/tree/main/java) | [olmo-mix-1124 Assembly and Java](https://huggingface.co/datasets/allenai/olmo-mix-1124/tree/main/data/starcoder/v1-decon-100_to_20k-2star-top_token_030/documents) |
| **GSM8K** | [gsm8k-test.jsonl](https://huggingface.co/datasets/openai/gsm8k/blob/main/main/test-00000-of-00001.parquet) (excluding the 200 OLMo 2 dev instances) | [gsm8k-train.jsonl](https://huggingface.co/datasets/allenai/dolmino-mix-1124/tree/main/data/math/gsm8k/main/train) |
| **Stack Exchange** | [Original 2024-09-30 Stack Exchange dump](https://archive.org/details/stackexchange_20240930) | [dolmino-mix-1124 Stack Exchange](https://huggingface.co/datasets/allenai/dolmino-mix-1124/tree/main/data/stackexchange) |
| **SFT** | Original [Aya](https://huggingface.co/datasets/CohereLabs/aya_dataset/tree/main/data) and [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M/tree/main/data) | [tulu-3-sft-olmo-2-mixture Aya and WildChat](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture/tree/main/data) (7B and 13B)<br>[tulu-3-sft-olmo-2-mixture-0225 Aya and WildChat](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225/tree/main/data) (1B and 32B) |
| **DPO** | [Original WildChat prompts](https://huggingface.co/datasets/allenai/WildChat-1M/tree/main/data) + [UltraFeedback pipeline for response generation](https://github.com/allenai/open-instruct/blob/main/scripts/synth_pref/README.md) | [olmo-2-0425-1b-preference-mix WildChat](https://huggingface.co/datasets/allenai/olmo-2-0425-1b-preference-mix) (1B)<br>[olmo-2-1124-7b-preference-mix WildChat](https://huggingface.co/datasets/allenai/olmo-2-1124-7b-preference-mix) (7B)<br>[olmo-2-1124-13b-preference-mix WildChat](https://huggingface.co/datasets/allenai/olmo-2-1124-13b-preference-mix) (13B)<br>[olmo-2-0325-32b-preference-mix WildChat](https://huggingface.co/datasets/allenai/olmo-2-0325-32b-preference-mix) (32B) |
| **RLVR** | [gsm8k-test.jsonl](https://huggingface.co/datasets/openai/gsm8k/blob/main/main/test-00000-of-00001.parquet) (excluding the 200 OLMo 2 dev instances) + 8-shot CoT<br>[MATH test split](https://huggingface.co/datasets/qwedsacf/competition_math/tree/main/data) (the file mixes train and test; split them first) + 3-shot CoT<br>[Tülu 2 SFT Mixture IFEval prompts](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture/tree/main/data) + constraints | [RLVR-GSM-MATH-IF-Mixed-Constraints](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints/tree/main/data) |


## Reproducing Results

**Step 1: Get Per-Record Scores.** 

Run the target method across all domains using the following script; scores are written under `results_repro/`:
```bash
sbatch run_all.slurm <method> [matched|shifted]   # default split: matched
```
`<method>` is one of: `loss_zlib_lowercase` (Perplexity, Zlib, Lowercase), `minkprob` (Min-K%, Min-K%++), `dcpdd`, `recall`, `camia`, `pac`, `neighborhood_attack`, `dcq`, `guided_instruction`, `cdd`, `selfcrit`. Alternatively, you can skip this step and use the scores already in `results/`. 

Methods with tunable hyperparameters are tuned on the `dev` split, with all three stages (pre-training, mid-training, and post-training) pooled into a single tuning set.

**Step 2: Compute AUC and TPR@5%FPR.** 

Run `evaluate.py` for the target method. Set `--results-dir` to the output directory generated in Step 1 (e.g., `results_repro/`). If omitted, the script evaluates the scores provided in `results/`.
```bash
python evaluate.py --list                                                 # show available method keys
python evaluate.py --method camia --show-tpr --results-dir results_repro  # AUC/TPR@5%FPR per stage / domain / subset
```

**Example.** 

To evaluate `loss_zlib_lowercase` end to end:
```bash
sbatch run_all.slurm loss_zlib_lowercase matched                               # scores -> results_repro/
python evaluate.py --method ppl       --show-tpr --results-dir results_repro   # Perplexity
python evaluate.py --method zlib      --show-tpr --results-dir results_repro   # Zlib
python evaluate.py --method lowercase --show-tpr --results-dir results_repro   # Lowercase
```


## Evaluating a New Method

OLMo-Detect is meant as an evaluation suite, so you can plug in your own detection method and measure it with the same protocol.

**1. Implement the method.** Add `methods/<your_method>_method.py` with a subclass of `BaseMethod` (see `base_method.py`). It needs a `name` and a `run(model_bundle, dataset)` that returns a `samples` list: one score per record, where a **higher score means more likely contaminated**:
```python
# methods/mymethod_method.py
from base_method import BaseMethod

class MyMethod(BaseMethod):
    name = "mymethod"

    def run(self, model_bundle, dataset):
        samples = []
        for i, rec in enumerate(dataset.records):
            text = rec["text"]                          # the instance to score
            score = my_score(text, model_bundle)        # higher = more likely contaminated
            samples.append({"record_index": i, "score": float(score)})
        return {"samples": samples, "num_scored_records": len(samples)}
```
`model_bundle` exposes the loaded model/tokenizer (and `llm_adapter` scoring helpers); `dataset.records` are the raw instances. The existing files under `methods/` are good templates: `loss_zlib_lowercase_method.py` / `minkprob_method.py` for likelihood-based scoring, `cdd_method.py` / `selfcrit_method.py` for generation-based. The loader finds `methods/<name>_method.py` automatically, so `--method <name>` works with no further registration (to add a short alias, edit `method_loader.py`).

**2. Run it across all domains** (scores → `results_repro/`):
```bash
sbatch run_all.slurm mymethod matched
```
`run_all` works with any method name. If your method needs the `dev` split as a reference, or long generations, add it to `DEV_METHODS` / `GEN_METHODS` in `run_all.py`.

**3. Score it.** Register the score column by adding one line to the `METHODS` map in `evaluate.py`: `"mymethod": ("mymethod", "score")` (file stem, score field), and then:
```bash
python evaluate.py --method mymethod --show-tpr --results-dir results_repro
```
