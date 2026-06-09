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
conda create -n olmo-detect python=3.11
conda activate olmo-detect
pip install -r requirements.txt
```


## Benchmark

### Overview
Built upon the OLMo 2 training pipeline, OLMo-Detect comprises nine domains across all three stages of modern LLM training: pre-training (DCLM-Baseline, peS2o, OpenWebMath, and StarCoder), mid-training (GSM8K and StackExchange), and post-training (SFT, DPO, and RLVR). The benchmark is organized as follows:

```
benchmark/
  <stage>/<domain>/
    contaminated/
      matched/<split>/    contaminated_<stage>_<domain>_<split>_matched.jsonl
      shifted/<split>/    contaminated_<stage>_<domain>_<split>_shifted.jsonl
    uncontaminated/
      <split>/            uncontaminated_<stage>_<domain>_<split>.jsonl
```

- `<split>`: `dev` (hyperparameter tuning) or `test` (evaluation).
- `matched`: `OLMo-Detect`, where contaminated and uncontaminated splits are explicitly aligned along up to three axes: text quality, temporal range, and lexical similarity.
- `shifted`: `OLMo-Detect (Shifted)`, where contaminated splits are sampled without distributional alignment to their uncontaminated counterparts.


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
| **RLVR** | [gsm8k-test.jsonl](https://huggingface.co/datasets/openai/gsm8k/blob/main/main/test-00000-of-00001.parquet) (excluding the 200 OLMo 2 dev instances) + 8-shot CoT<br>[MATH test split](https://huggingface.co/datasets/qwedsacf/competition_math/tree/main/data) + 3-shot CoT<br>[Tülu 2 SFT Mixture IFEval prompts](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture/tree/main/data) + constraints | [RLVR-GSM-MATH-IF-Mixed-Constraints](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints/tree/main/data) |


<!-- 
Verbatim contamination detection asks whether a given text appears *verbatim* in
an LLM's training data, **without access to the training corpus**. OLMo-Detect is
built on the fully open [OLMo 2](https://arxiv.org/abs/2501.00656) training
pipeline, so contaminated membership is known and uncontaminated splits can be
rigorously verified against the actual training data via infini-gram.

The benchmark has two complementary parts:

- **OLMo-Detect (Matched)** — contaminated and uncontaminated splits are explicitly
  aligned along text quality, temporal range, and lexical similarity. This isolates
  the *genuine* contamination signal and measures a method's intrinsic detection
  ability.
- **OLMo-Detect (Shifted)** — contaminated splits are sampled *without* distributional
  alignment, so the same domains carry a built-in distribution shift. This measures a
  method's *robustness*: whether it detects memorization or merely distributional
  mismatch.

---

---

## 2. Detection Methods Evaluated

We evaluate state-of-the-art methods spanning four families. Each is invoked by name
through `run_detection.py` (see §3). Method implementations live in `methods/`.

| Family | Method (key) | Notes |
|--------|--------------|-------|
| Likelihood-based | Perplexity / Zlib / Lowercase (`loss_zlib_lowercase`) | three scores from one pass |
| Likelihood-based | Min-K% / Min-K%++ (`minkprob`) | two scores from one pass |
| Likelihood-based | DC-PDD (`dcpdd`) | corpus token-frequency calibration |
| Likelihood-based | PAC (`pac`) | polarized perturbation calibration |
| Likelihood-based | RECALL (`recall`) | needs uncontaminated dev prefix (`--uncon-dev-path`) |
| Likelihood-based | CAMIA (`camia`) | context-aware loss dynamics |
| Likelihood-based | EMMIA (`emmia`) | EM-refined prefix selection (distinct output layout, see §4) |
| Perturbation | Neighborhood Attack (`neighborhood_attack`) | generation-based |
| Perturbation | DCQ (`dcq`) | multiple-choice; continuous score from first-token logprobs |
| Prompt | Guided Instruction (`guided_instruction`) | generation-based; ROUGE-L F1 score |
| Output-distribution | CDD (`cdd`) | output-distribution peakedness |
| Output-distribution | Self-Critique (`selfcrit`) | RL-oriented; evaluated on DPO and RLVR only |

> DCQ and Guided Instruction are originally *categorical*. Following the paper
> (Appendix L), we extract continuous per-sample scores (next-token logprobs for DCQ;
> ROUGE-L F1 for Guided Instruction) so AUC/TPR can be computed. The underlying
> signals are unchanged; only the final categorical decision step is replaced.

**Metrics.** We report **AUC** and **TPR@5%FPR**, both computed by the scorer in §4.

---

## 3. Reproducing Results

### 3.1 Requirements

- Python 3.11
- Install dependencies: `pip install -r requirements.txt`
  (`torch`, `transformers`, `datasets`, `numpy`, `scikit-learn`, `nltk`, `rouge_score`)
- The four OLMo 2 instruct checkpoints under `olmo_models/` (`OLMo2-1B-Instruct`,
  `OLMo2-7B-Instruct`, `OLMo2-13B-Instruct`, `OLMo2-32B-Instruct`), downloadable from
  [allenai on Hugging Face](https://huggingface.co/allenai).
- DCQ and Guided Instruction call an external judge model; set the relevant API key in
  your environment if you run those methods.

### 3.2 One domain, all methods and model sizes

`run.slurm` runs every method for every model size on one (contaminated,
uncontaminated) pair. It can be launched with `sbatch` (SLURM) or plain `bash`.

```bash
bash run.slurm \
  --con   benchmark/pretraining/openwebmath/contaminated/matched/test/contaminated_pretraining_openwebmath_test_matched.jsonl \
  --uncon benchmark/pretraining/openwebmath/uncontaminated/test/uncontaminated_pretraining_openwebmath_test.jsonl
```

Useful options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--con PATH` | (required) | contaminated test JSONL |
| `--uncon PATH` | (required) | uncontaminated test JSONL |
| `--models "A B …"` | all 4 sizes | model dir names under `--model-dir` |
| `--methods "a b …"` | all 12 | subset of methods to run |
| `--uncon-dev PATH` | — | **required by `recall` and `camia`** (uncontaminated dev split) |
| `--con-dev PATH` | — | contaminated dev split (calibrated methods) |
| `--model-dir DIR` | `olmo_models` | location of the checkpoints |
| `--out DIR` | `results` | output directory |
| `--batch-size N` | 1 | inference batch size |
| `--seed N` | 42 | global seed |
| `--env PATH` | — | virtualenv/conda activate script to source |

For the shifted setting, point `--con` at the corresponding `.../shifted/test/..._test_shifted.jsonl`.

Per-record scores are written to:

```
results/<dataset_stem>/<model_name>/<method>.json
```

### 3.3 Running a single method directly

`run.slurm` is a thin loop over `run_detection.py`. To run one method on one model:

```bash
python run_detection.py \
  --model  olmo_models/OLMo2-1B-Instruct \
  --data   benchmark/pretraining/openwebmath/contaminated/matched/test/contaminated_pretraining_openwebmath_test_matched.jsonl \
  --method dcpdd \
  --batch-size 1 --max-tokens 1024 --output-dir results
```

`--method` accepts one or more names; `--uncon-dev-path` supplies the uncontaminated
dev split for `recall`/`camia`.

---

## 4. Scoring: AUC and TPR@5%FPR

After scores are produced for both halves of a domain, `AUC_TPR@5%FPR.py` pairs the
contaminated (label 1) and uncontaminated (label 0) result directories and prints
**AUC** and **TPR@5%FPR** for every method and model size, plus the cross-size mean:

```bash
python "AUC_TPR@5%FPR.py" \
  --con   results/contaminated_pretraining_openwebmath_test_matched \
  --uncon results/uncontaminated_pretraining_openwebmath_test
```

Options:

| Flag | Default | Meaning |
|------|---------|---------|
| `--con DIR` / `--uncon DIR` | (required) | the two result directories |
| `--fpr FLOAT` | `0.05` | target FPR for the TPR metric |
| `--models "A B …"` | 4 sizes present | restrict to specific model dirs |
| `--emmia-dir DIR` | — | EMMIA cell dir (see below) |
| `--output PATH` | — | also dump the full table as JSON |

- **AUC** = `roc_auc_score`. **TPR@5%FPR** = `np.interp` on the ROC curve; for
  degenerate all-equal scores, TPR is set to the FPR (chance level).
- **EMMIA** has a different output layout (precomputed `final_auc` per sub-method plus
  per-sample `final_scores`), so it is only included when `--emmia-dir` is given. Its
  AUC and TPR each pick the best sub-method *independently*, matching the paper's
  separate aggregators.

---

## 5. Repository Map

```
benchmark/              the OLMo-Detect data (see §1.2)
methods/                detection-method implementations (one *_method.py per method)
run_detection.py        CLI: run one model × methods × data, write per-record scores
run.slurm               reproduction driver over run_detection.py for one domain
AUC_TPR@5%FPR.py        scorer: AUC + TPR@5%FPR from two result directories
detector.py             orchestration used by run_detection.py
data_utils.py           dataset loading / text extraction
model_utils.py          model loading / inference helpers
base_method.py          method base class
method_loader.py        maps method names to implementations
results/                precomputed per-record scores for all reported runs
requirements.txt        Python dependencies
``` -->
