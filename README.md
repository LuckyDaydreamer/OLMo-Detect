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


## 1 Benchmark
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

- `<split>`: `dev` (hyperparameter tuning, ~10%) or `test` (evaluation, ~90%).
- `matched`: `OLMo-Detect`, where contaminated and uncontaminated splits are explicitly aligned along up to three axes: text quality, temporal range, and lexical similarity.
- `shifted`: `OLMo-Detect (Shifted)`, where contaminated splits are sampled without distributional alignment to their uncontaminated counterparts.


## 2. Infini-Gram Indexed OLMo 2 Corpus
We employ Infini-gram (Liu et al., 2024), a suffix-array–based indexing system, to index the full OLMo 2 training corpus across all four model sizes and all three training stages. The complete indexed corpus (~12.3 TB) is available for download (link withheld for anonymity and will be released after the anonymity period).


<!-- ## 3. Data Processing Code -->




<!-- ### 1.3 Record schema

Every record contains a `text` field — **`text` is always the input scored by the
detection methods.** Records also retain domain-native fields and the per-record
13-gram `*_overlap_score` used during construction. A few examples:

- **Pre-training (e.g. OpenWebMath, DCLM, peS2o, StarCoder):** `text`, plus source
  metadata (URL, quality scores, dates).
- **SFT / DPO / RLVR:** `messages` (chat turns) and `text` (the rendered, chat-templated
  string). DPO additionally has `prompt`, `chosen`, `rejected`, `chosen_text`,
  `rejected_text`, and rating metadata.
- **RLVR CoT subsets:** `text` is the CoT-prefixed input; `text_no_cot` is the same
  sample with the shared CoT boilerplate stripped.
 -->

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
