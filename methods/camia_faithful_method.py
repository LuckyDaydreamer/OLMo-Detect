"""
Context-Aware Membership Inference Attacks against Pre-trained Large Language Models (CAMIA)
Paper:  https://aclanthology.org/2025.emnlp-main.370.pdf
Github: https://github.com/changhongyan123/context_aware_mia

This is a FAITHFUL re-implementation of the *main, unsupervised* CAMIA attack
(Section 4 of the paper: p-value composition over a bag of context-aware
signals). It follows the official code paths:

    run_ours_different_agg.py   (p-value composition; the main method)
    util_features.py            (collect_all_features, lz, approx-entropy, ...)
    run_ours_construct_mia_data.py  (repeated-input construction: 6 copies)
    util_features.load_data_from_model_history  (per-occurrence extraction)

It is NOT the supervised Appendix-B variant (run_ours_train_lr.py, the
LogisticRegression + PCA/GroupPCA "with-knowledge" attack). No member labels
are used to fit any classifier; only a non-member calibration null is used to
turn each signal into a p-value, and member/non-member dev means are used only
to orient signal signs (exactly as the official code does via `y_all`).

Signal set (per sample), matching the official code exactly:
  base (on the standalone sequence):
    loss, ppl, count_above, lz_complexity, token_diversity, count_mean,
    calibrated_loss, calibrated_ppl
  repetition differences (occ0 - occ1, occ0 - occ2), for every base signal
    except token_diversity
  slope, approximate_entropy   (on the full repeated sequence)
  count_above_pre              (on the standalone sequence)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle


# ══════════════════════════════════════════════════════════════════════════════
# Faithful signal definitions (mirror util_features.collect_all_features)
# ══════════════════════════════════════════════════════════════════════════════
#
# NOTE on the "-1" end-times: the official get_loss/get_ppl/get_count_mean/
# get_token_diversity are called with end_time=-1 for the "full" variant, which
# in numpy means the slice  x[0:-1]  -> the last token is dropped.  We replicate
# that exactly (see `_cut`) so numbers match the reference implementation.

_END_TIMES   = (-1, 200, 300)          # loss / ppl / token_diversity / count_mean
_THRESHOLDS  = (-1.0, -2.0, -3.0)      # count_above (on first 200 tokens)
_LZ_BINS     = (3, 4, 5)               # lz_complexity (on first 200 tokens)
_SLOPE_CUTS  = (600, 800, 1000)        # slope       (on full repeated sequence)
_APEN_CUTS   = (600, 800, 1000)        # approx-entropy (on full repeated sequence)
_PRE_CUTS    = (10 ** 9, 200, 300)     # count_above_pre (on standalone sequence)

_APEN_M = 8
_APEN_R = 0.8

# Base keys that receive overall_diff_{1,2} signals (token_diversity excluded,
# exactly as in run_ours_different_agg.py:  `if keys not in ["token_diversity"]`).
_DIFF_KEYS = (
    "loss", "ppl", "count_above", "lz_complexity",
    "count_mean", "calibrated_loss", "calibrated_ppl",
)


def _cut(a: np.ndarray, end: int) -> np.ndarray:
    """Replicate the official slice semantics `a[0:end]` (end=-1 drops last)."""
    return a[: int(end)]


def _lz(x: np.ndarray, bins: int) -> float:
    """Lempel-Ziv complexity — exact port of util_features.lempel_ziv_complexity."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan")
    edges = np.linspace(float(x.min()), 0.0, 100)
    x_disc = np.digitize(x, bins=edges, right=True)
    b2 = np.linspace(float(np.min(x_disc)), float(np.max(x_disc)), bins + 1)[1:]
    seq = np.searchsorted(b2, x_disc, side="left")
    subs: set = set()
    ind, inc = 0, 1
    while ind + inc <= n:
        ss = tuple(seq[ind: ind + inc])
        if ss in subs:
            inc += 1
        else:
            subs.add(ss)
            ind += inc
            inc = 1
    return len(subs) / n


def _apen(x: np.ndarray, m: int = _APEN_M, r: float = _APEN_R) -> float:
    """Approximate entropy — exact port of util_features.approximate_entropy.

    (No clamping of C: with self-matches C >= 1/(N-m+1) > 0 always, so log(C)
    is finite — identical to the reference.)"""
    x = np.asarray(x, dtype=float)
    N = x.size
    if N <= m + 1:
        return 0.0
    r_scaled = r * float(np.std(x))
    if r_scaled < 0:
        raise ValueError("r must be positive")
    if r_scaled == 0:
        return 0.0

    def _phi(mm: int) -> float:
        segs = np.array([x[i: i + mm] for i in range(N - mm + 1)])
        C = np.sum(
            np.max(np.abs(segs[:, None] - segs[None, :]), axis=2) <= r_scaled,
            axis=0,
        ) / (N - mm + 1)
        return float(np.sum(np.log(C)) / (N - mm + 1.0))

    return float(abs(_phi(m) - _phi(m + 1)))


def _slope(replp: np.ndarray, cut: int) -> float:
    """Linear-fit slope of the first `cut` tokens — port of util_features.get_slope."""
    s = replp[: int(cut)] if len(replp) >= cut else replp
    if len(s) <= 2:
        return float("nan")
    return float(np.polyfit(np.arange(len(s), dtype=float), s, 1)[0])


def _count_above_pre(lp: np.ndarray, cut: int) -> float:
    """Fraction of tokens whose log-prob exceeds the running mean of all
    preceding tokens — port of the `count_above_pre` block in
    run_ours_different_agg.py (cut-offs 1e9 / 200 / 300)."""
    n = min(int(cut), len(lp))
    if n <= 1:
        return float("nan")
    cur = lp[1:n]
    pre = np.array([np.mean(lp[:t]) for t in range(1, n)])
    return float(np.mean(cur > pre)) if len(cur) > 0 else float("nan")


def _features_one(lp: np.ndarray, ids: np.ndarray) -> dict[str, np.ndarray]:
    """Port of util_features.collect_all_features for a single sequence.

    Returns a dict of signal-blocks, each a vector over its variations.
    `lp`  : per-token log-probs (<= 0)
    `ids` : token ids (for token_diversity / calibration)
    """
    lp = np.asarray(lp, dtype=float)
    ids = np.asarray(ids, dtype=int)

    loss, ppl, div, cmean = [], [], [], []
    for e in _END_TIMES:
        s = _cut(lp, e)
        m = float(np.mean(s)) if len(s) > 0 else float("nan")
        loss.append(m)
        ppl.append(float(np.exp(-m)) if np.isfinite(m) else float("nan"))

        di = _cut(ids, e)
        div.append((len(set(di.tolist())) / len(di)) if len(di) > 0 else float("nan"))

        # count_above_mean == RAW COUNT of tokens above the sequence mean
        # (util_features.count_above_mean -> np.where(x > mean)[0].size)
        cmean.append(float(np.sum(s > np.mean(s))) if len(s) > 0 else float("nan"))

    loss = np.array(loss); ppl = np.array(ppl)
    div = np.array(div);   cmean = np.array(cmean)

    s200 = lp[:200]
    ca = [float(np.mean(s200 >= t)) if len(s200) > 0 else float("nan")
          for t in _THRESHOLDS]
    lz = [_lz(s200, b) if len(s200) > 0 else float("nan") for b in _LZ_BINS]

    with np.errstate(divide="ignore", invalid="ignore"):
        cal_loss = loss / div
        cal_ppl = ppl / div

    return {
        "loss":            loss,
        "ppl":             ppl,
        "count_above":     np.array(ca),
        "lz_complexity":   np.array(lz),
        "token_diversity": div,
        "count_mean":      cmean,
        "calibrated_loss": cal_loss,
        "calibrated_ppl":  cal_ppl,
    }


def _find_occurrence_lps(
    rep_ids: np.ndarray,
    rep_lp:  np.ndarray,
    pattern: np.ndarray,
    n_needed: int = 3,
    min_len:  int = 3,
) -> list[np.ndarray] | None:
    """Locate the first `n_needed` occurrences of the standalone token sequence
    inside the repeated sequence and return their per-token log-probs.

    Faithful to util_features.find_sublist_indices: search for the pattern; if
    fewer than `n_needed` matches are found, drop the leading token and retry
    (this absorbs the leading-space BPE merge that changes the first token of
    every occurrence after the first).  Because `rep_lp` is 1:1 aligned with
    `rep_ids` in this codebase, occurrence log-probs are simply rep_lp[s:s+L].
    """
    rep_ids = np.asarray(rep_ids, dtype=int)
    if len(rep_ids) == 0 or len(pattern) == 0:
        return None
    b = np.asarray(pattern, dtype=int)
    while len(b) >= min_len:
        L = len(b)
        if len(rep_ids) >= L:
            cand = np.where(rep_ids[: len(rep_ids) - L + 1] == b[0])[0]
        else:
            cand = np.array([], dtype=int)
        starts: list[int] = []
        for s in cand:
            if np.array_equal(rep_ids[s: s + L], b):
                starts.append(int(s))
                if len(starts) >= n_needed:
                    break
        if len(starts) >= n_needed:
            return [rep_lp[s: s + L] for s in starts[:n_needed]]
        b = b[1:]  # drop leading token, retry (handles the space-merge)
    return None


def _sample_blocks(
    lp_orig: list[float], ids_orig: list[int],
    rep_lp:  list[float], rep_ids:  list[int],
) -> dict[str, np.ndarray]:
    """All signal-blocks for one sample (base + overall_diff + slope/apen/pre)."""
    lp = np.asarray(lp_orig, dtype=float)
    ids = np.asarray(ids_orig, dtype=int)
    replp = np.asarray(rep_lp, dtype=float)
    repids = np.asarray(rep_ids, dtype=int)

    blocks: dict[str, np.ndarray] = {}
    if len(lp) == 0:
        return blocks

    # ── base signals on the standalone sequence ──────────────────────────────
    base = _features_one(lp, ids)
    blocks.update(base)

    # ── repetition-difference signals from occurrences 0/1/2 ─────────────────
    #    (feat(occ0) - feat(occ1), feat(occ0) - feat(occ2))
    occ = _find_occurrence_lps(repids, replp, ids, n_needed=3)
    if occ is not None and len(occ) >= 3:
        f0 = _features_one(occ[0], ids)   # ids only used for token_diversity,
        f1 = _features_one(occ[1], ids)   # which is excluded from diffs — the
        f2 = _features_one(occ[2], ids)   # official copied feats reuse orig ids.
        for k in _DIFF_KEYS:
            blocks[f"overall_diff_1_{k}"] = f0[k] - f1[k]
            blocks[f"overall_diff_2_{k}"] = f0[k] - f2[k]

    # ── fluctuation signals on the full repeated sequence ────────────────────
    blocks["slope"] = np.array([_slope(replp, c) for c in _SLOPE_CUTS])
    blocks["approximate_entropy"] = np.array(
        [_apen(replp[: c] if len(replp) >= c else replp) for c in _APEN_CUTS]
    )

    # ── count-above-running-mean on the standalone sequence ──────────────────
    blocks["count_above_pre"] = np.array([_count_above_pre(lp, c) for c in _PRE_CUTS])

    return blocks


def _stack_blocks(per_sample: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-sample block dicts into {key: (n_var, n_samples)} matrices."""
    keys: set[str] = set()
    for d in per_sample:
        keys.update(d.keys())
    out: dict[str, np.ndarray] = {}
    for k in keys:
        n_var = 0
        for d in per_sample:
            if k in d:
                n_var = len(d[k])
                break
        mat = np.full((n_var, len(per_sample)), float("nan"))
        for i, d in enumerate(per_sample):
            if k in d and len(d[k]) == n_var:
                mat[:, i] = d[k]
        out[k] = mat
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Direction calibration / p-values / composition  (main unsupervised method)
# ══════════════════════════════════════════════════════════════════════════════

def _directions(
    member_blocks:    dict[str, np.ndarray],
    nonmember_blocks: dict[str, np.ndarray],
) -> dict[str, int]:
    """Per-signal-block sign flip so members < non-members (pooled mean).

    Faithful to run_ours_different_agg.py:
        if feat.T[y==1].mean() > feat.T[y==0].mean(): feat = -feat
    The comparison pools over all variations of a block (a single scalar per
    key), and flips the whole block.
    """
    dirs: dict[str, int] = {}
    for key in set(member_blocks) | set(nonmember_blocks):
        if key not in member_blocks or key not in nonmember_blocks:
            dirs[key] = 1
            continue
        m = float(np.nanmean(member_blocks[key]))
        nm = float(np.nanmean(nonmember_blocks[key]))
        if not (np.isfinite(m) and np.isfinite(nm)):
            dirs[key] = 1
        else:
            dirs[key] = -1 if m > nm else 1
    return dirs


def _apply_directions(
    blocks: dict[str, np.ndarray], dirs: dict[str, int],
) -> dict[str, np.ndarray]:
    return {k: dirs.get(k, 1) * v for k, v in blocks.items()}


def _pvalues(
    target_blocks: dict[str, np.ndarray],
    pop_blocks:    dict[str, np.ndarray],
) -> np.ndarray | None:
    """Empirical one-sided p-values per (signal, variation) against the
    non-member calibration null `pop_blocks`.

    Faithful to  scipy.stats.percentileofscore(pop, value) / 100  with the
    default kind='rank' (the average of the strict and weak ranks), evaluated
    for every target value.  Returns a (n_signals, n_target) matrix.
    """
    rows: list[np.ndarray] = []
    for key in sorted(target_blocks):
        if key not in pop_blocks:
            continue
        tgt = target_blocks[key]   # (n_var, n_target)
        pop = pop_blocks[key]      # (n_var, n_pop)
        n_var = min(tgt.shape[0], pop.shape[0])
        for j in range(n_var):
            pop_col = pop[j][np.isfinite(pop[j])]
            if len(pop_col) == 0:
                continue
            sp = np.sort(pop_col)
            n = len(sp)
            v = tgt[j]
            left = np.searchsorted(sp, v, side="left")    # count(pop <  v)
            right = np.searchsorted(sp, v, side="right")  # count(pop <= v)
            p = (left + right) / (2.0 * n)                 # kind='rank'
            p = np.where(np.isfinite(v), p, 0.5)
            rows.append(p)
    if not rows:
        return None
    return np.asarray(rows)


def _combine(pmat: np.ndarray, composition: str) -> np.ndarray:
    """Compose per-signal p-values into one membership score per sample.
    Higher score  ==  stronger membership (contamination) evidence.

    Faithful to run_ours_different_agg.py:
      edgington/agg : -mean(p)                                (Edgington summation)
      fisher        : -combine_pvalues(p, 'fisher').pvalue
      pearson       : -combine_pvalues(p, 'pearson').pvalue
      george        : -combine_pvalues(p, 'mudholkar_george').pvalue
    """
    if pmat is None or pmat.size == 0:
        return np.array([])
    comp = (composition or "edgington").lower()
    if comp in ("edgington", "agg", "average", "mean"):
        return -np.mean(pmat, axis=0)

    from scipy.stats import combine_pvalues
    method = {
        "fisher":  "fisher",
        "pearson": "pearson",
        "george":  "mudholkar_george",
    }.get(comp, "fisher")
    pm = np.clip(pmat, 1e-10, 1 - 1e-10)
    n = pm.shape[1]
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = -float(combine_pvalues(pm[:, i], method=method).pvalue)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Method
# ══════════════════════════════════════════════════════════════════════════════

class CAMIAFaithfulMethod(BaseMethod):
    name = "camia_faithful"

    def __init__(
        self,
        uncon_dev_path:          str,
        con_dev_path:            str = "",
        composition:             str = "",
        max_tokens:              int = 512,
        batch_size:              int = 2,
        max_calibration_samples: int = 150,
        num_copies:              int = 6,     # 6 copies == official num_repeatitions=5
        rep_context_cap:         int = 4096,  # model context ceiling for repeated pass
        **kwargs,
    ):
        self.uncon_dev_path          = uncon_dev_path
        self.con_dev_path            = con_dev_path
        self.composition             = composition
        self.max_tokens              = max_tokens
        self.batch_size              = batch_size
        self.max_calibration_samples = max_calibration_samples
        # keep num_copies * max_tokens within the context window so every
        # occurrence survives the repeated forward pass (no silent truncation)
        self.rep_context_cap = rep_context_cap
        max_copies = max(3, (rep_context_cap - 8) // max(1, max_tokens))
        self.num_copies = int(min(num_copies, max_copies))

    # ───────────────────────────── i/o helpers ────────────────────────────────

    def _load_texts(self, path: str, limit: int) -> list[str]:
        texts: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = extract_text(json.loads(line))
                if text:
                    texts.append(text)
                    if len(texts) >= limit:
                        break
        return texts

    def _lp_ids_batch(
        self, model_bundle: ModelBundle, texts: list[str],
        max_tokens: int, desc: str,
    ) -> tuple[list[list[float]], list[list[int]]]:
        stats = model_bundle.llm_adapter.token_logprob_distribution_batch(
            texts=texts, batch_size=self.batch_size,
            max_tokens=max_tokens, progress_desc=desc,
        )
        lps = [s.get("token_log_probs", []) for s in stats]
        ids = [s.get("input_ids", [])       for s in stats]
        return lps, ids

    def _truncate(self, tok, text: str) -> str:
        enc = tok.encode(text, add_special_tokens=False)[: self.max_tokens]
        if not enc:
            return text
        return tok.decode(enc)

    def _compute_blocks(
        self, model_bundle: ModelBundle, texts: list[str], desc: str,
    ) -> dict[str, np.ndarray]:
        """Two forward passes (standalone + N-copy repeat) → stacked signal blocks."""
        tok = model_bundle.tokenizer
        trunc = [self._truncate(tok, t) for t in texts]

        # standalone pass (base signals, count_above_pre, occurrence pattern)
        lp_orig, ids_orig = self._lp_ids_batch(
            model_bundle, trunc, self.max_tokens, f"{desc}_orig",
        )
        # repeated pass (overall_diff occurrences, slope, approximate_entropy)
        rep_texts = [" ".join([t] * self.num_copies) for t in trunc]
        rep_cap = min(self.num_copies * self.max_tokens + 8, self.rep_context_cap)
        rep_lp, rep_ids = self._lp_ids_batch(
            model_bundle, rep_texts, rep_cap, f"{desc}_rep",
        )

        per_sample = [
            _sample_blocks(lp_orig[i], ids_orig[i], rep_lp[i], rep_ids[i])
            for i in range(len(trunc))
        ]
        return _stack_blocks(per_sample)

    # ───────────────────────────── run ────────────────────────────────────────

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:

        # ── auto-derive con_dev_path / composition from runtime context ─────
        if not self.con_dev_path:
            self.con_dev_path = _derive_con_dev_path(dataset.path, setting="matched")
        model_size = _detect_model_size(model_bundle.model_path)
        shifted    = _detect_shifted(dataset.path)
        _gen_name  = os.path.basename((model_bundle.model_path or "").rstrip("/"))
        if not self.composition:
            self.composition = GEN_COMPOSITION_CONFIG.get(_gen_name) or _lookup_composition(model_size, shifted)
        print(f"[camia] composition={self.composition}, "
              f"num_copies={self.num_copies}, con_dev_path={self.con_dev_path}")

        is_uncontam_target = _is_nonmember_path(dataset.path)
        alt_composition: str = ""
        abl_con_dev_path: str = ""
        if is_uncontam_target:
            primary_comp     = GEN_COMPOSITION_CONFIG.get(_gen_name) or _lookup_composition(model_size, shifted=False)
            alt_composition  = _lookup_composition(model_size, shifted=True)
            self.composition = primary_comp
            abl_con_dev_path = _derive_con_dev_path(dataset.path, setting="shifted")
            print(f"[camia] non-member target: primary={primary_comp}, "
                  f"sidecar={alt_composition}, abl_con_dev={abl_con_dev_path}")

        # ── prepare target records ──────────────────────────────────────────
        prepared: list[dict] = []
        for idx, record in enumerate(dataset.records):
            text = extract_text(record)
            if text:
                prepared.append({
                    "idx":  idx,
                    "id":   extract_sample_id(record, fallback=idx),
                    "text": text,
                })
        if not prepared:
            return self._empty(dataset)
        texts = [p["text"] for p in prepared]

        con_texts   = self._load_texts(self.con_dev_path,   self.max_calibration_samples)
        uncon_texts = self._load_texts(self.uncon_dev_path, self.max_calibration_samples)
        if not con_texts or not uncon_texts:
            return self._empty(dataset)

        print(f"[camia] member dev   ({len(con_texts)} samples)…")
        con_blocks   = self._compute_blocks(model_bundle, con_texts,   "camia_con_dev")
        print(f"[camia] non-member dev ({len(uncon_texts)} samples)…")
        uncon_blocks = self._compute_blocks(model_bundle, uncon_texts, "camia_uncon_dev")

        # ── direction calibration (members vs non-members, per block) ────────
        directions = _directions(con_blocks, uncon_blocks)
        calib_blocks = _apply_directions(uncon_blocks, directions)   # non-member null

        # ── target forward passes ────────────────────────────────────────────
        print(f"[camia] target             ({len(texts)} samples)…")
        target_blocks_raw = self._compute_blocks(model_bundle, texts, "camia_target")
        target_blocks     = _apply_directions(target_blocks_raw, directions)

        pmat = _pvalues(target_blocks, calib_blocks)

        def _build_payload(pm, composition, con_dev_used, n_dir_samples):
            scores = _combine(pm, composition)
            n_sig = 0 if pm is None else pm.shape[0]
            score_list: list[float] = []
            samples:    list[dict]  = []
            for i, p in enumerate(prepared):
                sc = float(scores[i]) if i < len(scores) else float("nan")
                score_list.append(sc)
                samples.append({
                    "record_index":         p["idx"],
                    "sample_id":            p["id"],
                    "text":                 p["text"],
                    "camia_faithful_score": sc,
                    "camia_score":          sc,   # alias for (method, field) readers
                    "score":                sc,   # generic alias
                })
            return {
                "method":                self.name,
                "dataset":               dataset.name,
                "dataset_path":          dataset.path,
                "dataset_data_type":     dataset.data_type,
                "composition":           composition,
                "max_tokens":            self.max_tokens,
                "num_copies":            self.num_copies,
                "con_dev_path":          con_dev_used,
                "uncon_dev_path":        self.uncon_dev_path,
                "n_calibration_samples": len(uncon_texts),
                "n_direction_samples":   n_dir_samples,
                "n_signals":             int(n_sig),
                "num_total_records":     len(dataset.records),
                "num_scored_records":    len(samples),
                "num_skipped_records":   len(dataset.records) - len(samples),
                "summary": {
                    "mean_camia_score": (
                        float(np.nanmean(score_list)) if score_list else float("nan")
                    ),
                },
                "samples": samples,
            }

        payload = _build_payload(
            pmat, self.composition, self.con_dev_path, len(con_texts),
        )

        # ── shifted-setting sidecar for non-member targets ───────────────
        if alt_composition and is_uncontam_target and abl_con_dev_path:
            abl_con_texts = self._load_texts(
                abl_con_dev_path, self.max_calibration_samples,
            )
            if abl_con_texts:
                print(f"[camia] shifted con_dev    ({len(abl_con_texts)} samples)…")
                abl_con_blocks = self._compute_blocks(
                    model_bundle, abl_con_texts, "camia_abl_con_dev",
                )
                abl_dirs         = _directions(abl_con_blocks, uncon_blocks)
                abl_calib_blocks = _apply_directions(uncon_blocks, abl_dirs)
                abl_target       = _apply_directions(target_blocks_raw, abl_dirs)
                abl_pmat         = _pvalues(abl_target, abl_calib_blocks)
                payload["_sidecar_outputs"] = {
                    f"{self.name}_shifted": _build_payload(
                        abl_pmat, alt_composition,
                        abl_con_dev_path, len(abl_con_texts),
                    ),
                }
                print(f"[camia] sidecar: composition={alt_composition}")
            else:
                print(f"[camia] sidecar: no shifted con_dev at "
                      f"{abl_con_dev_path}, skipping")

        return payload

    @staticmethod
    def _empty(dataset: DatasetBundle) -> dict[str, Any]:
        return {
            "method":              "camia_faithful",
            "dataset":             dataset.name,
            "dataset_path":        dataset.path,
            "dataset_data_type":   dataset.data_type,
            "num_total_records":   len(dataset.records),
            "num_scored_records":  0,
            "num_skipped_records": len(dataset.records),
            "summary":             {},
            "samples":             [],
        }


# ═══════════════════════════════ tuned configs ═══════════════════════════════

COMPOSITION_CONFIG: dict[str, str] = {
    "1B":  "george",
    "7B":  "fisher",
    "13B": "george",
    "32B": "george",
}

COMPOSITION_CONFIG_SHIFTED: dict[str, str] = {
    "1B":  "edgington",
    "7B":  "george",
    "13B": "george",
    "32B": "fisher",
}

# Per-MODEL-NAME composition overrides for non-OLMo generalization models (baked by
# bake_gen_configs.py from iclr_tune/camia_*.log). Empty by default -> zero effect on OLMo 2.
GEN_COMPOSITION_CONFIG: dict[str, str] = {}


def _lookup_composition(model_size: str, shifted: bool) -> str:
    # 2026-09-24: composition must match the non-member side (scored with the matched config) -> ignore the shifted flag.
    cfg = COMPOSITION_CONFIG
    for key in cfg:
        if key.lower() in model_size.lower():
            return cfg[key]
    return "edgington"


# ─────────────────────────── path / size auto-derivation ──────────────────────

def _detect_shifted(dataset_path: str) -> bool:
    p = (dataset_path or "").lower()
    return ("_shifted_" in p) or ("/shifted/" in p)


def _is_nonmember_path(dataset_path: str) -> bool:
    p = (dataset_path or "").lower()
    return ("/non-member/" in p) or ("non-member_" in os.path.basename(p))


def _detect_model_size(model_path: str) -> str:
    name = os.path.basename((model_path or "").rstrip("/")).lower()
    for size in ("32b", "13b", "7b", "1b"):
        if size in name:
            return size.upper()
    return ""


def _to_dev(p: str) -> str:
    """Map a *_test* path to the matching *_dev* path (mirrors run_all.to_dev)."""
    p = p.replace("/test/", "/dev/").replace("_test_", "_dev_")
    if p.endswith("_test.jsonl"):
        p = p[: -len("_test.jsonl")] + "_dev.jsonl"
    return p


def _derive_con_dev_path(dataset_path: str, setting: str = "matched") -> str:
    """Derive the member dev path from the eval dataset path.

    1. Contaminated eval path: swap /eval/->/dev/ and _eval->_dev.
    2. Unmember eval path with matched/shifted settings: insert
       /{setting}/ and _{setting}_, then swap eval->dev (tried first).
    3. Unmember eval path without settings: swap
       /non-member/->/member/ and eval->dev (fallback).
    """
    p = dataset_path

    # Contaminated target -> its own member dev split.
    if "/member/" in p:
        return _to_dev(p)

    # Unmember target -> member dev of the same domain (+setting).
    # (a) domains WITH a matched/shifted split: insert /{setting}/ and _{setting}.
    #     DPO chosen/rejected files put {setting} *before* _chosen_/_rejected_
    #     (member_..._dpo_1b_matched_chosen_dev.jsonl), so special-case it;
    #     all other domains carry {setting} as a suffix (..._dev_matched.jsonl).
    c1 = p.replace("/non-member/", f"/member/{setting}/")
    c1 = c1.replace("non-member_", "member_")
    base = os.path.basename(c1)
    if "_chosen_" in base:
        base = base.replace("_chosen_", f"_{setting}_chosen_")
    elif "_rejected_" in base:
        base = base.replace("_rejected_", f"_{setting}_rejected_")
    elif base.endswith("_test.jsonl"):
        base = base[: -len("_test.jsonl")] + f"_test_{setting}.jsonl"
    c1 = os.path.join(os.path.dirname(c1), base)
    c1 = _to_dev(c1)

    # (b) domains WITHOUT a split (gsm8k / rlvr): plain non-member->member.
    c2 = p.replace("/non-member/", "/member/")
    c2 = c2.replace("non-member_", "member_")
    c2 = _to_dev(c2)

    if os.path.exists(c1):
        return c1
    if os.path.exists(c2):
        return c2
    return c1


# ────────────────────────────── factory function ──────────────────────────────

def build_method(
    uncon_dev_path:          str,
    con_dev_path:            str  = "",
    model_size:              str  = "",
    shifted:                 bool = False,
    composition:             str  = "",
    max_tokens:              int  = 512,
    batch_size:              int  = 2,
    max_calibration_samples: int  = 150,
    num_copies:              int  = 6,
    **kwargs,
) -> BaseMethod:
    if not composition and (model_size or shifted):
        composition = _lookup_composition(model_size, shifted)
    return CAMIAFaithfulMethod(
        uncon_dev_path          = uncon_dev_path,
        con_dev_path            = con_dev_path,
        composition             = composition,
        max_tokens              = max_tokens,
        batch_size              = batch_size,
        max_calibration_samples = max_calibration_samples,
        num_copies              = num_copies,
    )
