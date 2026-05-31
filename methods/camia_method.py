"""
Context-Aware Membership Inference Attacks against Pre-trained Large Language Models (CAMIA)
Paper: https://aclanthology.org/2025.emnlp-main.370.pdf
Github Implementation:  https://github.com/changhongyan123/context_aware_mia
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text
from model_utils import ModelBundle


# ───────────────────────────────────────── utility functions ──────────────────

def _lempel_ziv_complexity(x: np.ndarray, bins: int) -> float:
    """LZ complexity — exact match with util_features.lempel_ziv_complexity."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan")
    x_min = float(x.min())
    bins_edges = np.linspace(x_min, 0.0, 100)
    x_disc = np.digitize(x, bins=bins_edges, right=True)
    bins_new = np.linspace(float(np.min(x_disc)), float(np.max(x_disc)), bins + 1)[1:]
    sequence = np.searchsorted(bins_new, x_disc, side="left")

    sub_strings: set = set()
    n = len(sequence)
    ind, inc = 0, 1
    while ind + inc <= n:
        sub_str = tuple(sequence[ind: ind + inc])
        if sub_str in sub_strings:
            inc += 1
        else:
            sub_strings.add(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings) / n


def _approx_entropy(x: np.ndarray, m: int = 8, r: float = 0.8) -> float:
    """Approximate entropy — exact match with util_features.approximate_entropy."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N <= m + 1:
        return 0.0
    r_scaled = r * float(np.std(x))
    if r_scaled <= 0:
        return 0.0

    def _phi(m_val: int) -> float:
        segs = np.array([x[i: i + m_val] for i in range(N - m_val + 1)])
        C = (
            np.sum(
                np.max(np.abs(segs[:, np.newaxis] - segs[np.newaxis, :]), axis=2)
                <= r_scaled,
                axis=0,
            )
            / (N - m_val + 1)
        )
        C = np.maximum(C, 1e-10)
        return float(np.sum(np.log(C)) / (N - m_val + 1.0))

    return float(abs(_phi(m) - _phi(m + 1)))


# ─────────────────────────────── raw signal computation ───────────────────────

def _sample_signals_raw(
    lp: np.ndarray,         # per-token log-probs of original text  (≤ 0)
    lp_r1: np.ndarray,      # log-probs of 2nd occurrence in text+" "+text
    lp_r2: np.ndarray,      # log-probs of 3rd occurrence in text+" "+text+" "+text
    token_ids: list[int],   # token IDs of original text (for diversity)
) -> dict[str, float]:
    sigs: dict[str, float] = {}
    if len(lp) == 0:
        return sigs

    tids = np.asarray(token_ids, dtype=int)

    def _mean(arr: np.ndarray, cut: int | None) -> float:
        s = arr[:cut] if cut else arr
        return float(np.mean(s)) if len(s) > 0 else float("nan")

    def _ppl(arr: np.ndarray, cut: int | None) -> float:
        s = arr[:cut] if cut else arr
        return float(np.exp(-np.mean(s))) if len(s) > 0 else float("nan")

    def _div(arr: np.ndarray, cut: int | None) -> float:
        s = tids[:cut] if cut else tids
        return (len(set(s.tolist())) / len(s)) if len(s) > 0 else float("nan")

    for tag, cut in (("full", None), ("200", 200), ("300", 300)):
        m = _mean(lp, cut)
        if np.isfinite(m):
            sigs[f"loss_{tag}"] = m
            sigs[f"ppl_{tag}"]  = float(np.exp(-m))

    for tag, cut in (("full", None), ("200", 200), ("300", 300)):
        m, d = _mean(lp, cut), _div(lp, cut)
        if np.isfinite(m) and np.isfinite(d) and d > 0:
            sigs[f"cal_loss_{tag}"] = m / d
            sigs[f"cal_ppl_{tag}"]  = float(np.exp(-m)) / d

    seq200 = lp[:200] if len(lp) >= 200 else lp
    if len(seq200) > 0:
        for thresh in (-1.0, -2.0, -3.0):
            sigs[f"count_above_{thresh}"] = float(np.mean(seq200 >= thresh))

    for tag, cut in (("full", None), ("200", 200), ("300", 300)):
        seq = lp[:cut] if cut else lp
        if len(seq) > 0:
            sigs[f"count_mean_{tag}"] = float(np.mean(seq > np.mean(seq)))

    for tag, cut in (("full", None), ("200", 200), ("300", 300)):
        seq = lp[:cut] if cut else lp
        if len(seq) > 1:
            cum_means = np.cumsum(seq[:-1]) / np.arange(1, len(seq), dtype=float)
            sigs[f"count_prev_mean_{tag}"] = float(np.mean(seq[1:] > cum_means))

    if len(seq200) > 5:
        for bins in (3, 4, 5):
            sigs[f"lz_{bins}"] = _lempel_ziv_complexity(seq200, bins)

    for tag, cut in (("200", 200), ("300", 300)):
        seq = lp[:cut] if len(lp) >= cut else lp
        if len(seq) > 20:
            sigs[f"apen_{tag}"] = _approx_entropy(seq, m=8, r=0.8)

    for tag, cut in (("600", 600), ("800", 800), ("1000", 1000)):
        seq = lp[:cut] if len(lp) >= cut else lp
        if len(seq) > 2:
            sigs[f"slope_{tag}"] = float(
                np.polyfit(np.arange(len(seq), dtype=float), seq, 1)[0]
            )

    for rep_name, rep_lp in (("r1", lp_r1), ("r2", lp_r2)):
        if len(rep_lp) == 0:
            continue
        s_o200 = lp[:200]     if len(lp)     >= 200 else lp
        s_r200 = rep_lp[:200] if len(rep_lp) >= 200 else rep_lp
        for tag, cut in (("full", None), ("200", 200), ("300", 300)):
            s_o = lp[:cut]     if cut else lp
            s_r = rep_lp[:cut] if cut else rep_lp
            if len(s_o) == 0 or len(s_r) == 0:
                continue
            m_o, m_r = float(np.mean(s_o)), float(np.mean(s_r))
            tid_o = tids[:cut] if cut else tids
            d_o   = (len(set(tid_o.tolist())) / max(1, len(tid_o))) if len(tid_o) > 0 else float("nan")
            sigs[f"{rep_name}_diff_loss_{tag}"] = m_o - m_r
            sigs[f"{rep_name}_diff_ppl_{tag}"]  = float(np.exp(-m_o) - np.exp(-m_r))
            if np.isfinite(d_o) and d_o > 0:
                sigs[f"{rep_name}_diff_cal_loss_{tag}"] = m_o / d_o - m_r / d_o
                sigs[f"{rep_name}_diff_cal_ppl_{tag}"]  = float(np.exp(-m_o) - np.exp(-m_r)) / d_o
            if len(s_o) > 0 and len(s_r) > 0:
                sigs[f"{rep_name}_diff_count_mean_{tag}"] = (
                    float(np.mean(s_o > np.mean(s_o))) -
                    float(np.mean(s_r > np.mean(s_r)))
                )
        if len(s_o200) > 0 and len(s_r200) > 0:
            for thresh in (-1.0, -2.0, -3.0):
                sigs[f"{rep_name}_diff_count_above_{thresh}"] = (
                    float(np.mean(s_o200 >= thresh)) -
                    float(np.mean(s_r200 >= thresh))
                )
        if len(s_o200) > 5 and len(s_r200) > 5:
            for bins in (3, 4, 5):
                sigs[f"{rep_name}_diff_lz_{bins}"] = (
                    _lempel_ziv_complexity(s_o200, bins) -
                    _lempel_ziv_complexity(s_r200, bins)
                )

    return sigs


# ─────────────────────────────── direction calibration ────────────────────────

def _calibrate_directions(
    member_sigs:    dict[str, np.ndarray],
    nonmember_sigs: dict[str, np.ndarray],
) -> dict[str, int]:
    directions: dict[str, int] = {}
    all_keys = set(member_sigs) | set(nonmember_sigs)
    for key in all_keys:
        if key not in member_sigs or key not in nonmember_sigs:
            directions[key] = 1
            continue
        m_mean  = float(np.nanmean(member_sigs[key]))
        nm_mean = float(np.nanmean(nonmember_sigs[key]))
        if not (np.isfinite(m_mean) and np.isfinite(nm_mean)):
            directions[key] = 1
            continue
        directions[key] = -1 if m_mean > nm_mean else 1
    return directions


def _apply_directions(
    sigs: dict[str, np.ndarray],
    directions: dict[str, int],
) -> dict[str, np.ndarray]:
    return {k: directions.get(k, 1) * v for k, v in sigs.items()}


def _signals_matrix(
    lp_list:     list[list[float]],
    lp_r1_list:  list[list[float]],
    lp_r2_list:  list[list[float]],
    tok_ids_list: list[list[int]],
) -> dict[str, np.ndarray]:
    n = len(lp_list)
    per_sample: list[dict[str, float]] = []
    all_keys: set[str] = set()
    for i in range(n):
        lp    = np.asarray(lp_list[i],    dtype=float) if lp_list[i]    else np.array([])
        lp_r1 = np.asarray(lp_r1_list[i], dtype=float) if lp_r1_list[i] else np.array([])
        lp_r2 = np.asarray(lp_r2_list[i], dtype=float) if lp_r2_list[i] else np.array([])
        s = _sample_signals_raw(lp, lp_r1, lp_r2, tok_ids_list[i])
        per_sample.append(s)
        all_keys.update(s.keys())
    return {
        k: np.array([ps.get(k, float("nan")) for ps in per_sample])
        for k in all_keys
    }


# ─────────────────────────────── p-values + combination ──────────────────────

def _pvalues(
    target: dict[str, np.ndarray],
    calib:  dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in target:
        if key not in calib:
            continue
        c_clean = calib[key][np.isfinite(calib[key])]
        if len(c_clean) == 0:
            continue
        out[key] = np.array([
            float(np.mean(c_clean <= v)) if np.isfinite(v) else 0.5
            for v in target[key]
        ])
    return out


def _combine(
    pvals: dict[str, np.ndarray],
    composition: str,
    n_samples: int,
) -> np.ndarray:
    if not pvals:
        return np.full(n_samples, float("nan"))
    pmat = np.clip(np.array(list(pvals.values())), 1e-10, 1 - 1e-10)
    if composition == "fisher":
        return -2.0 * np.sum(np.log(pmat), axis=0)
    if composition == "george":
        from scipy.stats import combine_pvalues as _cpv
        n = pmat.shape[1]
        return np.array([
            -float(_cpv(pmat[:, i], method="mudholkar_george").pvalue)
            for i in range(n)
        ])
    return -np.mean(pmat, axis=0)   # Edgington (default)


# ────────────────────────────────── main method ───────────────────────────────

class CAMIAMethod(BaseMethod):
    name = "camia"

    def __init__(
        self,
        uncon_dev_path:          str,
        con_dev_path:            str = "",
        composition:             str = "",
        max_tokens:              int = 512,
        batch_size:              int = 2,
        max_calibration_samples: int = 150,
        **kwargs,
    ):
        self.uncon_dev_path           = uncon_dev_path
        self.con_dev_path             = con_dev_path
        self.composition              = composition
        self.max_tokens               = max_tokens
        self.batch_size               = batch_size
        self.max_calibration_samples  = max_calibration_samples

    # ───────────────────────────── helpers ────────────────────────────────────

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

    def _lp_batch(
        self, model_bundle: ModelBundle, texts: list[str],
        max_tokens: int, desc: str,
    ) -> list[list[float]]:
        stats = model_bundle.llm_adapter.token_logprob_distribution_batch(
            texts=texts, batch_size=self.batch_size,
            max_tokens=max_tokens, progress_desc=desc,
        )
        return [s.get("token_log_probs", []) for s in stats]

    def _repeated_lp(
        self, model_bundle: ModelBundle, texts: list[str],
        orig_lps: list[list[float]], desc_pfx: str,
    ) -> tuple[list[list[float]], list[list[float]]]:
        tok  = model_bundle.tokenizer
        max1 = min(self.max_tokens * 2 + 20, 2048)
        max2 = min(self.max_tokens * 3 + 40, 2048)
        texts_r1 = [t + " " + t             for t in texts]
        texts_r2 = [t + " " + t + " " + t   for t in texts]
        lps_r1 = self._lp_batch(model_bundle, texts_r1, max1, f"{desc_pfx}_r1")
        lps_r2 = self._lp_batch(model_bundle, texts_r2, max2, f"{desc_pfx}_r2")
        out_r1, out_r2 = [], []
        for i, text in enumerate(texts):
            n = len(orig_lps[i])
            if n == 0:
                out_r1.append([]); out_r2.append([]); continue
            prefix_r1 = text + " "
            prefix_r2 = text + " " + text + " "
            start1 = len(tok.encode(prefix_r1, add_special_tokens=False))
            start2 = len(tok.encode(prefix_r2, add_special_tokens=False))
            lp1 = lps_r1[i]
            out_r1.append(lp1[start1: start1 + n] if start1 < len(lp1) else [])
            lp2 = lps_r2[i]
            out_r2.append(lp2[start2: start2 + n] if start2 < len(lp2) else [])
        return out_r1, out_r2

    def _compute_sigs(self, model_bundle, texts, desc):
        """Compute raw signal matrix for a list of texts."""
        tok = model_bundle.tokenizer
        lps = self._lp_batch(model_bundle, texts, self.max_tokens, desc)
        r1, r2 = self._repeated_lp(model_bundle, texts, lps, desc)
        tids = [tok.encode(t, add_special_tokens=False)[:self.max_tokens]
                for t in texts]
        return _signals_matrix(lps, r1, r2, tids)

    # ───────────────────────────── run ────────────────────────────────────────

    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:

        # ── auto-derive con_dev_path / composition from runtime context ─────
        if not self.con_dev_path:
            self.con_dev_path = _derive_con_dev_path(dataset.path, setting="matched")
        model_size = _detect_model_size(model_bundle.model_path)
        shifted   = _detect_shifted(dataset.path)
        if not self.composition:
            self.composition = _lookup_composition(model_size, shifted)
        print(f"[camia] composition={self.composition}, "
              f"con_dev_path={self.con_dev_path}")

        is_uncontam_target = _is_uncontaminated_path(dataset.path)
        alt_composition: str = ""
        abl_con_dev_path: str = ""
        if is_uncontam_target:
            primary_comp     = _lookup_composition(model_size, shifted=False)
            alt_composition  = _lookup_composition(model_size, shifted=True)
            self.composition = primary_comp
            abl_con_dev_path = _derive_con_dev_path(dataset.path, setting="shifted")
            print(f"[camia] uncontaminated target: primary={primary_comp}, "
                  f"sidecar={alt_composition}, "
                  f"abl_con_dev={abl_con_dev_path}")

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

        print(f"[camia] contaminated dev forward passes   ({len(con_texts)} samples)…")
        con_sigs   = self._compute_sigs(model_bundle, con_texts,   "camia_con_dev")
        print(f"[camia] uncontaminated dev forward passes ({len(uncon_texts)} samples)…")
        uncon_sigs = self._compute_sigs(model_bundle, uncon_texts, "camia_uncon_dev")

        # ── primary direction calibration ───────────────────────────────────
        directions = _calibrate_directions(con_sigs, uncon_sigs)
        calib_sigs = _apply_directions(uncon_sigs, directions)

        # ── target forward passes ─────────────────────────────────────────
        print(f"[camia] target forward passes             ({len(texts)} samples)…")
        target_sigs_raw = self._compute_sigs(model_bundle, texts, "camia_target")
        target_sigs     = _apply_directions(target_sigs_raw, directions)

        # ── primary p-values + combination ─────────────────────────────────
        pvals  = _pvalues(target_sigs, calib_sigs)

        def _build_payload(pv, composition, con_dev_used, n_dir_samples):
            scores = _combine(pv, composition, len(texts))
            score_list: list[float] = []
            samples:    list[dict]  = []
            for i, p in enumerate(prepared):
                sc = float(scores[i]) if i < len(scores) else float("nan")
                score_list.append(sc)
                samples.append({
                    "record_index": p["idx"],
                    "sample_id":    p["id"],
                    "text":         p["text"],
                    "camia_score":  sc,
                })
            return {
                "method":                self.name,
                "dataset":               dataset.name,
                "dataset_path":          dataset.path,
                "dataset_data_type":     dataset.data_type,
                "composition":           composition,
                "max_tokens":            self.max_tokens,
                "con_dev_path":          con_dev_used,
                "uncon_dev_path":        self.uncon_dev_path,
                "n_calibration_samples": len(uncon_texts),
                "n_direction_samples":   n_dir_samples,
                "n_signals":             len(pv),
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
            pvals, self.composition, self.con_dev_path, len(con_texts),
        )

        if alt_composition and is_uncontam_target and abl_con_dev_path:
            abl_con_texts = self._load_texts(
                abl_con_dev_path, self.max_calibration_samples,
            )
            if abl_con_texts:
                print(f"[camia] shifted con_dev forward passes   "
                      f"({len(abl_con_texts)} samples)…")
                abl_con_sigs = self._compute_sigs(
                    model_bundle, abl_con_texts, "camia_abl_con_dev",
                )
                abl_directions  = _calibrate_directions(abl_con_sigs, uncon_sigs)
                abl_calib_sigs  = _apply_directions(uncon_sigs, abl_directions)
                abl_target_sigs = _apply_directions(target_sigs_raw, abl_directions)
                abl_pvals       = _pvalues(abl_target_sigs, abl_calib_sigs)

                payload["_sidecar_outputs"] = {
                    f"{self.name}_shifted": _build_payload(
                        abl_pvals, alt_composition,
                        abl_con_dev_path, len(abl_con_texts),
                    ),
                }
                print(f"[camia] sidecar: composition={alt_composition}, "
                      f"n_signals={len(abl_pvals)}")
            else:
                print(f"[camia] sidecar: no shifted con_dev at "
                      f"{abl_con_dev_path}, skipping")

        return payload

    @staticmethod
    def _empty(dataset: DatasetBundle) -> dict[str, Any]:
        return {
            "method":              "camia",
            "dataset":             dataset.name,
            "dataset_path":        dataset.path,
            "dataset_data_type":   dataset.data_type,
            "num_total_records":   len(dataset.records),
            "num_scored_records":  0,
            "num_skipped_records": len(dataset.records),
            "summary":             {},
            "samples":             [],
        }


# ───────────────────────────────── tuned configs ─────────────────────────────

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


def _lookup_composition(model_size: str, shifted: bool) -> str:
    cfg = COMPOSITION_CONFIG_SHIFTED if shifted else COMPOSITION_CONFIG
    for key in cfg:
        if key.lower() in model_size.lower():
            return cfg[key]
    return "edgington"


# ──────────────────────────── path / size auto-derivation ─────────────────────

def _detect_shifted(dataset_path: str) -> bool:
    p = (dataset_path or "").lower()
    return ("_shifted_" in p) or ("/shifted/" in p)


def _is_uncontaminated_path(dataset_path: str) -> bool:
    p = (dataset_path or "").lower()
    return ("/uncontaminated/" in p) or ("uncontaminated_" in os.path.basename(p))


def _detect_model_size(model_path: str) -> str:
    name = os.path.basename((model_path or "").rstrip("/")).lower()
    for size in ("32b", "13b", "7b", "1b"):
        if size in name:
            return size.upper()
    return ""


def _derive_con_dev_path(dataset_path: str, setting: str = "matched") -> str:
    """
    Derive the contaminated dev path from the eval dataset path.

    Parameters
    ----------
    dataset_path : str
        Path to the eval JSONL (contaminated or uncontaminated).
    setting : str
        Which setting's con_dev to derive: "matched" or "shifted".
        Only affects uncontaminated input paths (contaminated paths
        already encode their own setting).

    Three cases:
      1. Contaminated eval path (has /contaminated/ in path):
         Just swap /eval/ → /dev/ and _eval → _dev.  The setting
         embedded in the path is preserved as-is.
      2. Uncontaminated eval path, domain has matched/shifted settings:
         Insert /{setting}/ directory and _{setting}_ in filename,
         then swap eval→dev.  Tries this candidate first.
      3. Uncontaminated eval path, domain has NO settings (Math-Mix, RLVR):
         Just swap /uncontaminated/ → /contaminated/ and eval→dev.
         Falls back to this if candidate 2 doesn't exist on disk.

    For DPO chosen/rejected files, _{setting}_ is inserted before
    _chosen_/_rejected_ to match the actual filename convention.
    """
    p = dataset_path

    if "/contaminated/" in p:
        return p.replace("/eval/", "/dev/").replace("_eval.jsonl", "_dev.jsonl")

    # ── Uncontaminated path: try with-setting first, then without ─────────

    c1 = p.replace("/uncontaminated/", f"/contaminated/{setting}/")
    c1 = c1.replace("uncontaminated_", "contaminated_")
    base = os.path.basename(c1)
    if "_chosen_" in base:
        base = base.replace("_chosen_", f"_{setting}_chosen_")
    elif "_rejected_" in base:
        base = base.replace("_rejected_", f"_{setting}_rejected_")
    else:
        base = base.replace("_eval.jsonl", f"_{setting}_eval.jsonl")
    c1 = os.path.join(os.path.dirname(c1), base)
    c1 = c1.replace("/eval/", "/dev/").replace("_eval.jsonl", "_dev.jsonl")

    c2 = p.replace("/uncontaminated/", "/contaminated/")
    c2 = c2.replace("uncontaminated_", "contaminated_")
    c2 = c2.replace("/eval/", "/dev/").replace("_eval.jsonl", "_dev.jsonl")

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
    shifted:                bool = False,
    composition:             str  = "",
    max_tokens:              int  = 512,
    batch_size:              int  = 2,
    max_calibration_samples: int  = 150,
    **kwargs,
) -> BaseMethod:
    if not composition and (model_size or shifted):
        composition = _lookup_composition(model_size, shifted)
    return CAMIAMethod(
        uncon_dev_path          = uncon_dev_path,
        con_dev_path            = con_dev_path,
        composition             = composition,
        max_tokens              = max_tokens,
        batch_size              = batch_size,
        max_calibration_samples = max_calibration_samples,
    )