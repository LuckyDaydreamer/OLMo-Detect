"""
Detecting Training Data of Large Language Models via Expectation Maximization (EMMIA)
Paper: https://aclanthology.org/2026.eacl-long.49.pdf
Github Implementation:  https://github.com/gyuwankim/em-mia

Architecture:
Phase 1a: For each sample x, one forward pass yields:
    - Loss   = mean LL                                  (Loss init)
    - Zlib   = sum_LL / zlib_compressed_bytes           (Zlib init)
    - Min-K  = bottom-K%(token_logprobs).mean()         (Min-K init)
    - Min-K++ = bottom-K%(normalized_token_lp).mean()   (Min-K++ init)

Phase 1b (unchanged, expensive): For ALL pairs (p, x):
    recall_matrix[p, x] = LL(x | p) / LL(x)

Phase 2: For each init i ∈ {loss, zlib, mink, minkpp}:
    m <- init_scores[i]; m[NaN] = 0
    for it in 1..max_iter:
        prefix_scores[p] = AUC(recall_matrix[p, :], m > median(m))
        m = -prefix_scores; m[NaN] = 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
import zlib
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────── path setup ────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _PROJECT_DIR)

from data_utils import extract_text
from model_utils import load_model_bundle


# Inits we run (in this fixed order for stable JSON output)
INIT_NAMES = ["loss", "zlib", "mink", "minkpp"]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_texts(path: str, limit: Optional[int] = None) -> list[str]:
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = extract_text(json.loads(line))
            if text:
                texts.append(text)
                if limit is not None and len(texts) >= limit:
                    break
    return texts


def calc_auc(scores, labels):
    """AUC-ROC from sklearn, matching original eval.py."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc(fpr, tpr)


def select_reference_idx(labels: np.ndarray, ref_pool: int, seed: int = 42) -> np.ndarray:
    """Pick a balanced pool of REFERENCE TARGET indices for the sub-N^2 EM.

    EM-MIA's O(N^2) cost comes from evaluating, for every prefix p, an AUC over
    ALL N targets. We instead evaluate that AUC over a fixed balanced subsample of
    K = ref_pool targets — the internal "probe basis". Membership is still produced
    for ALL N samples (each is scored as a prefix against the K probes), so the final
    evaluation set is unchanged; only the per-prefix AUC's target axis is subsampled.
    The benchmark is 50/50 by construction, so a class-balanced subsample is the
    minimum-variance faithful estimator of the full-N per-prefix AUC.

    Returns sorted indices (length <= ref_pool; smaller only if a class is tiny).
    """
    rng = np.random.default_rng(seed)
    con = np.where(labels == 1)[0]
    unc = np.where(labels == 0)[0]
    kc = min(len(con), ref_pool // 2)
    ku = min(len(unc), ref_pool - kc)
    kc = min(len(con), ref_pool - ku)  # top up if uncon was the limiting side
    ci = rng.choice(con, kc, replace=False)
    ui = rng.choice(unc, ku, replace=False)
    return np.sort(np.concatenate([ci, ui]))


# ═══════════════════════════════════════════════════════════════════════════
# infer_unconditional / infer_conditional_batch
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_unconditional(model, tokenizer, target_text: str, max_tokens: int):
    """Compute unconditional LL(x), logits, labels for a single target."""
    tokenizer.padding_side = "right"
    enc = tokenizer(
        target_text, truncation=True, max_length=max_tokens,
        return_tensors="pt"
    )
    input_ids = enc.input_ids.to(model.device)

    outputs = model(input_ids)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    logits = logits[0, :-1].float()
    labels = input_ids[0, 1:]

    ll = -F.cross_entropy(logits, labels, reduction="none").mean().item()
    return ll, logits, labels


@torch.no_grad()
def infer_conditional_batch(
    model, tokenizer, target_text: str, prefix_texts: list[str],
    max_tokens: int,
):
    """Compute LL(target | prefix) for a batch of prefixes."""
    tokenizer.padding_side = "right"
    target_enc = tokenizer(
        target_text, truncation=True, max_length=max_tokens,
        return_tensors="pt"
    )
    target_input_ids = target_enc.input_ids

    tokenizer.padding_side = "left"
    prefix_enc = tokenizer(
        prefix_texts, padding=True, truncation=True,
        max_length=max_tokens, return_tensors="pt"
    )
    prefix_input_ids = prefix_enc.input_ids
    prefix_attention_mask = prefix_enc.attention_mask
    num_prefix, prefix_len = prefix_input_ids.size()

    batch_target_ids = target_input_ids.repeat(num_prefix, 1)
    input_ids = torch.cat((prefix_input_ids, batch_target_ids), dim=1)
    target_attention_mask = torch.ones_like(batch_target_ids)
    attention_mask = torch.cat(
        (prefix_attention_mask, target_attention_mask), dim=1
    )

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    logits = logits[:, prefix_len:-1].contiguous().float()
    labels = batch_target_ids[:, 1:].contiguous().to(model.device)

    ll = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none"
    ).reshape(num_prefix, -1).mean(-1)

    return ll.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1a — compute all 4 inits in a single forward-pass loop
# ═══════════════════════════════════════════════════════════════════════════

def compute_unconditional_all_inits(
    model, tokenizer, texts: list[str],
    max_tokens: int, mink_k: int = 20, minkpp_k: int = 20,
) -> dict[str, np.ndarray]:
    """
    Compute Loss / Zlib / Min-K / Min-K++ for all samples from ONE pass each.
    loss   = mean log-likelihood
    zlib   = sum_LL / len(zlib.compress(text_utf8))
    mink   = bottom-K% of token_logprobs, averaged
    minkpp = bottom-K% of (token_lp - mu) / sigma, averaged
                (sigma clamped at 1e-9 before sqrt for stability)
    """
    N = len(texts)
    out = {name: np.zeros(N) for name in INIT_NAMES}
    n_tokens_arr = np.zeros(N, dtype=np.int64)

    for i in tqdm(range(N), desc="[emmia] unconditional", leave=False):
        ll, logits, labels = infer_unconditional(
            model, tokenizer, texts[i], max_tokens
        )
        n_tokens = labels.shape[0]
        n_tokens_arr[i] = n_tokens

        # Shared computations
        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        # 1. Loss = mean LL
        out["loss"][i] = ll

        # 2. Zlib = sum_LL / compressed_bytes
        z_bytes = len(zlib.compress(texts[i].encode("utf-8")))
        sum_ll = ll * n_tokens
        out["zlib"][i] = sum_ll / z_bytes if z_bytes > 0 else float("nan")

        # 3. Min-K% = bottom-K% raw token logprobs, averaged
        k_mink = max(1, int(n_tokens * mink_k / 100.0))
        out["mink"][i] = token_logprobs.topk(k_mink, largest=False)[0].mean().item()

        # 4. Min-K%++ = bottom-K% normalised token logprobs, averaged
        mu = (probs * logprobs).sum(-1)
        sigma = (probs * torch.square(logprobs)).sum(-1) - torch.square(mu)
        normalized = (token_logprobs - mu) / sigma.clamp(min=1e-9).sqrt()
        k_pp = max(1, int(n_tokens * minkpp_k / 100.0))
        out["minkpp"][i] = normalized.topk(k_pp, largest=False)[0].mean().item()

    return out, n_tokens_arr


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1b
# ═══════════════════════════════════════════════════════════════════════════

def compute_recall_matrix(
    model, tokenizer, texts: list[str],
    unconditional_lls: np.ndarray,
    max_tokens: int, prefix_batch_size: int = 4,
    row_range: Optional[tuple] = None,
) -> np.ndarray:
    """row_range=(r0, r1): only PREFIX rows r0..r1-1 are computed (2026-09-25: row-sharded 32B runs); other rows stay 0."""
    N = len(texts)
    recall_matrix = np.zeros((N, N), dtype=np.float64)
    r0, r1 = (0, N) if row_range is None else (int(row_range[0]), min(int(row_range[1]), N))

    for target_idx in tqdm(range(N), desc="[emmia] ReCaLL matrix", leave=False):
        target_text = texts[target_idx]
        ll = unconditional_lls[target_idx]

        for batch_start in range(r0, r1, prefix_batch_size):
            batch_end = min(batch_start + prefix_batch_size, r1)
            prefix_batch = [texts[p] for p in range(batch_start, batch_end)]

            cond_lls = infer_conditional_batch(
                model, tokenizer, target_text, prefix_batch, max_tokens
            )

            for i, p_idx in enumerate(range(batch_start, batch_end)):
                recall_matrix[p_idx, target_idx] = cond_lls[i] / ll

    return recall_matrix


def compute_recall_matrix_sub(
    model, tokenizer, texts: list[str],
    unconditional_lls: np.ndarray, ref_idx: np.ndarray,
    max_tokens: int, prefix_batch_size: int = 4,
) -> np.ndarray:
    """Sub-N^2 ReCaLL: only the K = len(ref_idx) reference TARGET columns.

    recall_sub[p, j] = LL(texts[ref_idx[j]] | texts[p]) / LL(texts[ref_idx[j]])
    for ALL N prefixes p and the K reference targets. Cost is O(N*K) forward
    passes instead of O(N^2). The row axis (prefixes) is full N so membership is
    still produced for every sample.
    """
    N = len(texts)
    K = len(ref_idx)
    recall_sub = np.zeros((N, K), dtype=np.float64)

    for j, target_idx in enumerate(
        tqdm(ref_idx, desc="[emmia] ReCaLL sub", leave=False)
    ):
        target_text = texts[int(target_idx)]
        ll = unconditional_lls[int(target_idx)]

        for batch_start in range(0, N, prefix_batch_size):
            batch_end = min(batch_start + prefix_batch_size, N)
            prefix_batch = [texts[p] for p in range(batch_start, batch_end)]

            cond_lls = infer_conditional_batch(
                model, tokenizer, target_text, prefix_batch, max_tokens
            )

            for i, p_idx in enumerate(range(batch_start, batch_end)):
                recall_sub[p_idx, j] = cond_lls[i] / ll

    return recall_sub


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: now returns AUC trajectory across iterations
# ═══════════════════════════════════════════════════════════════════════════

def em_iterations(
    recall_matrix: np.ndarray,
    init_scores: np.ndarray,
    labels: np.ndarray,
    n_iterations: int = 10,
    verbose: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """
    Run EM-MIA iterations. EM update is IDENTICAL to official emmia.py
    (NegPref + AUC-ROC, default branch).

    Returns:
        membership_scores: (N,) final scores
        auc_trajectory:    list of length n_iterations+1, AUC at each step
                           (index 0 = init AUC, index k = after iteration k)
    """
    N = len(init_scores)
    membership_scores = init_scores.copy()
    membership_scores[np.isnan(membership_scores)] = 0

    auc_trajectory = [float(calc_auc(membership_scores, labels))]

    for it in range(1, n_iterations + 1):
        pseudo_labels = (
            membership_scores > np.median(membership_scores)
        ).astype(int)

        prefix_scores = np.array([
            calc_auc(recall_matrix[idx], pseudo_labels)
            for idx in range(N)
        ])

        membership_scores = -prefix_scores
        membership_scores[np.isnan(membership_scores)] = 0

        auc_trajectory.append(float(calc_auc(membership_scores, labels)))
        if verbose:
            print(f"    iter {it:2d}/{n_iterations}: AUC = {auc_trajectory[-1]:.4f}")

    return membership_scores, auc_trajectory


def em_iterations_sub(
    recall_sub: np.ndarray,
    ref_idx: np.ndarray,
    init_scores: np.ndarray,
    labels: np.ndarray,
    n_iterations: int = 10,
    verbose: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """Sub-N^2 EM. Identical NegPref update to em_iterations, except each prefix's
    AUC is taken over the K reference TARGET columns only.

    recall_sub: (N, K) = recall_matrix[:, ref_idx]
    ref_idx:    (K,) indices of the reference targets within the full N samples
    Membership is produced for all N samples (prefix axis is full).
    """
    N = len(init_scores)
    membership_scores = init_scores.copy()
    membership_scores[np.isnan(membership_scores)] = 0

    auc_trajectory = [float(calc_auc(membership_scores, labels))]

    for it in range(1, n_iterations + 1):
        pseudo_all = (
            membership_scores > np.median(membership_scores)
        ).astype(int)
        pseudo_ref = pseudo_all[ref_idx]

        prefix_scores = np.array([
            calc_auc(recall_sub[x], pseudo_ref)
            for x in range(N)
        ])

        membership_scores = -prefix_scores
        membership_scores[np.isnan(membership_scores)] = 0

        auc_trajectory.append(float(calc_auc(membership_scores, labels)))
        if verbose:
            print(f"    iter {it:2d}/{n_iterations}: AUC = {auc_trajectory[-1]:.4f}")

    return membership_scores, auc_trajectory


# ═══════════════════════════════════════════════════════════════════════════
# Cache helpers (Phase 1a + 1b persistence)
# ═══════════════════════════════════════════════════════════════════════════

def _cache_path(cache_dir: Optional[str], output_path: str,
                ref_pool: Optional[int] = None) -> Optional[str]:
    """Return the .npz cache path next to the output, or None if disabled.
    Sub-N^2 caches get a distinct `__phase1_subK{K}` suffix so they never collide
    with (or are mistaken for) a full N×N cache."""
    if cache_dir is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(output_path))[0]
    suffix = f"__phase1_subK{ref_pool}" if ref_pool else "__phase1"
    return os.path.join(cache_dir, f"{base}{suffix}.npz")


def _load_cache(npz_path: str) -> Optional[dict]:
    if not os.path.isfile(npz_path):
        return None
    print(f"[emmia] Loading cached Phase 1 from {npz_path}")
    z = np.load(npz_path, allow_pickle=False)
    out = {
        "n_tokens": z["n_tokens"],
        "init_scores": {name: z[f"init_{name}"] for name in INIT_NAMES},
    }
    if "recall_sub" in z.files:          # sub-N^2 cache
        out["recall_sub"] = z["recall_sub"]
        out["ref_idx"] = z["ref_idx"]
    else:                                # full N×N cache
        out["recall_matrix"] = z["recall_matrix"]
    return out


def _save_cache(npz_path: str, recall_matrix, n_tokens, init_scores):
    print(f"[emmia] Saving Phase 1 cache to {npz_path}")
    np.savez_compressed(
        npz_path,
        recall_matrix=recall_matrix,
        n_tokens=n_tokens,
        **{f"init_{name}": init_scores[name] for name in INIT_NAMES},
    )


def _save_cache_sub(npz_path: str, recall_sub, ref_idx, n_tokens, init_scores):
    print(f"[emmia] Saving Phase 1 (sub-N^2) cache to {npz_path}")
    np.savez_compressed(
        npz_path,
        recall_sub=recall_sub,
        ref_idx=ref_idx,
        n_tokens=n_tokens,
        **{f"init_{name}": init_scores[name] for name in INIT_NAMES},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════

def run_emmia(
    model_path: str,
    con_eval_path: str,
    uncon_eval_path: str,
    output_path: str,
    max_tokens: int = 512,
    prefix_batch_size: int = 4,
    n_iterations: int = 10,
    mink_k: int = 20,
    minkpp_k: int = 20,
    cache_dir: Optional[str] = None,
    ref_pool: Optional[int] = None,
    rows: Optional[str] = None,
    merge_rows: bool = False,
):
    # ── Load data ────────────────────────────────────────────────────────
    print(f"[emmia] Loading member eval: {con_eval_path}")
    con_texts = load_texts(con_eval_path)
    print(f"[emmia] Loading non-member eval: {uncon_eval_path}")
    uncon_texts = load_texts(uncon_eval_path)

    texts = con_texts + uncon_texts
    labels = np.array(
        [1] * len(con_texts) + [0] * len(uncon_texts), dtype=int
    )
    N = len(texts)
    print(f"[emmia] Combined D_test: {len(con_texts)} members + "
          f"{len(uncon_texts)} non-members = {N} total")

    # ── Decide full N×N vs sub-N^2 (O(N*K)) ──────────────────────────────
    use_sub = ref_pool is not None and ref_pool < N
    recall_matrix = None
    recall_sub = None
    ref_idx = None
    if use_sub:
        ref_idx = select_reference_idx(labels, ref_pool)
        print(f"[emmia] SUB-N^2 mode: K={len(ref_idx)} reference targets "
              f"(N*K = {N*len(ref_idx)} pairs vs N^2 = {N*N})")

    # ── Try cache first ──────────────────────────────────────────────────
    npz_path = _cache_path(cache_dir, output_path, ref_pool if use_sub else None)
    _base = os.path.splitext(os.path.basename(output_path))[0]
    if merge_rows:   # 2026-09-25: assemble row-sharded partial caches ({base}__rows{a}_{b}.npz) into the standard full cache
        assert cache_dir and not use_sub, "--merge-rows needs --cache-dir and full N×N mode"
        import glob as _glob
        parts = sorted(_glob.glob(os.path.join(cache_dir, f"{_base}__rows*_*.npz")))
        full = np.zeros((N, N), dtype=np.float64); covered = np.zeros(N, dtype=bool); init0 = None; ntok0 = None
        for pf in parts:
            z = np.load(pf, allow_pickle=False); a, b = int(z["row_start"]), int(z["row_end"])
            assert z["recall_rows"].shape == (b - a, N), (pf, z["recall_rows"].shape)
            full[a:b] = z["recall_rows"]; covered[a:b] = True
            if init0 is None: init0 = {name: z[f"init_{name}"] for name in INIT_NAMES}; ntok0 = z["n_tokens"]
        assert covered.all(), f"rows not covered: {np.where(~covered)[0][:10]} ... ({(~covered).sum()} missing) from {len(parts)} parts"
        print(f"[emmia] merged {len(parts)} row shards -> {npz_path}")
        _save_cache(npz_path, full, ntok0, init0)
    cached = _load_cache(npz_path) if npz_path else None

    if use_sub:
        cache_ok = (cached is not None and cached.get("recall_sub") is not None
                    and cached["recall_sub"].shape == (N, len(ref_idx)))
    else:
        cache_ok = (cached is not None and cached.get("recall_matrix") is not None
                    and cached["recall_matrix"].shape == (N, N))

    if cache_ok:
        init_scores = cached["init_scores"]
        n_tokens_arr = cached["n_tokens"]
        if use_sub:
            recall_sub = cached["recall_sub"]
            ref_idx = cached["ref_idx"]          # authoritative from cache
        else:
            recall_matrix = cached["recall_matrix"]
        print(f"[emmia] Cache hit — skipping Phase 1a/1b")
    else:
        if cached is not None:
            got = (cached.get("recall_sub") if use_sub
                   else cached.get("recall_matrix"))
            shp = None if got is None else got.shape
            print(f"[emmia] Cache miss/mismatch (got {shp}) — recomputing")

        # ── Load model (only needed when computing from scratch) ─────────
        print(f"[emmia] Loading model: {model_path}")
        model_bundle = load_model_bundle(model_path)
        model = model_bundle.model
        tokenizer = model_bundle.tokenizer
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model.eval()

        # ── Phase 1a ─────────────────────────────────────────────────────
        print(f"[emmia] Phase 1a: Unconditional forward passes ({N} samples)...")
        init_scores, n_tokens_arr = compute_unconditional_all_inits(
            model, tokenizer, texts, max_tokens,
            mink_k=mink_k, minkpp_k=minkpp_k,
        )

        # ── Row shard (2026-09-25): compute prefix rows r0..r1-1 only, save partial cache, exit ──
        if rows:
            assert cache_dir and not use_sub, "--rows needs --cache-dir and full N×N mode"
            r0, r1 = (int(x) for x in rows.split(":")); r1 = min(r1, N)
            print(f"[emmia] ROW SHARD {r0}:{r1} of {N} (ReCaLL rows only)")
            part = compute_recall_matrix(model, tokenizer, texts, init_scores["loss"], max_tokens, prefix_batch_size, row_range=(r0, r1))
            pf = os.path.join(cache_dir, f"{_base}__rows{r0}_{r1}.npz")
            np.savez_compressed(pf, recall_rows=part[r0:r1], row_start=r0, row_end=r1, n_tokens=n_tokens_arr,
                                **{f"init_{name}": init_scores[name] for name in INIT_NAMES})
            print(f"[emmia] ROWS_DONE {r0}:{r1} -> {pf}", flush=True)
            return

        # ── Phase 1b ─────────────────────────────────────────────────────
        if use_sub:
            print(f"[emmia] Phase 1b (sub): ReCaLL {N}×{len(ref_idx)} = "
                  f"{N*len(ref_idx)} pairs, batch_size={prefix_batch_size}...")
            recall_sub = compute_recall_matrix_sub(
                model, tokenizer, texts, init_scores["loss"], ref_idx,
                max_tokens, prefix_batch_size,
            )
            print(f"[emmia] ReCaLL sub computed. Shape: {recall_sub.shape}, "
                  f"range: [{recall_sub.min():.4f}, {recall_sub.max():.4f}]")
        else:
            print(f"[emmia] Phase 1b: ReCaLL matrix ({N}×{N} = "
                  f"{N*N} pairs, batch_size={prefix_batch_size})...")
            recall_matrix = compute_recall_matrix(
                model, tokenizer, texts, init_scores["loss"],
                max_tokens, prefix_batch_size,
            )
            print(f"[emmia] ReCaLL matrix computed. "
                  f"Shape: {recall_matrix.shape}, "
                  f"range: [{recall_matrix.min():.4f}, {recall_matrix.max():.4f}]")

        # Free GPU memory before Phase 2 (CPU-only)
        del model, model_bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if npz_path:
            if use_sub:
                _save_cache_sub(npz_path, recall_sub, ref_idx, n_tokens_arr, init_scores)
            else:
                _save_cache(npz_path, recall_matrix, n_tokens_arr, init_scores)

    # ── Init AUCs ────────────────────────────────────────────────────────
    init_aucs = {name: float(calc_auc(scores, labels))
                 for name, scores in init_scores.items()}
    print("\n[emmia] Init AUCs (Phase 1a):")
    for name in INIT_NAMES:
        print(f"    {name:>8} : {init_aucs[name]:.4f}")

    # ── Phase 2: run EM once per init on the shared recall_matrix ────────
    print(f"\n[emmia] Phase 2: EM iterations ({n_iterations} per init, "
          f"{len(INIT_NAMES)} inits)...")
    em_results = {}
    final_scores = {}
    for name in INIT_NAMES:
        print(f"\n  [emmia] EM with init = {name}")
        if use_sub:
            fs, traj = em_iterations_sub(
                recall_sub, ref_idx, init_scores[name], labels,
                n_iterations, verbose=True,
            )
        else:
            fs, traj = em_iterations(
                recall_matrix, init_scores[name], labels,
                n_iterations, verbose=True,
            )
        final_scores[name] = fs
        em_results[name] = {
            "init_auc": init_aucs[name],
            "final_auc": float(calc_auc(fs, labels)),
            "auc_trajectory": traj,
        }

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print(f"[emmia] Summary — {os.path.basename(model_path)} × "
          f"{os.path.basename(con_eval_path)}")
    print(f"  N = {N} ({len(con_texts)} members / {len(uncon_texts)} non-members)")
    print(f"  {'init':<10} {'init AUC':>10} {'final AUC':>10} {'Δ':>9} "
          f"{'|AUC-0.5|':>11} {'sign-flip':>10}")
    print("  " + "─" * 66)
    for name in INIT_NAMES:
        r = em_results[name]
        delta = r["final_auc"] - r["init_auc"]
        absdist = abs(r["final_auc"] - 0.5)
        flipped = "YES" if (r["init_auc"] - 0.5) * (r["final_auc"] - 0.5) < 0 else ""
        print(f"  {name:<10} {r['init_auc']:>10.4f} {r['final_auc']:>10.4f} "
              f"{delta:>+9.4f} {absdist:>11.4f} {flipped:>10}")
    print("═" * 72)

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "method": "emmia_multi_init",
        "model_path": model_path,
        "con_eval_path": con_eval_path,
        "uncon_eval_path": uncon_eval_path,
        "n_con": len(con_texts),
        "n_uncon": len(uncon_texts),
        "n_total": N,
        "max_tokens": max_tokens,
        "n_iterations": n_iterations,
        "mink_k": mink_k,
        "minkpp_k": minkpp_k,
        "inits": INIT_NAMES,
        "phase1b_mode": "sub" if use_sub else "full",
        "ref_pool": int(len(ref_idx)) if use_sub else None,
        "ref_idx": [int(x) for x in ref_idx] if use_sub else None,
        "em_results": em_results,
        "samples": [
            {
                "index": i,
                "label": int(labels[i]),
                "source": "member" if labels[i] == 1 else "non-member",
                "n_tokens": int(n_tokens_arr[i]),
                "init_scores": {
                    name: float(init_scores[name][i]) for name in INIT_NAMES
                },
                "final_scores": {
                    name: float(final_scores[name][i]) for name in INIT_NAMES
                },
            }
            for i in range(N)
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[emmia] Results saved to {output_path}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EM-MIA (multi-init): runs EM iterations from Loss, Zlib, "
                    "Min-K, and Min-K++ initialisations on a shared ReCaLL matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--con-eval-path", type=str, required=True)
    parser.add_argument("--uncon-eval-path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prefix-batch-size", type=int, default=4)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--mink-k", type=int, default=20,
                        help="K%% for Min-K init (paper default: 20)")
    parser.add_argument("--minkpp-k", type=int, default=20,
                        help="K%% for Min-K++ init (paper default: 20)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="If set, save/load Phase 1a+1b to/from a .npz "
                             "in this dir (filename derived from --output). "
                             "Skips all GPU work on re-runs with same args.")
    parser.add_argument("--ref-pool", type=int, default=None,
                        help="Sub-N^2 mode: evaluate each prefix's EM AUC over a "
                             "balanced pool of K reference targets instead of all N. "
                             "Cost O(N*K) instead of O(N^2). Omit (or set >= N) for "
                             "the exact full N×N EM. Recommended K=384 for large cells.")
    parser.add_argument("--rows", type=str, default=None, help="row shard 'a:b' (prefix rows) -> partial cache only, no Phase 2 (2026-09-25)")
    parser.add_argument("--merge-rows", action="store_true", help="assemble {base}__rows*.npz shards from --cache-dir into the full cache, then run Phase 2")
    args = parser.parse_args()

    run_emmia(
        model_path=args.model,
        con_eval_path=args.con_eval_path,
        uncon_eval_path=args.uncon_eval_path,
        output_path=args.output,
        max_tokens=args.max_tokens,
        prefix_batch_size=args.prefix_batch_size,
        n_iterations=args.n_iterations,
        mink_k=args.mink_k,
        minkpp_k=args.minkpp_k,
        cache_dir=args.cache_dir,
        ref_pool=args.ref_pool,
        rows=args.rows,
        merge_rows=args.merge_rows,
    )
