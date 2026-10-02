"""
Fine-tuned Score Deviation (FSD)
Paper: "Fine-tuning can help detect pretraining data from large language models"
       (Zhang et al., ICLR 2025)  https://github.com/ml-stat-Sustech/Fine-tuned-Score-Deviation

Idea: fine-tune the target model (LoRA, self-supervised next-token loss) on a few
*certified non-members from the same domain*, then score each candidate with BOTH
the base model f_theta and the fine-tuned model f_theta'.  The FSD statistic is the
per-example score deviation  S(x; f_theta) - S(x; f_theta').  Fine-tuning drives the
likelihood of non-members up sharply while members (already fit) barely move, so the
deviation separates the two classes.

Integration with OLMo-Detect
----------------------------
* Certified non-members for fine-tuning = the non-member DEV split of the same
  domain (disjoint from the test split), supplied via `uncon_dev_path` (the same
  mechanism RECALL/CAMIA use).  No test data is ever used for fine-tuning.
* We wrap the four scoring functions the paper uses plus Min-K%++:
      loss (mean log-likelihood), zlib, lowercase, Min-K%, Min-K%++.
  Every base score in this repo follows the "higher = member" convention.  Fine-tuning
  raises the score of non-members (they become more likely) while leaving members ~flat,
  so we orient the deviation as
      FSD(x) = S(x; f_theta) - S(x; f_theta')   ("higher = member")
  Members ~ 0 (small change); non-members become strongly negative -> members rank higher,
  matching evaluate.py's ROC-AUC convention.
* The LoRA adapter is trained once per (model, dev file, hyper-params) and cached to
  disk, so the member and non-member test runs of a domain share one adapter.

Output columns (per sample): fsd_loss_score, fsd_zlib_score, fsd_lowercase_score,
fsd_mink_score, fsd_minkpp_score (plus base_*/ft_* for diagnostics).
"""

from __future__ import annotations

import hashlib
import math
import os
import zlib
from pathlib import Path
from typing import Any

import torch

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text, load_dataset
from model_utils import HFGenerationAdapter, ModelBundle
from progress_utils import progress
from methods.minkprob_method import get_optimal_k, mink_prob_score, mink_pp_score

CACHE_ROOT = Path(__file__).resolve().parent.parent / "fsd_adapter_cache"


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _score_texts(
    adapter: HFGenerationAdapter,
    texts: list[str],
    k_mink: float,
    k_minkpp: float,
    batch_size: int,
    desc: str,
) -> list[dict[str, float]]:
    """Score a list of plain texts, returning per-text {loss, zlib, lowercase, mink, minkpp}.

    lowercase needs the log-likelihood of the lowercased text, so each text is scored
    together with its .lower() variant (interleaved) in one batch.
    """
    out: list[dict[str, float]] = []
    for start in progress(
        range(0, len(texts), batch_size),
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        chunk = texts[start : start + batch_size]
        combined: list[str] = []
        for t in chunk:
            combined.append(t)
            combined.append(t.lower())

        full_logits, target_ids, _ = adapter.score_plain_batch(
            texts=combined,
            batch_size=len(combined),
        )

        for i, raw_text in enumerate(chunk):
            o_logits = full_logits[i * 2]
            o_ids = target_ids[i * 2]
            l_logits = full_logits[i * 2 + 1]
            l_ids = target_ids[i * 2 + 1]

            if o_ids.numel() == 0 or l_ids.numel() == 0:
                out.append({k: float("nan") for k in ("loss", "zlib", "lowercase", "mink", "minkpp")})
                continue

            o_lp = torch.log_softmax(o_logits, dim=-1).gather(-1, o_ids.unsqueeze(-1)).squeeze(-1)
            ll_sum = float(o_lp.sum().item())
            mean_ll = float(o_lp.mean().item())

            l_lp = torch.log_softmax(l_logits, dim=-1).gather(-1, l_ids.unsqueeze(-1)).squeeze(-1)
            mean_ll_low = float(l_lp.mean().item())

            z_bytes = len(zlib.compress(raw_text.encode("utf-8")))
            out.append({
                "loss": mean_ll,
                "zlib": ll_sum / z_bytes if z_bytes > 0 else float("nan"),
                "lowercase": mean_ll - mean_ll_low,
                "mink": float(mink_prob_score(o_logits, o_ids, k_percent=k_mink)),
                "minkpp": float(mink_pp_score(o_logits, o_ids, k_percent=k_minkpp)),
            })
    return out


class FSDMethod(BaseMethod):
    name = "fsd"

    def __init__(
        self,
        uncon_dev_path: str | None = None,
        batch_size: int = 1,
        max_tokens: int = 1024,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        epochs: int = 3,
        learning_rate: float = 1e-3,
        train_batch_size: int = 8,
        max_train_len: int = 1024,
        seed: int = 42,
        **kwargs,
    ):
        self.uncon_dev_path = uncon_dev_path
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.max_train_len = max_train_len
        self.seed = seed

    # ------------------------------------------------------------------ fine-tune

    # ------------------------------------------------------------------ cache key (content-aware)
    # 2026-09-22: the key used to hash only the dev-file *name*; a regenerated dev split with the
    # same stem silently reused a stale adapter.  Now the key also hashes the dev-file *contents*.
    # A legacy (name-keyed) cache is migrated only if it is newer than every dev file it was
    # trained on; otherwise it is ignored and the adapter is retrained.
    def _dev_paths_for_key(self) -> list:
        return [p for p in (self.uncon_dev_path,) if p]

    def _dev_content_tag(self) -> str:
        hh = hashlib.sha1()
        for p in self._dev_paths_for_key():
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        hh.update(chunk)
            else:
                hh.update(b"missing")
        return hh.hexdigest()[:8]

    def _content_keyed_dir(self, legacy: Path) -> Path:
        new = legacy.parent / f"{legacy.name}__d{self._dev_content_tag()}"
        if (new / "adapter_config.json").exists() or not (legacy / "adapter_config.json").exists():
            return new
        try:
            dev_m = max(os.path.getmtime(p) for p in self._dev_paths_for_key() if os.path.exists(p))
            leg_m = min(os.path.getmtime(f) for f in legacy.iterdir())
        except ValueError:
            return new
        if leg_m >= dev_m:
            import shutil
            shutil.copytree(legacy, new, dirs_exist_ok=True)
            print(f"[fsd] migrated legacy adapter cache (newer than dev split) -> {new}")
        else:
            print(f"[fsd] legacy adapter cache {legacy.name} predates the dev split -> ignored, retraining")
        return new

    def _adapter_cache_dir(self, model_bundle: ModelBundle) -> Path:
        stem = Path(self.uncon_dev_path).stem if self.uncon_dev_path else "none"
        key = (
            f"{model_bundle.model_name}|{stem}|r{self.lora_rank}|a{self.lora_alpha}"
            f"|e{self.epochs}|lr{self.learning_rate}|tb{self.train_batch_size}"
            f"|ml{self.max_train_len}|s{self.seed}"
        )
        h = hashlib.sha1(key.encode()).hexdigest()[:10]
        return self._content_keyed_dir(CACHE_ROOT / f"{model_bundle.model_name}__{stem}__{h}")

    def _train_texts(self) -> list[str]:
        if not self.uncon_dev_path or not os.path.exists(self.uncon_dev_path):
            return []
        dev = load_dataset(self.uncon_dev_path)
        texts: list[str] = []
        for rec in dev.records:
            t = extract_text(rec)
            if t:
                texts.append(t)
        return texts

    def _get_finetuned_model(self, model_bundle: ModelBundle, train_texts: list[str]):
        """Return a PeftModel fine-tuned on `train_texts` (loading from cache if present)."""
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        cache_dir = self._adapter_cache_dir(model_bundle)
        tokenizer = model_bundle.tokenizer

        if (cache_dir / "adapter_config.json").exists():
            print(f"[fsd] loading cached LoRA adapter from {cache_dir}")
            peft_model = PeftModel.from_pretrained(model_bundle.model, str(cache_dir))
            peft_model.eval()
            return peft_model

        torch.manual_seed(self.seed)
        config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        )
        peft_model = get_peft_model(model_bundle.model, config)
        peft_model.train()
        # OOM control: gradient checkpointing trades compute for memory by
        # recomputing activations in the backward pass -> NUMERICALLY IDENTICAL
        # results (same loss/gradients; RNG state preserved for LoRA dropout),
        # just a large drop in peak activation memory. enable_input_require_grads
        # is required so gradients flow through the frozen 4-bit/fp16 base to the
        # LoRA adapters when checkpointing is on.
        try:
            peft_model.gradient_checkpointing_enable()
            peft_model.enable_input_require_grads()
            if getattr(peft_model, "config", None) is not None:
                peft_model.config.use_cache = False
            print("[fsd] gradient checkpointing enabled (OOM control)")
        except Exception as e:  # pragma: no cover - defensive
            print(f"[fsd] gradient checkpointing unavailable ({e}); continuing without")
        device = next(peft_model.parameters()).device

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize dev texts (plain, self-supervised next-token objective).
        encodings = [
            tokenizer(t, add_special_tokens=True, truncation=True, max_length=self.max_train_len).input_ids
            for t in train_texts
        ]
        encodings = [e for e in encodings if len(e) >= 2]
        if not encodings:
            print("[fsd] WARNING: no usable dev texts for fine-tuning; adapter left untrained.")
            peft_model.eval()
            return peft_model

        bs = max(1, min(self.train_batch_size, len(encodings)))
        num_batches = math.ceil(len(encodings) / bs)
        total_steps = max(1, self.epochs * num_batches)

        trainable = [p for p in peft_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate)
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)

        pad_id = tokenizer.pad_token_id
        step = 0
        for epoch in range(self.epochs):
            g = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(len(encodings), generator=g).tolist()
            for bstart in range(0, len(encodings), bs):
                batch = [encodings[j] for j in order[bstart : bstart + bs]]
                maxlen = max(len(x) for x in batch)
                input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
                attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
                labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
                for r, ids in enumerate(batch):
                    input_ids[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attn[r, : len(ids)] = 1
                    labels[r, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                labels = labels.to(device)

                out = peft_model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
            print(f"[fsd] fine-tune epoch {epoch + 1}/{self.epochs} last-batch loss={float(loss.item()):.4f}")

        peft_model.eval()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(str(cache_dir))
            print(f"[fsd] saved LoRA adapter to {cache_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"[fsd] WARNING: could not cache adapter ({exc}).")
        return peft_model

    # ------------------------------------------------------------------ run
    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        k_mink, k_minkpp = get_optimal_k(model_bundle.model_path, dataset.path)

        prepared: list[tuple[int, str, str]] = []
        for idx, record in enumerate(dataset.records):
            text = extract_text(record)
            if text:
                prepared.append((idx, extract_sample_id(record, fallback=idx), text))
        texts = [p[2] for p in prepared]

        # 1) Base scores (must run BEFORE LoRA is injected into the shared model).
        base_scores = _score_texts(
            model_bundle.llm_adapter, texts, k_mink, k_minkpp, self.batch_size,
            desc=f"fsd base [{dataset.name}]",
        )

        # 2) Fine-tune LoRA on certified non-members (non-member dev of this domain).
        train_texts = self._train_texts()
        peft_model = self._get_finetuned_model(model_bundle, train_texts)
        ft_adapter = HFGenerationAdapter(model=peft_model, tokenizer=model_bundle.tokenizer, seed=self.seed)

        # 3) Fine-tuned scores.
        ft_scores = _score_texts(
            ft_adapter, texts, k_mink, k_minkpp, self.batch_size,
            desc=f"fsd ft [{dataset.name}]",
        )

        samples: list[dict[str, Any]] = []
        agg: dict[str, list[float]] = {f"fsd_{k}": [] for k in ("loss", "zlib", "lowercase", "mink", "minkpp")}
        for (record_index, sample_id, text), b, f in zip(prepared, base_scores, ft_scores):
            row: dict[str, Any] = {"record_index": record_index, "sample_id": sample_id, "text": text}
            for key in ("loss", "zlib", "lowercase", "mink", "minkpp"):
                bv, fv = b[key], f[key]
                dev = (bv - fv) if (math.isfinite(bv) and math.isfinite(fv)) else float("nan")
                row[f"base_{key}"] = bv
                row[f"ft_{key}"] = fv
                row[f"fsd_{key}_score"] = dev
                agg[f"fsd_{key}"].append(dev)
            samples.append(row)

        # Free the LoRA graph before the process moves on.
        del peft_model, ft_adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "method": self.name,
            "dataset": dataset.name,
            "dataset_data_type": dataset.data_type,
            "uncon_dev_path": self.uncon_dev_path,
            "num_train_nonmembers": len(train_texts),
            "k_mink_used": k_mink,
            "k_minkpp_used": k_minkpp,
            "lora": {
                "rank": self.lora_rank, "alpha": self.lora_alpha, "epochs": self.epochs,
                "learning_rate": self.learning_rate, "train_batch_size": self.train_batch_size,
            },
            "num_total_records": len(dataset.records),
            "num_scored_records": len(samples),
            "summary": {k: _safe_mean(v) for k, v in agg.items()},
            "samples": samples,
        }


def build_method(
    uncon_dev_path: str | None = None,
    batch_size: int = 1,
    max_tokens: int = 1024,
    **kwargs,
) -> BaseMethod:
    return FSDMethod(
        uncon_dev_path=uncon_dev_path,
        batch_size=batch_size,
        max_tokens=max_tokens,
        **{k: v for k, v in kwargs.items() if k in {
            "lora_rank", "lora_alpha", "lora_dropout", "epochs", "learning_rate",
            "train_batch_size", "max_train_len", "seed",
        }},
    )
