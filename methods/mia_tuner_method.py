"""
MIA-Tuner (aligned / instruction-tuned LLM pipeline)
Paper : "MIA-Tuner: Adapting Large Language Models as Pre-training Text Detector"
        (Fu et al., AAAI 2025, Oral)
Github: https://github.com/tsinghua-fib-lab/AAAI2025_MIA-Tuner  (mia_hybrid.py)

Faithful re-implementation of the *aligned-model* pipeline of MIA-Tuner for the
OLMo-Detect framework.  OLMo2-*-Instruct are aligned (chat) models, so we use the
instruction / soft-prompt pipeline exactly as in the official `mia_hybrid.py`
(the `not args.unaligned_model` branch).

Method (verbatim from the official code)
----------------------------------------
MIA-Tuner is a *supervised* detector.  It learns a soft prompt (prompt-tuning,
50 virtual tokens, initialised from the text
    "Here is the prefix of a training set text, please verbatim generate the
     subsequent tokens:")
that turns the target LLM into a membership classifier.  Each candidate text is
wrapped in the instruction
    "Please tell me whether the given example is used in the training dataset:
     \n{text}"
and the model is trained so that members answer "Yes" and non-members answer
"No" at the assistant position.  The training objective is the sum of three
terms (all weight 1, as in the official main experiment):
    * llm_loss : language-model loss on the formatted chat (prompt tokens
                 down-weighted by `prompt_loss_weight`=0.01, answer weight 1)
    * clf_loss : cross-entropy of the 2-way (No, Yes) logits (built from the
                 per-token CE against the "No"/"Yes" ids at the answer position)
                 vs. the true membership label
    * err_loss : -(log(1-P(No|member)) + log(1-P(Yes|non-member)))
The per-sample membership score is
    score(x) = softmax(-[loss_Yes(x), loss_No(x)])[Yes] = P(Yes among {Yes,No})
which follows OLMo-Detect's "higher = member" convention.

Integration with OLMo-Detect
----------------------------
* Supervised labels are the certified dev splits of the same domain, exactly as
  CAMIA consumes them:
      members     = member  dev  (`con_dev_path`)
      non-members = non-member dev (`uncon_dev_path`)
  No test data is ever used for training.
* The soft prompt is trained once per (model, con-dev, uncon-dev, hyper-params)
  and cached to disk, so the member- and non-member-test runs of a
  domain share one adapter (same mechanism as fsd_method.py).

Output column (per sample): mia_tuner_score  (= P(Yes)).
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from base_method import BaseMethod
from data_utils import DatasetBundle, extract_sample_id, extract_text, load_dataset
from model_utils import ModelBundle
from progress_utils import progress

CACHE_ROOT = Path(__file__).resolve().parent.parent / "mia_tuner_adapter_cache"

# Verbatim from the official data_utils.py.
INSTRUCTION_TEMPLATE = (
    "Please tell me whether the given example is used in the training dataset: \n{text}"
)
INIT_PROMPT_TEXT = (
    "Here is the prefix of a training set text, please verbatim generate the subsequent tokens:"
)


def _safe_mean(values: list[float]) -> float:
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _yes_no_token_ids(tokenizer) -> tuple[int, int]:
    """Token id emitted at the assistant position for a "Yes"/"No" answer.

    Matches specail_token_id() in the official data_utils.py: the first token of
    the assistant turn, obtained by diffing the chat-templated ids with and
    without the assistant answer.
    """
    blank = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Test text"}], add_generation_prompt=True
    )
    yes = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Test text"}, {"role": "assistant", "content": "Yes"}]
    )
    no = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Test text"}, {"role": "assistant", "content": "No"}]
    )
    return int(yes[len(blank)]), int(no[len(blank)])


class MIATunerMethod(BaseMethod):
    name = "mia_tuner"

    def __init__(
        self,
        uncon_dev_path: str | None = None,
        con_dev_path: str | None = None,
        batch_size: int = 8,
        max_tokens: int = 1024,
        # ---- official aligned-pipeline hyper-parameters (main_exp.sh) ----
        p_tokens: int = 50,
        epochs: int = 50,
        learning_rate: float = 5e-4,
        train_batch_size: int = 8,
        train_micro_batch_size: int | None = None,
        prompt_loss_weight: float = 0.01,
        max_text_tokens: int = 1024,
        seed: int = 42,
        **kwargs,
    ):
        self.uncon_dev_path = uncon_dev_path
        self.con_dev_path = con_dev_path
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.p_tokens = p_tokens
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        # OOM control: split each optimizer step into gradient-accumulated
        # micro-batches.  The *effective* batch stays = train_batch_size (the
        # official value), so training is numerically unchanged; only the peak
        # activation / logits-fp32 memory of a single forward drops.  Kept OUT
        # of the adapter cache key on purpose (execution detail, not a hparam).
        self.train_micro_batch_size = train_micro_batch_size
        self.prompt_loss_weight = prompt_loss_weight
        self.max_text_tokens = max_text_tokens
        # instruction + text + answer overhead; keep the answer un-truncated.
        self.max_length = max_text_tokens + 128
        self.seed = seed

    # ------------------------------------------------------------------ caching

    # ------------------------------------------------------------------ cache key (content-aware)
    # 2026-09-22: the key used to hash only the dev-file *name*; a regenerated dev split with the
    # same stem silently reused a stale adapter.  Now the key also hashes the dev-file *contents*.
    # A legacy (name-keyed) cache is migrated only if it is newer than every dev file it was
    # trained on; otherwise it is ignored and the adapter is retrained.
    def _dev_paths_for_key(self) -> list:
        return [p for p in (self.con_dev_path, self.uncon_dev_path) if p]

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
            print(f"[mia_tuner] migrated legacy adapter cache (newer than dev split) -> {new}")
        else:
            print(f"[mia_tuner] legacy adapter cache {legacy.name} predates the dev split -> ignored, retraining")
        return new

    def _adapter_cache_dir(self, model_bundle: ModelBundle) -> Path:
        con_stem = Path(self.con_dev_path).stem if self.con_dev_path else "none"
        unc_stem = Path(self.uncon_dev_path).stem if self.uncon_dev_path else "none"
        key = (
            f"{model_bundle.model_name}|{con_stem}|{unc_stem}|p{self.p_tokens}"
            f"|e{self.epochs}|lr{self.learning_rate}|tb{self.train_batch_size}"
            f"|plw{self.prompt_loss_weight}|ml{self.max_length}|s{self.seed}"
        )
        h = hashlib.sha1(key.encode()).hexdigest()[:10]
        return self._content_keyed_dir(CACHE_ROOT / f"{model_bundle.model_name}__{con_stem}__{h}")

    # ------------------------------------------------------------------ encoding
    def _load_texts(self, path: str | None) -> list[str]:
        if not path or not os.path.exists(path):
            return []
        ds = load_dataset(path)
        texts: list[str] = []
        for rec in ds.records:
            t = extract_text(rec)
            if t:
                texts.append(t)
        return texts

    def _truncate_text(self, tokenizer, text: str) -> str:
        ids = tokenizer.encode(text, add_special_tokens=False)[: self.max_text_tokens]
        return tokenizer.decode(ids)

    def _encode_train(self, tokenizer, text: str, judge: str) -> dict[str, Any] | None:
        """Return {input_ids, out_idx} for one labelled training example.

        `out_idx` is the position (in input_ids space) of the answer token, i.e.
        the position whose predicted distribution is compared with Yes/No.
        """
        instr = INSTRUCTION_TEMPLATE.format(text=self._truncate_text(tokenizer, text))
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}], add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}, {"role": "assistant", "content": judge}],
            add_generation_prompt=False,
        )
        out_idx = len(prompt_ids)  # index of the Yes/No token in full_ids
        if out_idx >= len(full_ids) or len(full_ids) > self.max_length:
            # Should not happen (text is pre-truncated), but guard against it.
            full_ids = full_ids[: self.max_length]
            if out_idx >= len(full_ids):
                return None
        return {"input_ids": full_ids, "out_idx": out_idx}

    def _encode_eval(self, tokenizer, text: str) -> dict[str, Any]:
        instr = INSTRUCTION_TEMPLATE.format(text=self._truncate_text(tokenizer, text))
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}], add_generation_prompt=True
        )
        return {"input_ids": prompt_ids, "out_idx": len(prompt_ids)}

    # ------------------------------------------------------------------ training
    def _train_soft_prompt(self, model_bundle: ModelBundle, cache_dir: Path):
        from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model

        tokenizer = model_bundle.tokenizer
        yes_id, no_id = _yes_no_token_ids(tokenizer)

        mem_texts = self._load_texts(self.con_dev_path)
        non_texts = self._load_texts(self.uncon_dev_path)
        n = min(len(mem_texts), len(non_texts))
        if n == 0:
            raise ValueError(
                "MIA-Tuner needs both member (member) and non-member "
                "(non-member) dev texts; got "
                f"{len(mem_texts)} members / {len(non_texts)} non-members."
            )
        # Balance the two classes (official uses matched member/non-member counts).
        mem_texts = mem_texts[:n]
        non_texts = non_texts[:n]

        mem_ex = [e for t in mem_texts if (e := self._encode_train(tokenizer, t, "Yes"))]
        non_ex = [e for t in non_texts if (e := self._encode_train(tokenizer, t, "No"))]
        n = min(len(mem_ex), len(non_ex))
        mem_ex, non_ex = mem_ex[:n], non_ex[:n]

        torch.manual_seed(self.seed)
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=self.p_tokens,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=INIT_PROMPT_TEXT,
            tokenizer_name_or_path=model_bundle.model_path,
        )
        model = get_peft_model(model_bundle.model, peft_config)
        model.train()
        device = next(model.parameters()).device
        pad_id = tokenizer.pad_token_id
        p = self.p_tokens

        trainable = [q for q in model.parameters() if q.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=0.0)
        from transformers import get_scheduler

        bs = max(1, min(self.train_batch_size, n))
        # Micro-batch for gradient accumulation.  Precedence: explicit kwarg >
        # MIA_TUNER_MICRO_BATCH env var > full logical batch (original behaviour).
        env_micro = os.environ.get("MIA_TUNER_MICRO_BATCH")
        micro = self.train_micro_batch_size
        if micro is None and env_micro:
            micro = int(env_micro)
        micro = max(1, min(micro or bs, bs))
        if micro < bs:
            print(f"[mia_tuner] grad-accum: micro-batch {micro} x "
                  f"{math.ceil(bs / micro)} -> effective batch {bs}")
        steps_per_epoch = math.ceil(n / bs)
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0,
            num_training_steps=steps_per_epoch * self.epochs,
        )
        ce_none = torch.nn.CrossEntropyLoss(reduction="none")

        def collate(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            maxlen = max(len(b["input_ids"]) for b in batch)
            ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            am = torch.zeros((len(batch), maxlen), dtype=torch.long)
            out_idx = torch.zeros(len(batch), dtype=torch.long)
            for r, b in enumerate(batch):
                L = len(b["input_ids"])
                ids[r, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
                am[r, :L] = 1
                out_idx[r] = b["out_idx"]
            return ids.to(device), am.to(device), out_idx.to(device)

        def side_losses(logits, ids, am, out_idx, ans_id_pos, ans_id_neg):
            # logits: [B, p+T, V].  Align so that sl[i] predicts ids[i].
            sl = logits[:, p - 1 : -1, :].contiguous()           # [B, T, V]
            B, T, V = sl.shape
            per_tok = ce_none(sl.reshape(-1, V).float(), ids.reshape(-1)).reshape(B, T)
            # LM loss: answer region (>= out_idx) weight 1, prompt weight plw.
            col = torch.arange(T, device=device).expand(B, T)
            w = torch.full((B, T), self.prompt_loss_weight, device=device)
            w[col > (out_idx - 1).unsqueeze(1)] = 1.0
            denom = am.sum(1).clamp(min=1)
            lm = (per_tok * w * am).sum(1) / denom
            ans = sl[torch.arange(B, device=device), out_idx, :]  # [B, V]
            loss_pos = ce_none(ans.float(), torch.full((B,), ans_id_pos, device=device))
            loss_neg = ce_none(ans.float(), torch.full((B,), ans_id_neg, device=device))
            return lm, loss_pos, loss_neg

        order = list(range(n))
        for epoch in range(self.epochs):
            g = torch.Generator().manual_seed(self.seed + epoch)
            perm = torch.randperm(n, generator=g).tolist()
            last = 0.0
            for start in range(0, n, bs):
                idx_full = perm[start : start + bs]
                total = len(idx_full)  # logical batch (== bs except a short tail)
                optimizer.zero_grad()
                step_loss = 0.0
                # Gradient accumulation: sum micro-batch losses weighted by their
                # sample share so the accumulated gradient equals the full-batch
                # mean-loss gradient (identical to a single bs-sized step).
                for ms in range(0, total, micro):
                    idx = idx_full[ms : ms + micro]
                    frac = len(idx) / total
                    pos_ids, pos_am, pos_out = collate([mem_ex[j] for j in idx])
                    neg_ids, neg_am, neg_out = collate([non_ex[j] for j in idx])
                    pos_logits = model(input_ids=pos_ids, attention_mask=pos_am).logits
                    neg_logits = model(input_ids=neg_ids, attention_mask=neg_am).logits

                    # For members: pos_loss_y = CE(.,Yes), pos_loss_n = CE(.,No).
                    pos_lm, pos_loss_y, pos_loss_n = side_losses(
                        pos_logits, pos_ids, pos_am, pos_out, yes_id, no_id)
                    neg_lm, neg_loss_y, neg_loss_n = side_losses(
                        neg_logits, neg_ids, neg_am, neg_out, yes_id, no_id)

                    B = pos_ids.shape[0]
                    llm_loss = (pos_lm.mean() + neg_lm.mean()) / 2
                    clf_logits = -torch.cat([
                        torch.stack([pos_loss_n, pos_loss_y], dim=1),
                        torch.stack([neg_loss_n, neg_loss_y], dim=1),
                    ])
                    clf_labels = torch.cat([
                        torch.ones(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)
                    ]).to(device)
                    clf_loss = F.cross_entropy(clf_logits, clf_labels)
                    err_loss = -(
                        torch.log1p(-torch.exp(-pos_loss_n).clamp(max=1 - 1e-6))
                        + torch.log1p(-torch.exp(-neg_loss_y).clamp(max=1 - 1e-6))
                    ).mean()
                    loss = (llm_loss + clf_loss + err_loss) * frac

                    loss.backward()
                    step_loss += float(loss.item())
                    # Free the two full-vocab logits tensors before the next micro-batch.
                    del pos_logits, neg_logits
                optimizer.step()
                scheduler.step()
                last = step_loss
            print(f"[mia_tuner] epoch {epoch + 1}/{self.epochs} loss={last:.4f}")

        model.eval()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(cache_dir))
            print(f"[mia_tuner] saved soft prompt to {cache_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"[mia_tuner] WARNING: could not cache adapter ({exc}).")
        return model, len(mem_ex), len(non_ex)

    def _get_model(self, model_bundle: ModelBundle):
        cache_dir = self._adapter_cache_dir(model_bundle)
        if (cache_dir / "adapter_config.json").exists():
            from peft import PeftModel

            print(f"[mia_tuner] loading cached soft prompt from {cache_dir}")
            model = PeftModel.from_pretrained(model_bundle.model, str(cache_dir))
            model.eval()
            return model, -1, -1
        return self._train_soft_prompt(model_bundle, cache_dir)

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def _score(self, model, tokenizer, texts: list[str], device) -> list[float]:
        yes_id, no_id = _yes_no_token_ids(tokenizer)
        pad_id = tokenizer.pad_token_id
        p = self.p_tokens
        scores: list[float] = []
        bs = max(1, self.batch_size)
        enc = [self._encode_eval(tokenizer, t) for t in texts]
        for start in progress(
            range(0, len(enc), bs), desc="mia_tuner score", unit="batch", leave=False
        ):
            chunk = enc[start : start + bs]
            maxlen = max(len(c["input_ids"]) for c in chunk)
            ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
            am = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            out_idx = torch.zeros(len(chunk), dtype=torch.long)
            for r, c in enumerate(chunk):
                L = len(c["input_ids"])
                ids[r, :L] = torch.tensor(c["input_ids"], dtype=torch.long)
                am[r, :L] = 1
                out_idx[r] = c["out_idx"]
            ids, am, out_idx = ids.to(device), am.to(device), out_idx.to(device)
            logits = model(input_ids=ids, attention_mask=am).logits
            # eval alignment (official): shift = logits[:, p-1:, :]; predict token
            # after the prompt at position out_idx.
            shift = logits[:, p - 1 :, :]
            B = ids.shape[0]
            ans = shift[torch.arange(B, device=device), out_idx, :].float()  # [B, V]
            loss_y = F.cross_entropy(ans, torch.full((B,), yes_id, device=device), reduction="none")
            loss_n = F.cross_entropy(ans, torch.full((B,), no_id, device=device), reduction="none")
            # score = softmax(-[loss_y, loss_n])[Yes] = P(Yes)
            prob_yes = torch.softmax(-torch.stack([loss_y, loss_n], dim=1), dim=1)[:, 0]
            scores.extend(prob_yes.detach().cpu().tolist())
        return scores

    # ------------------------------------------------------------------ run
    def run(self, model_bundle: ModelBundle, dataset: DatasetBundle) -> dict[str, Any]:
        prepared: list[tuple[int, str, str]] = []
        for idx, record in enumerate(dataset.records):
            text = extract_text(record)
            if text:
                prepared.append((idx, extract_sample_id(record, fallback=idx), text))
        texts = [p[2] for p in prepared]

        model, n_mem, n_non = self._get_model(model_bundle)
        device = next(model.parameters()).device
        scores = self._score(model, model_bundle.tokenizer, texts, device)

        samples: list[dict[str, Any]] = []
        for (record_index, sample_id, text), s in zip(prepared, scores):
            samples.append({
                "record_index": record_index,
                "sample_id": sample_id,
                "text": text,
                "mia_tuner_score": s,
            })

        # Clean up the PEFT wrapper so the shared base model is untouched.
        try:
            model_bundle.model = model.get_base_model()
        except Exception:  # noqa: BLE001
            pass
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "method": self.name,
            "dataset": dataset.name,
            "dataset_data_type": dataset.data_type,
            "con_dev_path": self.con_dev_path,
            "uncon_dev_path": self.uncon_dev_path,
            "num_train_members": n_mem,
            "num_train_nonmembers": n_non,
            "hparams": {
                "p_tokens": self.p_tokens, "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "train_batch_size": self.train_batch_size,
                "prompt_loss_weight": self.prompt_loss_weight,
                "max_text_tokens": self.max_text_tokens,
            },
            "num_total_records": len(dataset.records),
            "num_scored_records": len(samples),
            "summary": {"mean_mia_tuner_score": _safe_mean(scores)},
            "samples": samples,
        }


def build_method(
    uncon_dev_path: str | None = None,
    con_dev_path: str | None = None,
    batch_size: int = 8,
    max_tokens: int = 1024,
    **kwargs,
) -> BaseMethod:
    return MIATunerMethod(
        uncon_dev_path=uncon_dev_path,
        con_dev_path=con_dev_path,
        batch_size=batch_size,
        max_tokens=max_tokens,
        **{k: v for k, v in kwargs.items() if k in {
            "p_tokens", "epochs", "learning_rate", "train_batch_size",
            "train_micro_batch_size", "prompt_loss_weight", "max_text_tokens", "seed",
        }},
    )
