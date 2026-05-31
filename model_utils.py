from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from progress_utils import progress
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HFGenerationAdapter:
    """
    Hugging Face model adapter shared across detection methods.
    - Generate path (used by RL-MIA)
    - Scoring path (used by Min-K methods)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        seed: int = 42,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._seed = seed
        self._device = next(self._model.parameters()).device

    def generate(self, prompts: str | list[str], sampling_params: Any) -> list[dict[str, Any]]:
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = prompts

        raw_max_tokens = getattr(sampling_params, "max_tokens", 1024)
        raw_temperature = getattr(sampling_params, "temperature", 0.0)
        raw_batch_size = getattr(sampling_params, "batch_size", 1)
        raw_progress_desc = getattr(sampling_params, "progress_desc", "selfcrit generate")

        max_tokens = int(1024 if raw_max_tokens is None else raw_max_tokens)
        temperature = float(0.0 if raw_temperature is None else raw_temperature)
        batch_size = int(1 if raw_batch_size is None else raw_batch_size)
        progress_desc = str(raw_progress_desc)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        seed = getattr(sampling_params, "seed", self._seed)
        do_sample = temperature > 0.0

        generator = None
        if do_sample:
            run_seed = self._seed if seed is None else seed
            generator = torch.Generator(device=str(self._device))
            generator.manual_seed(int(run_seed))

        outputs: list[dict[str, Any]] = []
        original_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "left"
        try:
            for start in progress(
                range(0, len(prompt_list), batch_size),
                desc=progress_desc,
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            ):
                batch_prompts = prompt_list[start : start + batch_size]
                encoded = self._tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}

                generation_kwargs = dict(
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
                if do_sample:
                    generation_kwargs["temperature"] = temperature
                    generation_kwargs["generator"] = generator

                with torch.no_grad():
                    generated = self._model.generate(**encoded, **generation_kwargs)

                # New tokens begin after padded prompt length for all rows.
                prompt_len = encoded["input_ids"].shape[1]
                generated_ids_by_row = generated.sequences[:, prompt_len:]

                for row_idx in range(generated_ids_by_row.shape[0]):
                    generated_ids = generated_ids_by_row[row_idx].tolist()

                    kept_token_ids: list[int] = []
                    generated_token_entropies: list[float] = []
                    for step_idx, token_id in enumerate(generated_ids):
                        if token_id == self._tokenizer.eos_token_id:
                            break
                        if self._tokenizer.pad_token_id is not None and token_id == self._tokenizer.pad_token_id:
                            break

                        kept_token_ids.append(token_id)
                        if step_idx < len(generated.scores):
                            step_logits = generated.scores[step_idx][row_idx]
                            step_log_probs = torch.log_softmax(step_logits, dim=-1)
                            step_probs = step_log_probs.exp()
                            entropy = float(-(step_probs * step_log_probs).sum())
                            generated_token_entropies.append(entropy)
                        else:
                            generated_token_entropies.append(float("nan"))

                    text = self._tokenizer.decode(kept_token_ids, skip_special_tokens=True).strip()
                    outputs.append(
                        {
                            "text": text,
                            "generated_token_entropies": generated_token_entropies,
                        }
                    )
        finally:
            self._tokenizer.padding_side = original_padding_side
        return outputs

    def _score_batch(
        self,
        questions: list[str],
        answers: list[str],
        batch_size: int = 3,
        mode: str = "chat",
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
        """
        Internal batched scoring.
        Returns:
            full_token_logits: list of 2D tensors [answer_len, vocab]
            target_token_ids: list of 1D tensors [answer_len]
            target_texts_raw: list of decoded strings from unfiltered target ids
        """
        if self._model is None or self._tokenizer is None:
            raise ValueError("model and tokenizer must be provided.")
        if mode not in {"chat", "plain"}:
            raise ValueError("mode must be 'chat' or 'plain'.")
        if len(questions) != len(answers):
            raise ValueError("questions and answers must have the same length.")
        self._tokenizer.padding_side = "right"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        full_token_logits: list[torch.Tensor] = []
        target_token_ids: list[torch.Tensor] = []
        target_texts_raw: list[str] = []
        special_ids = set(self._tokenizer.all_special_ids)

        for i in range(0, len(questions), batch_size):
            batch_q = questions[i : i + batch_size]
            batch_a = answers[i : i + batch_size]

            prompt_ids_list = []
            full_ids_list = []
            prompt_lens = []

            # Build input IDs for each item in the batch
            for q, a in zip(batch_q, batch_a):
                if mode == "chat":
                    user_messages = [{"role": "user", "content": q}]
                    prompt_ids = self._tokenizer.apply_chat_template(
                        user_messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    full_messages = [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                    full_ids = self._tokenizer.apply_chat_template(
                        full_messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                    if len(prompt_ids) >= len(full_ids):
                        raise ValueError("Answer is empty after tokenization.")
                    prompt_ids_list.append(prompt_ids)
                    full_ids_list.append(full_ids)
                    prompt_lens.append(len(prompt_ids))
                else:
                    # Plain text: score the full text as the "answer"
                    full_ids = self._tokenizer(
                        a, 
                        add_special_tokens=True,
                        truncation=True,        
                        max_length=4096).input_ids
                    prompt_ids_list.append([])
                    full_ids_list.append(full_ids)
                    prompt_lens.append(0)

            # Padding to max length in batch
            max_len = max(len(x) for x in full_ids_list)
            input_ids = torch.full(
                (len(full_ids_list), max_len),
                self._tokenizer.pad_token_id,
                dtype=torch.long,
                device=self._device,
            )

            # Preparing attention masks and labels
            attention_mask = torch.zeros_like(input_ids)
            labels = torch.full_like(input_ids, -100)

            for b, full_ids in enumerate(full_ids_list):
                seq_len = len(full_ids)
                input_ids[b, :seq_len] = torch.tensor(full_ids, dtype=torch.long, device=self._device)
                # Set attention mask to 1 for all tokens up to the end of the answer
                attention_mask[b, :seq_len] = 1
                # Set labels to the answer tokens, ignoring the prompt tokens
                labels[b, prompt_lens[b] : seq_len] = input_ids[b, prompt_lens[b] : seq_len]

            # Get the logits
            with torch.no_grad():
                out = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits  # [B, T, V]

            # Next prediction alignment
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            del logits, out

            for b in range(shift_labels.size(0)):
                mask = shift_labels[b] != -100
                # Edge case of no valid answer tokens, unlikely happen
                if mask.sum() == 0:
                    full_token_logits.append(torch.empty((0, shift_logits.size(-1)), device=self._device))
                    target_token_ids.append(torch.empty(0, dtype=torch.long, device=self._device))
                    continue
                # Current Answer token positions
                idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                # Collect full logits for answer tokens and their corresponding target token IDs
                full = shift_logits[b, idx, :].detach()
                targets = shift_labels[b][mask].detach()

                if self._tokenizer.eos_token_id is not None and targets.numel() > 0:
                    if targets[-1].item() == self._tokenizer.eos_token_id:
                        targets = targets[:-1]
                        full = full[:-1]
                if special_ids:
                    has_special = any(t.item() in special_ids for t in targets)
                    if has_special:
                        raise ValueError("Special token found in target ids after EOS removal.")
                raw_text = self._tokenizer.decode(targets, skip_special_tokens=False)
                target_texts_raw.append(raw_text)
                full_token_logits.append(full)
                target_token_ids.append(targets)
            
            del shift_logits, shift_labels
            torch.cuda.empty_cache()

        return full_token_logits, target_token_ids, target_texts_raw

    def score_chat_batch(
        self,
        questions: list[str],
        answers: list[str],
        batch_size: int = 3,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
        """Q-A chat format: question and gold answer."""
        return self._score_batch(
            questions=questions,
            answers=answers,
            batch_size=batch_size,
            mode="chat",
        )

    def score_plain_batch(
        self,
        texts: list[str],
        batch_size: int = 3,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
        """Plain text format: score each paragraph/sentence as-is."""
        return self._score_batch(
            questions=texts,
            answers=texts,
            batch_size=batch_size,
            mode="plain",
        )


    def token_logprob_distribution_batch(
        self,
        texts: list[str],
        batch_size: int = 3,
        max_tokens: int = 4096,
        progress_desc: str = "token prob",
    ) -> list[dict[str, Any]]:
        """
        Compute next-token log-prob distribution statistics for each input text.
        Returns one dict per text with keys: input_ids, token_log_probs, mu, sigma.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if max_tokens < 2:
            raise ValueError("max_tokens must be >= 2.")

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        outputs: list[dict[str, Any]] = []
        original_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = "right"

        try:
            for start in progress(
                range(0, len(texts), batch_size),
                desc=progress_desc,
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            ):
                batch_texts = texts[start : start + batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                    add_special_tokens=True,
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}

                with torch.no_grad():
                    logits = self._model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    ).logits

                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                for row_idx in range(input_ids.shape[0]):
                    seq_len = int(attention_mask[row_idx].sum().item())
                    if seq_len <= 1:
                        outputs.append(
                            {
                                "input_ids": [],
                                "token_log_probs": [],
                                "mu": [],
                                "sigma": [],
                            }
                        )
                        continue

                    row_input_ids = input_ids[row_idx, :seq_len]
                    row_logits = logits[row_idx, : seq_len - 1, :]
                    target_ids = row_input_ids[1:]

                    log_probs = torch.log_softmax(row_logits, dim=-1)
                    probs = log_probs.exp()
                    mu = (probs * log_probs).sum(dim=-1)
                    sigma = (probs * torch.square(log_probs)).sum(dim=-1) - torch.square(mu)

                    row_index = torch.arange(log_probs.shape[0], device=self._device)
                    token_log_probs = log_probs[row_index, target_ids]

                    outputs.append(
                        {
                            "input_ids": target_ids.detach().cpu().tolist(),
                            "token_log_probs": token_log_probs.detach().cpu().tolist(),
                            "mu": mu.detach().cpu().tolist(),
                            "sigma": sigma.detach().cpu().tolist(),
                        }
                    )
                
                del logits
                torch.cuda.empty_cache()
        finally:
            self._tokenizer.padding_side = original_padding_side

        return outputs


@dataclass
class ModelBundle:
    model_path: str
    model_name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    llm_adapter: HFGenerationAdapter
    seed: int
    runtime_cache: dict[str, Any] = field(default_factory=dict)


# def load_model_bundle(model_path: str, seed: int = 42) -> ModelBundle:
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         trust_remote_code=True,
#         torch_dtype="auto",
#         device_map="auto",
#     )
#     model.eval()

#     adapter = HFGenerationAdapter(model=model, tokenizer=tokenizer, seed=seed)
#     return ModelBundle(
#         model_path=model_path,
#         model_name=Path(model_path).name,
#         model=model,
#         tokenizer=tokenizer,
#         llm_adapter=adapter,
#         seed=seed,
#     )

def load_model_bundle(model_path: str, seed: int = 42) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if the model is large (13B or 32B) to apply quantization
    model_name = Path(model_path).name.lower()
    load_in_4bit = "13b" in model_name or "32b" in model_name

    if load_in_4bit:
        print(f"--> Large model detected ({model_name}). Loading in 4-bit to save VRAM...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for OLMo2
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quant_config, # Pass the config here
    )
    model.eval()
    adapter = HFGenerationAdapter(model=model, tokenizer=tokenizer, seed=seed)

    return ModelBundle(
        model_path=model_path,
        model_name=Path(model_path).name,
        model=model,
        tokenizer=tokenizer,
        llm_adapter=adapter,
        seed=seed,
    )
