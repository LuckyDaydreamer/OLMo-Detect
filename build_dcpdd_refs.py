#!/usr/bin/env python
"""
Build DC-PDD C4 token-frequency references for NON-OLMo2 tokenizers (e.g. SmolLM2-1.7B),
matching the released OLMo2 reference scale (max_tokens_per_sample=4096,
~1.07M C4 docs). Reads C4 *.json.gz shards directly (no arrow conversion needed) and writes
methods/dcpdd_data/c4_token_occurrence_<model_name>.json in the SAME payload format the DCPDD
method loader expects (token_frequency dict keyed by str(token_id); counts include repeats up to
max_tokens per doc, add_special_tokens=True — identical semantics to build_token_frequency_for_tokenizer).

Usage:
  python build_dcpdd_refs.py --models olmo_models/SmolLM2-1.7B \
      --c4-glob 'data/c4_raw/en/*.json.gz' --out-dir methods/dcpdd_data \
      --max-tokens 4096 --max-docs 1068952 --batch-size 512
Idempotent: skips a model whose output json already exists (unless --force).
"""
from __future__ import annotations
import argparse, glob, gzip, json
from pathlib import Path
from transformers import AutoTokenizer


def iter_c4_texts(c4_glob: str):
    for fp in sorted(glob.glob(c4_glob)):
        with gzip.open(fp, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("text")
                if isinstance(t, str) and t:
                    yield t


def build_for_model(model_dir: str, c4_glob: str, out_dir: Path,
                    max_tokens: int, max_docs: int, batch_size: int, force: bool):
    model_name = Path(model_dir).name
    out_path = out_dir / f"c4_token_occurrence_{model_name}.json"
    if out_path.exists() and not force:
        print(f"[skip] {model_name}: {out_path} already exists", flush=True)
        return
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    vocab_size = len(tok)
    freq = [0] * vocab_size
    n = 0
    batch: list[str] = []

    def flush(b: list[str]):
        enc = tok(b, add_special_tokens=True, truncation=True, max_length=max_tokens)["input_ids"]
        for ids in enc:
            for tid in ids:
                if 0 <= tid < vocab_size:
                    freq[tid] += 1

    for text in iter_c4_texts(c4_glob):
        if n >= max_docs:
            break
        batch.append(text)
        n += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
        if n % 100000 == 0:
            print(f"  [{model_name}] processed {n} docs...", flush=True)
    if batch:
        flush(batch)

    tf_map = {str(i): int(c) for i, c in enumerate(freq) if c != 0}
    payload = {
        "model_name":                  model_name,
        "model_path":                  model_dir,
        "tokenizer_name_or_path":      getattr(tok, "name_or_path", model_dir),
        "c4_dir":                      c4_glob,
        "max_tokens_per_sample":       max_tokens,
        "vocab_size":                  vocab_size,
        "num_processed_examples":      n,
        "num_non_zero_token_ids":      len(tf_map),
        "num_total_token_occurrences": int(sum(tf_map.values())),
        "token_frequency":             tf_map,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[done] {model_name}: docs={n} vocab={vocab_size} "
          f"nonzero={len(tf_map)} total_occ={payload['num_total_token_occurrences']} -> {out_path}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--c4-glob", default="data/c4_raw/en/*.json.gz")
    ap.add_argument("--out-dir", default="methods/dcpdd_data")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-docs", type=int, default=1068952)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    n_files = len(sorted(glob.glob(a.c4_glob)))
    print(f"C4 glob '{a.c4_glob}' -> {n_files} shard(s); max_docs={a.max_docs} max_tokens={a.max_tokens}", flush=True)
    if n_files == 0:
        raise FileNotFoundError(f"No C4 shards match {a.c4_glob}")
    for m in a.models:
        build_for_model(m, a.c4_glob, Path(a.out_dir), a.max_tokens, a.max_docs, a.batch_size, a.force)


if __name__ == "__main__":
    main()
