#!/usr/bin/env python
"""13-gram overlap scan of query documents against a Hugging Face parquet corpus (2026-09-25).
Reproduces the benchmark's infini-gram filter statistic: for each query document, the fraction of its
Llama-2-tokenised 13-grams that occur anywhere in the corpus (benchmark threshold: 20%).

  build     query jsonl -> queries.pkl (token ids per doc, 13-gram index, Aho-Corasick automaton over the decoded
            inner 11-token string of every 13-gram = a necessary condition for a token-level match; no false negatives)
  scan      stream parquet files of a HF dataset (column projection: text,url,id), prefilter with the automaton,
            exact token-level check on a +-400-char window around every match; writes <out>/<slug>.hits.jsonl + .done
  summarize aggregate hits -> per-query overlap fraction, list of hit corpus docs, completeness check vs file list

Recall test: build --plant-file <repo path> --plant-n 30 adds 30 real corpus docs as queries (keys plant_*); scanning that
file must give them overlap 1.0.
"""
import argparse, json, os, pickle, sys, time, hashlib, re
from pathlib import Path

N = 13
INNER_MIN_CHARS = 16


def load_tok(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, add_bos_token=False, add_eos_token=False, use_fast=True)


def doc_grams(ids):
    return [tuple(ids[i:i + N]) for i in range(len(ids) - N + 1)]


def read_jsonl(p):
    with open(p) as f:
        for i, l in enumerate(f):
            if l.strip():
                r = json.loads(l); yield i, r


def cmd_build(a):
    import ahocorasick
    tok = load_tok(a.tokenizer)
    docs = {}  # key -> {"ids": [...], "n_grams": int, "meta": {...}}
    for qf in a.queries:
        stem = Path(qf).stem
        for i, r in read_jsonl(qf):
            key = f"{stem}#{i}"
            ids = tok(r["text"], add_special_tokens=False)["input_ids"]
            docs[key] = {"ids": ids, "n_grams": max(0, len(ids) - N + 1), "url": r.get("url", ""), "src": qf}
    if a.plant_file:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        with fs.open(f"datasets/{a.dataset}/{a.plant_file}", "rb") as fh:
            pf = pq.ParquetFile(fh); tbl = pf.read_row_group(0, columns=["text"]).to_pydict()["text"]
        for j, t in enumerate(tbl[:a.plant_n]):
            ids = tok(t, add_special_tokens=False)["input_ids"]
            docs[f"plant#{j}"] = {"ids": ids, "n_grams": max(0, len(ids) - N + 1), "url": "", "src": a.plant_file}
    gram_index = {}  # gram tuple -> list of (doc_key, position)
    A = ahocorasick.Automaton()
    n_pat = 0
    for key, d in docs.items():
        ids = d["ids"]
        for pos in range(len(ids) - N + 1):
            g = tuple(ids[pos:pos + N]); gram_index.setdefault(g, []).append((key, pos))
            inner = tok.decode(ids[pos + 1:pos + N - 1]).strip()
            if len(inner) < INNER_MIN_CHARS:
                inner = tok.decode(ids[pos:pos + N]).strip()
            if len(inner) >= 8 and not A.exists(inner):
                A.add_word(inner, 1); n_pat += 1
    A.make_automaton()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"docs": docs, "gram_index": gram_index, "automaton": A, "tokenizer": a.tokenizer, "N": N}, f)
    print(f"build: {len(docs)} query docs, {len(gram_index)} distinct 13-grams, {n_pat} patterns -> {a.out}", flush=True)


_G = {}


def _init(qpath, tokpath):
    global _G
    with open(qpath, "rb") as f:
        Q = pickle.load(f)
    _G = {"Q": Q, "tok": load_tok(tokpath)}
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _check_text(text):
    """-> {query_key: set(gram positions hit)} for one corpus doc, or {} """
    Q = _G["Q"]; A = Q["automaton"]; gi = Q["gram_index"]; tok = _G["tok"]
    ends = [end for end, _ in A.iter(text)]
    if not ends:
        return {}
    # merge windows around matches
    wins = []
    for e in ends:
        s, t = max(0, e - 400), min(len(text), e + 400)
        if wins and s <= wins[-1][1]:
            wins[-1][1] = max(wins[-1][1], t)
        else:
            wins.append([s, t])
    hits = {}
    for s, t in wins:
        ids = tok(text[s:t], add_special_tokens=False)["input_ids"]
        for g in doc_grams(ids):
            for key, pos in gi.get(g, ()):
                hits.setdefault(key, set()).add(pos)
    return hits


def _scan_file(args):
    dataset, rel, out_dir = args
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    slug = rel.replace("/", "__")
    done = Path(out_dir) / f"{slug}.done"
    if done.exists():
        return rel, "skip", 0, 0
    hits_path = Path(out_dir) / f"{slug}.hits.jsonl"
    n_rows = n_cand = n_hit = 0; t0 = time.time()
    for attempt in range(6):
        try:
            fs = HfFileSystem()
            with fs.open(f"datasets/{dataset}/{rel}", "rb", block_size=64 * 1024 * 1024) as fh, open(hits_path, "w") as hf:
                pf = pq.ParquetFile(fh)
                cols = [c for c in ("text", "url", "id") if c in pf.schema.names]
                for rg in range(pf.num_row_groups):
                    tb = pf.read_row_group(rg, columns=cols).to_pydict()
                    texts = tb["text"]; urls = tb.get("url", [""] * len(texts)); ids_ = tb.get("id", [""] * len(texts))
                    for i, text in enumerate(texts):
                        n_rows += 1
                        h = _check_text(text)
                        if h:
                            n_hit += 1
                            hf.write(json.dumps({"file": rel, "row": n_rows - 1, "url": urls[i], "id": str(ids_[i]),
                                                 "hits": {k: sorted(v) for k, v in h.items()}}) + "\n")
            done.write_text(json.dumps({"file": rel, "rows": n_rows, "hit_docs": n_hit, "sec": round(time.time() - t0, 1)}))
            return rel, "ok", n_rows, n_hit
        except Exception as e:  # retry on network errors
            err = repr(e)[:200]; time.sleep(30 * (attempt + 1)); n_rows = n_hit = 0
    return rel, f"FAILED {err}", 0, 0


def list_files(dataset, dumps_first, cache):
    if Path(cache).exists():
        files = json.loads(Path(cache).read_text())
    else:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(dataset, files_metadata=True)
        files = [(s.rfilename, s.size or 0) for s in info.siblings if s.rfilename.endswith(".parquet") and s.rfilename.startswith("data/")]
        Path(cache).write_text(json.dumps(files))
    first = set(dumps_first)
    def dump_of(p):
        m = re.match(r"data/([^/]+)/", p); return m.group(1) if m else ""
    files.sort(key=lambda x: (0 if dump_of(x[0]) in first else 1, x[0]))
    return files


def cmd_scan(a):
    from multiprocessing import Pool
    files = list_files(a.dataset, a.dumps_first.split(",") if a.dumps_first else [], a.file_list)
    if a.files:
        files = [(f, 0) for f in a.files]
    mine = [f for i, (f, _) in enumerate(files) if i % a.n_shards == a.shard]
    if a.limit: mine = mine[:a.limit]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print(f"scan: shard {a.shard}/{a.n_shards}: {len(mine)} files, workers={a.workers}", flush=True)
    with Pool(a.workers, initializer=_init, initargs=(a.queries, a.tokenizer)) as pool:
        for rel, st, n, h in pool.imap_unordered(_scan_file, [(a.dataset, f, a.out) for f in mine]):
            print(f"[{time.strftime('%H:%M:%S')}] {st:8s} rows={n:8d} hit_docs={h:5d} {rel}", flush=True)
    print("SCAN_SHARD_DONE", flush=True)


def cmd_summarize(a):
    with open(a.queries, "rb") as f:
        Q = pickle.load(f)
    docs = Q["docs"]; hit_pos = {k: set() for k in docs}; hit_docs = {k: [] for k in docs}
    done = 0; rows = 0; files_seen = set()
    for hp in sorted(Path(a.out).glob("*.hits.jsonl")):
        for l in open(hp):
            r = json.loads(l)
            for k, ps in r["hits"].items():
                hit_pos[k].update(ps); hit_docs[k].append((r["file"], r["url"], len(ps)))
    for dp in Path(a.out).glob("*.done"):
        d = json.loads(dp.read_text()); done += 1; rows += d["rows"]; files_seen.add(d["file"])
    res = {}
    for k, d in docs.items():
        frac = len(hit_pos[k]) / d["n_grams"] if d["n_grams"] else 0.0
        res[k] = {"n_grams": d["n_grams"], "grams_hit": len(hit_pos[k]), "overlap": round(frac, 4), "url": d["url"],
                  "corpus_docs": sorted(hit_docs[k], key=lambda x: -x[2])[:10]}
    Path(a.report).write_text(json.dumps({"files_done": done, "rows": rows, "per_query": res}, indent=1))
    groups = {}
    for k, r in res.items():
        groups.setdefault(k.split("#")[0], []).append(r["overlap"])
    print(f"files done: {done}  rows: {rows}")
    for g, v in groups.items():
        above = sum(1 for x in v if x > 0.2)
        print(f"  {g:60s} n={len(v):4d} overlap mean={sum(v)/len(v):.3f} max={max(v):.3f} >0={sum(1 for x in v if x>0)} >0.2={above}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--queries", nargs="+", required=True); b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", required=True); b.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    b.add_argument("--plant-file", default=None); b.add_argument("--plant-n", type=int, default=30)
    s = sub.add_parser("scan"); s.add_argument("--queries", required=True); s.add_argument("--tokenizer", required=True)
    s.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu"); s.add_argument("--file-list", default="fineweb_edu_files.json")
    s.add_argument("--dumps-first", default=""); s.add_argument("--files", nargs="*", default=None)
    s.add_argument("--shard", type=int, default=0); s.add_argument("--n-shards", type=int, default=1)
    s.add_argument("--workers", type=int, default=8); s.add_argument("--out", required=True); s.add_argument("--limit", type=int, default=0)
    m = sub.add_parser("summarize"); m.add_argument("--queries", required=True); m.add_argument("--out", required=True); m.add_argument("--report", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "scan": cmd_scan, "summarize": cmd_summarize}[a.cmd](a)
