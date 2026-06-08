'''
Constrained sampling via simulated annealing to build a contaminated split.
Select a subset D of the candidate pool that matches the uncontaminated split in
instance count (exact) and token count (within +-250), while minimizing the loss
L = JSD(p, q) + 1[temporal] * (|mu_p - mu_q| + |sigma_p - sigma_q|) / sigma_q.

Two phases:
1. token-matching (token count outside the margin): swap only if it moves the token count toward the target.
2. alignment (token count within margin): keep the swap in margin and accept by Metropolis -- always if delta L < 0, 
else with probability exp(-delta L / T). T starts at 1e-3 and decays by 0.999995 after each accepted swap.
'''


import json
import argparse
import math
import random
import glob
import os
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer


def get_nested(obj, dotted_key):
    # look up a possibly nested field, e.g. "metadata.WARC-Date"
    cur = obj
    for part in dotted_key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def parse_date_to_days(value):
    # convert a date into days (date ordinal); None if missing or unparseable
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    # benchmark dates are ISO 8601 with a trailing Z, e.g. 2021-06-22T02:00:35Z
    iso = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
        return dt.toordinal() + (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
    except ValueError:
        pass

    # fallback formats, just in case
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", "%Y%m%d",
                "%a, %d %b %Y %H:%M:%S", "%d %b %Y"):
        try:
            return float(datetime.strptime(s, fmt).toordinal())
        except ValueError:
            continue
    return None


def load_instances(paths, text_field):
    # read every .jsonl line from a file or directory into a list of dicts
    if os.path.isdir(paths):
        files = sorted(glob.glob(os.path.join(paths, "**", "*.jsonl"), recursive=True))
    else:
        files = [paths]
    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if text_field in data and data[text_field] is not None:
                    rows.append(data)
    return rows


def jensen_shannon(d_counts, total, q_prob):
    # JSD between the selected split's token distribution and the uncontaminated q_prob
    p = d_counts / total
    m = 0.5 * (p + q_prob)
    out = 0.0
    mask = p > 0
    out += 0.5 * np.sum(p[mask] * np.log(p[mask] / m[mask]))
    mask = q_prob > 0
    out += 0.5 * np.sum(q_prob[mask] * np.log(q_prob[mask] / m[mask]))
    return float(out)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate_path", type=str, required=True,
                        help="Candidate pool: a .jsonl file or a directory of .jsonl files.")
    parser.add_argument("--uncon_path", type=str, required=True,
                        help="Reference uncontaminated split (.jsonl) to match against.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to write the selected contaminated split (.jsonl).")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Field holding the text (default: text).")
    parser.add_argument("--date_field", type=str, default=None,
                        help="Date field for temporal alignment (ISO 8601, e.g. "
                             "'2021-06-22T02:00:35Z'). Per domain: DCLM -> "
                             "'metadata.WARC-Date'; OpenWebMath / peS2o / Stack Exchange "
                             "-> 'created'. Omit to disable temporal alignment (e.g. for "
                             "GSM8K, StarCoder, SFT, DPO, RLVR, which have no dates).")
    parser.add_argument("--uncon_date_field", type=str, default=None,
                        help="Date field in the uncontaminated split (default: same as --date_field).")
    parser.add_argument("--token_margin", type=int, default=250,
                        help="Allowed deviation from the target token count (default: 250).")
    parser.add_argument("--max_steps", type=int, default=10_000_000,
                        help="Number of simulated-annealing steps (default: 1e7).")
    parser.add_argument("--temperature", type=float, default=1e-3,
                        help="Initial temperature T0 (default: 1e-3).")
    parser.add_argument("--cooling_rate", type=float, default=0.999995,
                        help="Geometric cooling factor per accepted swap (default: 0.999995).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_temporal = args.date_field is not None
    uncon_date_field = args.uncon_date_field or args.date_field

    print("configurations:")
    print("  candidate path:", args.candidate_path)
    print("  uncontaminated path:", args.uncon_path)
    print("  output path:", args.output_path)
    print("  temporal alignment:", use_temporal)

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", add_bos_token=False, add_eos_token=False
    )

    # load reference uncontaminated split
    uncon_rows = load_instances(args.uncon_path, args.text_field)
    target_n = len(uncon_rows)
    if target_n == 0:
        raise ValueError("the uncontaminated reference split is empty.")

    token_to_idx = {}

    def encode(text):
        ids = tokenizer.encode(text)
        local = {}
        for t in ids:
            if t not in token_to_idx:
                token_to_idx[t] = len(token_to_idx)
            local[token_to_idx[t]] = local.get(token_to_idx[t], 0) + 1
        return len(ids), local

    target_tokens = 0
    q_dates = []
    q_accum = {}  # token -> total count in the uncontaminated split
    for row in uncon_rows:
        n_tok, local = encode(row[args.text_field])
        target_tokens += n_tok
        for k, v in local.items():
            q_accum[k] = q_accum.get(k, 0) + v
        if use_temporal:
            d = parse_date_to_days(get_nested(row, uncon_date_field))
            if d is not None:
                q_dates.append(d)

    # uncontaminated temporal stats
    mu_q = sigma_q = None
    if use_temporal and q_dates:
        mu_q = float(np.mean(q_dates))
        sigma_q = float(np.std(q_dates))
        if sigma_q == 0.0:
            sigma_q = 1.0  # all dates coincide; avoid divide-by-zero

    # load candidate pool and tokenize
    cand_rows = load_instances(args.candidate_path, args.text_field)
    if len(cand_rows) < target_n:
        raise ValueError(
            f"candidate pool ({len(cand_rows)}) is smaller than the target "
            f"instance count ({target_n})."
        )

    cand_idx = []     # token indices per instance
    cand_cnt = []     # token counts per instance
    cand_ntok = []    # total tokens per instance
    cand_date = []    # date in days (or None) per instance
    for row in cand_rows:
        n_tok, local = encode(row[args.text_field])
        idx = np.fromiter(local.keys(), dtype=np.int64, count=len(local))
        cnt = np.fromiter(local.values(), dtype=np.float64, count=len(local))
        cand_idx.append(idx)
        cand_cnt.append(cnt)
        cand_ntok.append(n_tok)
        cand_date.append(parse_date_to_days(get_nested(row, args.date_field)) if use_temporal else None)

    support = len(token_to_idx)
    q_prob = np.zeros(support, dtype=np.float64)
    for k, v in q_accum.items():
        q_prob[k] = v
    q_prob /= q_prob.sum()

    # start from a random subset D of size target_n
    order = list(range(len(cand_rows)))
    random.shuffle(order)
    d_list = order[:target_n]          # currently in D
    leftover_list = order[target_n:]   # currently outside D

    d_counts = np.zeros(support, dtype=np.float64)
    cur_tokens = 0
    s1 = s2 = date_n = 0.0  # running date sum, sum-of-squares, count
    for ci in d_list:
        d_counts[cand_idx[ci]] += cand_cnt[ci]
        cur_tokens += cand_ntok[ci]
        if use_temporal and cand_date[ci] is not None:
            s1 += cand_date[ci]
            s2 += cand_date[ci] ** 2
            date_n += 1

    def temporal_loss(s1, s2, date_n):
        if not use_temporal or mu_q is None or date_n == 0:
            return 0.0
        mu_p = s1 / date_n
        var = max(s2 / date_n - mu_p * mu_p, 0.0)
        sigma_p = math.sqrt(var)
        return (abs(mu_p - mu_q) + abs(sigma_p - sigma_q)) / sigma_q

    cur_L = jensen_shannon(d_counts, cur_tokens, q_prob) + temporal_loss(s1, s2, date_n)

    T = args.temperature
    margin = args.token_margin
    accepted = 0

    for step in range(args.max_steps):
        pi = random.randrange(target_n)
        pj = random.randrange(len(leftover_list))
        i = d_list[pi]        # leaving D
        j = leftover_list[pj] # entering D

        new_tokens = cur_tokens - cand_ntok[i] + cand_ntok[j]
        in_margin = abs(cur_tokens - target_tokens) <= margin

        if not in_margin:
            # greedy token-matching phase: only swap if it moves toward the target tokens
            if abs(new_tokens - target_tokens) < abs(cur_tokens - target_tokens):
                d_counts[cand_idx[i]] -= cand_cnt[i]
                d_counts[cand_idx[j]] += cand_cnt[j]
                cur_tokens = new_tokens
                if use_temporal:
                    if cand_date[i] is not None:
                        s1 -= cand_date[i]; s2 -= cand_date[i] ** 2; date_n -= 1
                    if cand_date[j] is not None:
                        s1 += cand_date[j]; s2 += cand_date[j] ** 2; date_n += 1
                d_list[pi], leftover_list[pj] = j, i
            continue

        # alignment phase: skip swaps that would leave the margin
        if abs(new_tokens - target_tokens) > margin:
            continue

        # try the swap, compute L, then keep or revert
        d_counts[cand_idx[i]] -= cand_cnt[i]
        d_counts[cand_idx[j]] += cand_cnt[j]
        ns1, ns2, ndate_n = s1, s2, date_n
        if use_temporal:
            if cand_date[i] is not None:
                ns1 -= cand_date[i]; ns2 -= cand_date[i] ** 2; ndate_n -= 1
            if cand_date[j] is not None:
                ns1 += cand_date[j]; ns2 += cand_date[j] ** 2; ndate_n += 1
        new_L = jensen_shannon(d_counts, new_tokens, q_prob) + temporal_loss(ns1, ns2, ndate_n)
        dL = new_L - cur_L

        if dL < 0 or random.random() < math.exp(-dL / T):
            cur_tokens = new_tokens
            cur_L = new_L
            s1, s2, date_n = ns1, ns2, ndate_n
            d_list[pi], leftover_list[pj] = j, i
            accepted += 1
            T *= args.cooling_rate
        else:
            # revert
            d_counts[cand_idx[j]] -= cand_cnt[j]
            d_counts[cand_idx[i]] += cand_cnt[i]

        if (step + 1) % 100000 == 0:
            print("step {}/{}  L={:.6f}  tokens={} (target {})  T={:.3e}  accepted={}".format(
                step + 1, args.max_steps, cur_L, cur_tokens, target_tokens, T, accepted))

    print("done. final L={:.6f}  tokens={} (target {}, margin +-{})  instances={}".format(
        cur_L, cur_tokens, target_tokens, margin, len(d_list)))

    if abs(cur_tokens - target_tokens) > margin:
        raise RuntimeError("final token count {} is outside the +-{} margin of the target {}".format(cur_tokens, margin, target_tokens))

    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for ci in d_list:
            out_f.write(json.dumps(cand_rows[ci], ensure_ascii=False) + "\n")
    print("wrote {} contaminated instances to {}".format(len(d_list), args.output_path))


if __name__ == "__main__":
    main()
