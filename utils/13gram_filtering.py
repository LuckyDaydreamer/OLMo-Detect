'''
13-gram filtering to obtain uncontaminated instances.
'''


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from nltk.util import ngrams
import json
from infini_gram.engine import InfiniGramEngine
from infini_gram_mini.engine.src.engine import InfiniGramMiniEngine
import argparse
import time

import fasttext
from collections import Counter
import regex

WHITESPACE_REGEX = regex.compile(r"\w+|[^\w\s]+")


os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    add_bos_token=False,
    add_eos_token=False
)


INFINI_GRAM_ENGINE_DIRS = {
    # pretraining
    "non_dclm": "OLMo2_Infini_Gram_Corpus/pretraining/Non_DCLM",

    # midtraining
    'dclm_50b': 'OLMo2_Infini_Gram_Corpus/midtraining/dclm_50B/index',
    'dclm_100B': 'OLMo2_Infini_Gram_Corpus/midtraining/dclm_100B/index',
    'dclm_300B': 'OLMo2_Infini_Gram_Corpus/midtraining/dclm_300B/index',
    'flan_50B': 'OLMo2_Infini_Gram_Corpus/midtraining/flan_50B/index',
    'flan_all': 'OLMo2_Infini_Gram_Corpus/midtraining/flan_all/index',
    'stack_exchange': 'OLMo2_Infini_Gram_Corpus/midtraining/stack_exchange/index',
    'peS2o_50B': 'OLMo2_Infini_Gram_Corpus/midtraining/peS2o_50B/index',
    'peS2o_100B': 'OLMo2_Infini_Gram_Corpus/midtraining/peS2o_100B/index',
    'peS2o_all': 'OLMo2_Infini_Gram_Corpus/midtraining/peS2o_all/index',
    'wiki': 'OLMo2_Infini_Gram_Corpus/midtraining/wiki/index',
    'math': 'OLMo2_Infini_Gram_Corpus/midtraining/math/index',

    # posttraining/sft
    "sft_1b-32b": "OLMo2_Infini_Gram_Corpus/posttraining/SFT/1b-32b/index",
    "sft_7b-13b": "OLMo2_Infini_Gram_Corpus/posttraining/SFT/7b-13b/index",
    
    # posttraining/dpo
    'pref_1b': 'OLMo2_Infini_Gram_Corpus/posttraining/DPO/1b_pref/index',
    'pref_7b': 'OLMo2_Infini_Gram_Corpus/posttraining/DPO/7b_pref/index',
    'pref_13b': 'OLMo2_Infini_Gram_Corpus/posttraining/DPO/13b_pref/index',
    'pref_32b': 'OLMo2_Infini_Gram_Corpus/posttraining/DPO/32b_pref/index',

    # posttraining/rlvr 
    'rlvr': 'OLMo2_Infini_Gram_Corpus/posttraining/RLVR/index',
    # additional rlvr runs
    "cot_gsm8k_train": "OLMo2_Infini_Gram_Corpus/posttraining/CoT_GSM8K/index",
    "cot_math_train": "OLMo2_Infini_Gram_Corpus/posttraining/CoT_MATH/index",
}



_INFINI_GRAM_ENGINE_CACHE = {}
for name, dir_path in INFINI_GRAM_ENGINE_DIRS.items():
    _INFINI_GRAM_ENGINE_CACHE[name] = InfiniGramEngine(index_dir=dir_path, eos_token_id=tokenizer.eos_token_id)


_INFINI_GRAM_MINI_ENGINE_CACHE = {}
index_dirs = [f"OLMo2_Infini_Gram_Corpus/pretraining/DCLM/{i:02d}" for i in range(25)]
_INFINI_GRAM_MINI_ENGINE_CACHE["dclm"] = InfiniGramMiniEngine(index_dirs=index_dirs, load_to_ram=False, get_metadata=False)


def get_engine(name):
    if name == "dclm":
        return _INFINI_GRAM_MINI_ENGINE_CACHE[name]
    else:
        return _INFINI_GRAM_ENGINE_CACHE[name]


def count_engine_infini_gram(name, query_ids):
    engine = get_engine(name)
    return engine.count(query_ids)['count']


# infini-gram-mini takes raw text as input, rather than token IDs
def count_engine_infini_gram_mini(name, queries):
    engine = get_engine(name)
    return engine.count(queries)['count']


def count_dolmino_mix_50b(query_ids):
    return (
        count_engine_infini_gram('dclm_50b', query_ids)
        + count_engine_infini_gram('flan_50B', query_ids)
        + count_engine_infini_gram('stack_exchange', query_ids)
        + count_engine_infini_gram('peS2o_50B', query_ids)
        + count_engine_infini_gram('wiki', query_ids)
        + count_engine_infini_gram('math', query_ids)
    )


def count_dolmino_mix_100b(query_ids):
    return (
        count_engine_infini_gram('dclm_100B', query_ids)
        + count_engine_infini_gram('flan_all', query_ids)
        + count_engine_infini_gram('stack_exchange', query_ids) * 2
        + count_engine_infini_gram('peS2o_100B', query_ids)
        + count_engine_infini_gram('wiki', query_ids)
        + count_engine_infini_gram('math', query_ids) * 2
    )


def count_dolmino_mix_300b(query_ids):
    return (
        count_engine_infini_gram('dclm_300B', query_ids)
        + count_engine_infini_gram('flan_all', query_ids) * 2
        + count_engine_infini_gram('stack_exchange', query_ids) * 4
        + count_engine_infini_gram('peS2o_all', query_ids)
        + count_engine_infini_gram('wiki', query_ids) * 4
        + count_engine_infini_gram('math', query_ids) * 4
    )


def count_1b_model(query_ids, queries, is_variant = False):
    if not is_variant:
        olmo_mix_dclm_count = count_engine_infini_gram_mini("dclm", queries)

    olmo_mix_non_dclm_count = count_engine_infini_gram("non_dclm", query_ids)

    dolmino_count = count_dolmino_mix_50b(query_ids)

    sft_1b = count_engine_infini_gram("sft_1b-32b", query_ids)
    pref_1b = count_engine_infini_gram('pref_1b', query_ids)
    rlvr = count_engine_infini_gram("rlvr", query_ids)
    # additional CoT-MATH for rlvr
    cot_math_train = count_engine_infini_gram('cot_math_train', query_ids)

    if not is_variant:
        return olmo_mix_dclm_count + olmo_mix_non_dclm_count + dolmino_count + sft_1b * 2 + pref_1b + rlvr + cot_math_train
    else:
        return olmo_mix_non_dclm_count + dolmino_count + sft_1b * 2 + pref_1b + rlvr + cot_math_train


def count_7b_model(query_ids, queries, is_variant = False):
    if not is_variant:
        olmo_mix_dclm_count = count_engine_infini_gram_mini("dclm", queries)

    olmo_mix_non_dclm_count = count_engine_infini_gram("non_dclm", query_ids)

    dolmino_count = count_dolmino_mix_50b(query_ids) * 3

    sft_7b = count_engine_infini_gram("sft_7b-13b", query_ids)
    pref_7b = count_engine_infini_gram('pref_7b', query_ids)
    rlvr = count_engine_infini_gram("rlvr", query_ids)

    if not is_variant:
        return olmo_mix_dclm_count + olmo_mix_non_dclm_count + dolmino_count + sft_7b * 2 + pref_7b + rlvr
    else:
        return olmo_mix_non_dclm_count + dolmino_count + sft_7b * 2 + pref_7b + rlvr
    

def count_13b_model(query_ids, queries, is_variant = False):
    if not is_variant:
        olmo_mix_dclm_count = count_engine_infini_gram_mini("dclm", queries)

    olmo_mix_non_dclm_count = count_engine_infini_gram("non_dclm", query_ids)

    dolmino_count = count_dolmino_mix_100b(query_ids) * 3 + count_dolmino_mix_300b(query_ids)

    sft_13b = count_engine_infini_gram("sft_7b-13b", query_ids)
    pref_13b = count_engine_infini_gram('pref_13b', query_ids)
    rlvr = count_engine_infini_gram("rlvr", query_ids)
    # additional CoT-GSM8K + CoT-MATH
    cot_gsm8k_train = count_engine_infini_gram('cot_gsm8k_train', query_ids)
    cot_math_train = count_engine_infini_gram('cot_math_train', query_ids)

    if not is_variant:
        return olmo_mix_dclm_count + olmo_mix_non_dclm_count + dolmino_count + sft_13b * 2 + pref_13b + rlvr + cot_gsm8k_train + cot_math_train
    else:
        return olmo_mix_non_dclm_count + dolmino_count + sft_13b * 2 + pref_13b + rlvr + cot_gsm8k_train + cot_math_train
    

def count_32b_model(query_ids, queries, is_variant = False):
    if not is_variant:
        olmo_mix_dclm_count = count_engine_infini_gram_mini("dclm", queries)

    olmo_mix_non_dclm_count = count_engine_infini_gram("non_dclm", query_ids)

    dolmino_count = count_dolmino_mix_100b(query_ids) * 3 + count_dolmino_mix_300b(query_ids)

    sft_32b = count_engine_infini_gram("sft_1b-32b", query_ids)
    pref_32b = count_engine_infini_gram('pref_32b', query_ids)
    rlvr = count_engine_infini_gram("rlvr", query_ids)

    if not is_variant:
        return olmo_mix_dclm_count + olmo_mix_non_dclm_count + dolmino_count + sft_32b * 2 + pref_32b + rlvr
    else:
        return olmo_mix_non_dclm_count + dolmino_count + sft_32b * 2 + pref_32b + rlvr


def get_counts(text, model):
    tokenized_tokens = tokenizer.tokenize(text)

    if len(tokenized_tokens) < 13:
        print("This instance is too short; discard it.")
        return 1.0
    
    token_ngrams = [list(g) for g in ngrams(tokenized_tokens, 13)]
    token_ngrams_input_ids = [tokenizer.convert_tokens_to_ids(ng) for ng in token_ngrams]
    ngram_to_texts = [tokenizer.convert_tokens_to_string(ng) for ng in token_ngrams]

    # llama2 tokenizer is very sensitive to the first token's variance
    variant_token_ngrams = []
    for ng in token_ngrams:
        first_token = ng[0]
        if first_token.isdigit() or first_token == '▁' or first_token == '▁▁':
            variant_token_ngrams.append(None)
        else:
            if first_token.startswith('▁'):
                new_first = first_token.lstrip('▁')
            else:
                new_first = '▁' + first_token
            variant = [new_first] + ng[1:]
            variant_token_ngrams.append(variant)

    variant_token_ngrams_input_ids = [
        tokenizer.convert_tokens_to_ids(ng) if ng is not None else None
        for ng in variant_token_ngrams
    ]

    ngram_counts = []
    non_zero_count = 0
    for input_ids, input_texts in zip(token_ngrams_input_ids, ngram_to_texts):
        if model == '1b':
            count = count_1b_model(input_ids, input_texts)
        elif model == '7b':
            count = count_7b_model(input_ids, input_texts)
        elif model == '13b':
            count = count_13b_model(input_ids, input_texts)
        elif model == '32b':
            count = count_32b_model(input_ids, input_texts)
        else:
            raise ValueError('wrong model type')
        ngram_counts.append(count)

        if count > 0:
            non_zero_count += 1
        if (non_zero_count / len(token_ngrams_input_ids)) >= 0.2:
            print("the contaminated 13-grams' percentage already surpasses 20%. Early stop the search.")
            return 1.0

    variant_ngram_counts = []
    for input_ids in variant_token_ngrams_input_ids:
        if input_ids is None:
            count = 0
        else:
            if model == '1b':
                count = count_1b_model(input_ids, None, True)
            elif model == '7b':
                count = count_7b_model(input_ids, None, True)
            elif model == '13b':
                count = count_13b_model(input_ids, None, True)
            elif model == '32b':
                count = count_32b_model(input_ids, None, True)
            else:
                raise ValueError('wrong model type')
        variant_ngram_counts.append(count)
    
    all_counts = [ngram_count + variant_ngram_count for ngram_count, variant_ngram_count in zip(ngram_counts, variant_ngram_counts)]
    non_zero_count_percent = sum(count > 0 for count in all_counts) / len(all_counts)

    print("all counts:")
    print(all_counts)
    print("non_zero_count_percent:")
    print(non_zero_count_percent)

    return non_zero_count_percent


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type = str, required = True)
    parser.add_argument('--output_path', type = str, required = True)
    parser.add_argument('--dataset', type = str, required = True)
    args = parser.parse_args()

    return args


def compute_word_frequencies(text):
    tokens = WHITESPACE_REGEX.findall(text)
    token_len = len(tokens)
    most_common = Counter(tokens).most_common(2)
    if len(most_common) < 2:
        return None
    top1_word, top1_count = most_common[0]
    top2_word, top2_count = most_common[1]

    return top1_word, top1_count / token_len, top2_word, top2_count / token_len


if __name__ == '__main__':
    args = get_args()

    input_path = args.input_path
    output_path = args.output_path
    dataset = args.dataset

    print('configurations:')
    print("input path:", input_path)
    print("output path:", output_path)
    print("dataset:", dataset)

    if dataset == "DCLM":
        # Download the model from https://huggingface.co/mlfoundations/fasttext-oh-eli5 into utils/fasttext_dir/.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fasttext_model_path = os.path.join(
            repo_root,
            "utils/fasttext_dir/models--mlfoundations--fasttext-oh-eli5/snapshots/cd8b714a90f2dbcd3b02cf5fc972e5d7c7f4f107/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin",
        )
        fasttext_model = fasttext.load_model(fasttext_model_path)

    existing_texts = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding = "utf-8") as output_f:
            for line in output_f:
                data = json.loads(line)
                text = data['text']
                existing_texts.add(text)

    current_instance_index = 0
    uncontaminated_count = 0  
    with open(output_path, "a", encoding = "utf-8", buffering = 1) as output_f:
        with open(input_path, "r", encoding = "utf-8") as input_f:
            for line in input_f:
                print("current instance index:", current_instance_index)
                current_instance_index += 1

                data = json.loads(line)
                text = data["text"]

                if text in existing_texts:
                    print('this instance is already existed in the output file; skip it.')
                    continue

                # boundary data for DCLM-Baseline, StarCoder and Stack Exchange
                if dataset == "DCLM":
                    text_no_newlines = " ".join(text.strip().splitlines())
                    labels, probs = fasttext_model.predict(text_no_newlines, k = 2)
                    for lbl, p in zip(labels, probs):
                        if lbl == "__label__hq":
                            p_hq = float(p)
                    data["fasttext-oh-eli5_quality_score"] = p_hq
                    if p_hq < 0.018:
                        print("this instance's quality score is below 0.018; discard it.")
                        continue
                elif dataset == "StarCoder":
                    github_star = data["max_stars_count"]
                    if github_star < 2:
                        print("this instance's github star is less than 2; discard it.")
                        continue
                    word_freqs = compute_word_frequencies(text)
                    if word_freqs is None:
                        print("too few distinct tokens for word-frequency boundary; discard it.")
                        continue
                    data["top1_word"], data["top1_freq"], data["top2_word"], data["top2_freq"] = word_freqs
                    if not ((data["top1_freq"] <= 0.3 and 0.5 < (data["top1_freq"] + data["top2_freq"]) < 0.51) or (0.3 < data["top1_freq"] < 0.31 and (data["top1_freq"] + data["top2_freq"]) <= 0.5)):
                        print("top-1 and top-2 word frequencies are not at boundary; discard it.")
                        continue
                elif dataset == "Stack Exchange":
                    question_score = data["question_score"]
                    answer_score = data["answer_score"]
                    if not ((question_score == 2 and answer_score >= 5) or (question_score >= 3 and answer_score == 4)):
                        print("this instance's question & answer scores are not at boundary; discard it.")
                        continue

                start_time = time.time()

                is_uncontaminated = True
                all_models = ["1b", "7b", "13b", "32b"]
                for model in all_models:
                    overlap_score = get_counts(text, model)
                    print("model: {}, 13-gram overlap score: {}".format(model, overlap_score))
                    if overlap_score >= 0.2:
                        print("this instance is contaminated; discard it.")
                        is_uncontaminated = False
                        break
                    else:
                        data[f"{model}_13-gram_overlap_score"] = overlap_score

                end_time = time.time()
                print("execution time: ", end_time - start_time)

                if is_uncontaminated:
                    existing_texts.add(text)
                    uncontaminated_count += 1
                    print("this instance's all 13-gram overlap scores are uncontaminated; add it to the output file.")
                    output_f.write(json.dumps(data, ensure_ascii = False) + "\n")
        
    print("found {} number of uncontaminated instances out of {} total instances.".format(uncontaminated_count, current_instance_index))