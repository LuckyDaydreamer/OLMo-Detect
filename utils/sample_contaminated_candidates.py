import json
import argparse
import os
import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    input_path = args.input_path
    output_path = args.output_path
    dataset = args.dataset

    input_files = sorted(glob.glob(os.path.join(args.input_path, "**", "*.jsonl"), recursive=True))
    unique_texts = set()

    with open(output_path, "a", encoding = "utf-8", buffering = 1) as output_f:
        for infile in input_files:
            with open(infile, "r", encoding = "utf-8") as input_f:
                for line in input_f:
                    data = json.loads(line)
                    text = data["text"]

                    if dataset == "DCLM":
                        # DCLM-Baseline is already the top-quality subset (all scores at
                        # or above 0.01811), so this band keeps instances right at the
                        # bottom edge — immediately above OLMo 2's filter threshold.
                        text_quality_score = data["fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob"]
                        if text_quality_score < 0.0181122:
                            print("text quality score is {}, add it to the output.".format(text_quality_score))
                            if text not in unique_texts:
                                unique_texts.add(text)
                                output_f.write(json.dumps(data, ensure_ascii = False) + "\n")
                    elif dataset == "StarCoder":
                        top1_freq = data["attributes"]["top_20_tokens__top_20_tokens__p1"][0][-1]
                        top2_freq = data["attributes"]["top_20_tokens__top_20_tokens__p2"][0][-1]
                        if top1_freq >= 0.28 or (top1_freq + top2_freq) >= 0.48:
                            print("top1_freq is {}, top1_freq + top2_freq is {},  add it to the output.".format(top1_freq, top1_freq + top2_freq))
                            if text not in unique_texts:
                                unique_texts.add(text)
                                output_f.write(json.dumps(data, ensure_ascii = False) + "\n")
                    elif dataset == "Stack Exchange":
                        question_score = data["metadata"]["question_score"]
                        answer_score = data["metadata"]["answer_score"]
                        if question_score == 3 or answer_score == 5:
                            print("question vote is {}, answer vote is {}, add it to the output.".format(question_score, answer_score))
                            if text not in unique_texts:
                                unique_texts.add(text)
                                output_f.write(json.dumps(data, ensure_ascii = False) + "\n")
                    else:
                        raise ValueError("wrong dataset")