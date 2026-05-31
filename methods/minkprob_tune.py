"""
Tune MinK and MinK++'s score_type on the dev split.
"""

import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from data_utils import load_dataset
from model_utils import load_model_bundle
from minkprob_method import compute_mink_scores

def run_tuning_experiment(contam_path, uncontam_path, model_name):
    # --- 1. Load Model Bundle ---
    print(f"--> Loading model bundle: {model_name}...")
    model_bundle = load_model_bundle(model_name) 

    # --- 2. Load Datasets ---
    print(f"--> Loading Datasets...")
    contam_dataset = load_dataset(contam_path)
    uncontam_dataset = load_dataset(uncontam_path)

    # --- 3. Grid Search over K ---
    candidate_ks = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    print(f"\n{'k':>5} | {'Min-K% AUC':>12} | {'Min-K%++ AUC':>14}")
    print("-" * 40)

    for k in candidate_ks:
        # contam_output = compute_mink_scores(model_bundle, contam_dataset, k_percent=k, batch_size=1)
        # uncontam_output = compute_mink_scores(model_bundle, uncontam_dataset, k_percent=k, batch_size=1)

        contam_output = compute_mink_scores(model_bundle, contam_dataset, k_mink=k, k_minkpp=k, batch_size=1)
        uncontam_output = compute_mink_scores(model_bundle, uncontam_dataset, k_mink=k, k_minkpp=k, batch_size=1)
        
        # Extract raw scores
        c_mink_raw = [s["min_k_prob_mean_logprob"] for s in contam_output["samples"]]
        c_minkpp_raw = [s["min_k_pp_score"] for s in contam_output["samples"]]
        
        u_mink_raw = [s["min_k_prob_mean_logprob"] for s in uncontam_output["samples"]]
        u_minkpp_raw = [s["min_k_pp_score"] for s in uncontam_output["samples"]]
        
        # --- Numerical Stability Fix ---
        # Convert to numpy arrays and handle inf/nan values
        # We use a large finite number (1e6) to represent infinity so AUC can still rank them
        c_mink = np.nan_to_num(np.array(c_mink_raw), nan=0.0, posinf=1e6, neginf=-1e6)
        u_mink = np.nan_to_num(np.array(u_mink_raw), nan=0.0, posinf=1e6, neginf=-1e6)
        
        c_minkpp = np.nan_to_num(np.array(c_minkpp_raw), nan=0.0, posinf=1e6, neginf=-1e6)
        u_minkpp = np.nan_to_num(np.array(u_minkpp_raw), nan=0.0, posinf=1e6, neginf=-1e6)

        # Binary Labels: 1 for Contaminated, 0 for Uncontaminated
        y_true = [1] * len(c_mink) + [0] * len(u_mink)
        
        # Combine scores for AUC-ROC
        scores_mink = np.concatenate([c_mink, u_mink])
        scores_minkpp = np.concatenate([c_minkpp, u_minkpp])
        
        # Calculate AUC-ROC
        auc_mink = roc_auc_score(y_true, scores_mink)
        auc_minkpp = roc_auc_score(y_true, scores_minkpp)
        
        results.append({"k": k, "mink": auc_mink, "minkpp": auc_minkpp})
        print(f"{k:5.2f} | {auc_mink:12.4f} | {auc_minkpp:14.4f}")

    # --- 4. Identify Best Results ---
    best_mink = max(results, key=lambda x: x["mink"])
    best_minkpp = max(results, key=lambda x: x["minkpp"])

    print("\n" + "="*45)
    print(f"MODEL: {os.path.basename(model_name)}")
    print(f"Optimal Min-K%:   k={best_mink['k']} (AUC: {best_mink['mink']:.4f})")
    print(f"Optimal Min-K%++: k={best_minkpp['k']} (AUC: {best_minkpp['minkpp']:.4f})")
    print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune K for Min-K methods with batch size 1.")
    parser.add_argument("--contam_path", type=str, required=True)
    parser.add_argument("--uncontam_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    
    run_tuning_experiment(
        contam_path=args.contam_path, 
        uncontam_path=args.uncontam_path, 
        model_name=args.model_name
    )

