import os
import sys
import csv
import yaml
import json
import argparse
import numpy as np
from src.roc import save_roc_plot
import shutil
from pathlib import Path

# make sure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from m1.similarity import cosine_similarity
from src.embedding import embed_pair
from src.thresholding import (
    apply_threshold, compute_confusion_matrix, compute_accuracy, 
    compute_precision, compute_recall, compute_f1_score, compute_balanced_accuracy
)
from src.tracking import build_run_record, save_run

def load_balanced_pairs(config, split_name, sample_per_class=200):
    pairs_path = config["pairs_paths"][split_name]
    
    all_rows = []
    with open(pairs_path, "r") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    positives = [r for r in all_rows if int(r['label']) == 1]
    negatives = [r for r in all_rows if int(r['label']) == 0]
    
    n = min(sample_per_class, len(positives), len(negatives))
    pair_rows = positives[:n] + negatives[:n]
    
    print(f"Loaded {split_name}: {len(pair_rows)} total pairs ({n} per class).")
    return pair_rows

def compute_similarity_scores(pair_rows, config, norm_type=None):
    left_vectors, right_vectors, true_labels = [], [], []

    for i, row in enumerate(pair_rows):
        lv, rv = embed_pair(row["left_path"], row["right_path"], config, normalization=norm_type)
        left_vectors.append(lv)
        right_vectors.append(rv)
        true_labels.append(int(row["label"]))

    scores = cosine_similarity(np.array(left_vectors), np.array(right_vectors))
    return scores, np.array(true_labels)

def compute_metrics(scores, true_labels, threshold):
    preds = apply_threshold(scores, threshold, higher_is_same=True)
    conf = compute_confusion_matrix(true_labels, preds)
    return {
        "threshold": float(threshold),
        "accuracy": float(compute_accuracy(conf)),
        "precision": float(compute_precision(conf)),
        "recall": float(compute_recall(conf)),
        "f1": float(compute_f1_score(conf)),
        "balanced_accuracy": float(compute_balanced_accuracy(conf)),
        "tp": int(conf["tp"]), "tn": int(conf["tn"]), "fp": int(conf["fp"]), "fn": int(conf["fn"])
    }

def error_exs(pair_rows, scores, labels, threshold, output_dir, run_id, config, limit=5):
    error_dir = Path(output_dir) / "runs" / run_id / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)

    preds = (scores >= threshold)
    base_data_path = Path(config["data_dir"])
    
    fp_count, fn_count = 0, 0
    
    for i in range(len(labels)):
        is_fp = (preds[i] == 1 and labels[i] == 0)
        is_fn = (preds[i] == 0 and labels[i] == 1)
        
        if (is_fp and fp_count < limit) or (is_fn and fn_count < limit):
            label_type = "FP" if is_fp else "FN"
            pair_idx = fp_count if is_fp else fn_count
            pair_folder = error_dir / f"{label_type}_{pair_idx}"
            pair_folder.mkdir(parents=True, exist_ok=True)
            
            for side in ["left_path", "right_path"]:
                
                rel_path = pair_rows[i][side].replace("\\", "/")
                
                src_path = base_data_path / rel_path
                
                if src_path.exists():
                    shutil.copy(src_path, pair_folder)
                else:
                    print(f"Warning: Could not find image at {src_path}")
            
            if is_fp: fp_count += 1
            else: fn_count += 1

def run_evaluation_pipeline(config, run_id, norm_type, metric_rule, limit):
    print(f"\n--- Starting {run_id} | Norm: {norm_type} | Metric: {metric_rule} ---")

    # val
    val_rows = load_balanced_pairs(config, "val", limit)
    val_scores, val_labels = compute_similarity_scores(val_rows, config, norm_type)
    
    thresholds = np.linspace(-1.0, 1.0, 50)
    sweep = [compute_metrics(val_scores, val_labels, t) for t in thresholds]
    
    # plot roc curve
    save_roc_plot(sweep, run_id, config['outputs_dir'])
    # ---------------------------

    best_val = max(sweep, key=lambda x: x[metric_rule])
    selected_t = best_val['threshold']
    

    # test
    test_rows = load_balanced_pairs(config, "test", limit)
    test_scores, test_labels = compute_similarity_scores(test_rows, config, norm_type)
    
    # save errors
    error_exs(test_rows, test_scores, test_labels, selected_t, config['outputs_dir'], run_id, config)

    test_metrics = compute_metrics(test_scores, test_labels, selected_t)

    # save summary & record
    summary = {"run_id": run_id, "selected_threshold": selected_t, "test_metrics": test_metrics}
    summary_path = os.path.join(config["outputs_dir"], "eval", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    record = build_run_record(
        run_id=run_id, split_name="test", pair_version="baseline",
        threshold_info={"rule": metric_rule, "value": selected_t},
        metrics=test_metrics, note=f"Norm: {norm_type}",
        parameters={"norm": norm_type, "limit": limit}
    )
    save_run(config['outputs_dir'], run_id, record)
    
    print(f"Run {run_id} Complete. Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1.yaml") 
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--norm", choices=["z-score", "min-max"], default=None)
    parser.add_argument("--metric", default="balanced_accuracy")
    parser.add_argument("--limit", type=int, default=200)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_evaluation_pipeline(config, args.run_id, args.norm, args.metric, args.limit)

if __name__ == "__main__":
    main()