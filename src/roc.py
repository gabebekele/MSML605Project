import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

def save_roc_plot(sweep_results, run_id, outputs_dir):
    
    # 1. Calculate TPR and FPR
    # TPR is 'recall'. FPR = FP / (FP + TN)
    fpr = []
    recall = []
    
    for m in sweep_results:
        # Calculate FPR (handle division by zero if all negatives are missing)
        fp_tn = m['fp'] + m['tn']
        fpr_val = m['fp'] / fp_tn if fp_tn > 0 else 0
        
        fpr.append(fpr_val)
        recall.append(m['recall'])

    # create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, recall, color='orange', lw=2, label='ROC curve')

    # plot diagonal for comparison with random guessing
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR / Recall)')
    plt.title(f'ROC Curve: {run_id}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # save plot under the run
    plot_dir = os.path.join(outputs_dir, "runs", run_id)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "roc_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"--- ROC Plot successfully saved to: {plot_path} ---")