import json
import argparse
from pathlib import Path
import numpy as np

def load_json(path):
    with open(path) as f:
        return json.load(f)

def calculate_metrics(y_true, y_pred):
    y = np.array(y_true)
    p = np.array(y_pred)
    
    # Calculate accuracy
    accuracy = (y == p).mean()
    
    # Calculate ASR for vulnerable samples
    vuln_mask = (y == 1)
    if vuln_mask.sum() > 0:
        asr = ((vuln_mask) & (p == 0)).sum() / vuln_mask.sum()
    else:
        asr = 0.0
    
    return accuracy, asr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--original_file", required=True)
    parser.add_argument("--defended_file", required=True)
    parser.add_argument("--defense_analysis", required=True)
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    # Load predictions
    original_preds = load_json(model_dir / args.original_file)
    defended_preds = load_json(model_dir / args.defended_file)
    defense_analysis = load_json(model_dir / args.defense_analysis)
    
    # Extract predictions and true labels
    y_true = [p["true_label"] for p in original_preds if "true_label" in p]
    y_orig = [p["pred"] for p in original_preds]
    y_def = [p["pred"] for p in defended_preds]
    
    # Calculate metrics
    orig_acc, orig_asr = calculate_metrics(y_true, y_orig)
    def_acc, def_asr = calculate_metrics(y_true, y_def)
    
    # Print comparison
    print("\nDefense Effectiveness Analysis")
    print("----------------------------------------")
    print(f"Total samples analyzed: {len(y_true)}")
    print(f"Detected adversarial samples: {defense_analysis['summary']['detected_adversarial']}")
    print(f"Adversarial detection rate: {defense_analysis['summary']['detection_rate']:.2%}")
    print("\nModel Performance Comparison:")
    print(f"{'Metric':<25} {'Original':>10} {'With Defense':>15} {'Change':>10}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {orig_acc:>10.2%} {def_acc:>15.2%} {def_acc-orig_acc:>+10.2%}")
    print(f"{'Attack Success Rate':<25} {orig_asr:>10.2%} {def_asr:>15.2%} {def_asr-orig_asr:>+10.2%}")
    
    # Analyze defense decisions
    changed_predictions = sum(o != d for o, d in zip(y_orig, y_def))
    print(f"\nDefense Impact:")
    print(f"Changed predictions: {changed_predictions} ({changed_predictions/len(y_true):.2%} of samples)")
    
    # Save comparison results
    comparison = {
        "metrics": {
            "original": {
                "accuracy": float(orig_acc),
                "attack_success_rate": float(orig_asr)
            },
            "with_defense": {
                "accuracy": float(def_acc),
                "attack_success_rate": float(def_asr)
            }
        },
        "defense_summary": defense_analysis["summary"],
        "changes": {
            "accuracy_change": float(def_acc - orig_acc),
            "asr_change": float(def_asr - orig_asr),
            "predictions_changed": changed_predictions,
            "change_rate": float(changed_predictions/len(y_true))
        }
    }
    
    with open(model_dir / "defense_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nDetailed comparison saved to {model_dir}/defense_comparison.json")

if __name__ == "__main__":
    main()