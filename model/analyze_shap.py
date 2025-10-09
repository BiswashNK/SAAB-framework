import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import shap

def load_shap_data(model_dir):
    """Load SHAP values and indices from model directory"""
    model_dir = Path(model_dir)
    shap_values = np.load(model_dir / "shap_values.npy")
    shap_indices = np.load(model_dir / "shap_indices.npy")
    return shap_values, shap_indices

def load_vocab(model_dir):
    """Load vocabulary from model directory"""
    model_dir = Path(model_dir)
    with open(model_dir / "vocab.txt", "r") as f:
        vocab = [line.strip() for line in f]
    return vocab

def analyze_shap_values(shap_values, vocab, indices, top_k=20):
    """Analyze SHAP values to identify most influential features"""
    # Average absolute SHAP values across samples
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Get indices of top k features by absolute SHAP value
    top_indices = np.argsort(-mean_abs_shap)[:top_k]
    
    # Get corresponding vocab terms and SHAP values
    top_terms = [vocab[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]
    
    return top_terms, top_values

def plot_top_features(terms, values, output_path):
    """Plot top features by SHAP value"""
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(terms))
    
    # Create horizontal bar plot
    plt.barh(y_pos, values)
    plt.yticks(y_pos, terms)
    
    plt.title("Top Features by SHAP Value Impact")
    plt.xlabel("Mean Absolute SHAP Value")
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(v, i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_feature_interactions(shap_values, vocab, indices, top_k=10):
    """Analyze interactions between top features"""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(-mean_abs_shap)[:top_k]
    
    # Calculate correlation between SHAP values for top features
    top_shap_values = shap_values[:, top_indices]
    correlation = np.corrcoef(top_shap_values.T)
    
    return [vocab[i] for i in top_indices], correlation

def plot_feature_interactions(terms, correlation, output_path):
    """Plot feature interaction heatmap"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, fmt=".2f", 
                xticklabels=terms, yticklabels=terms, 
                cmap="coolwarm", center=0)
    
    plt.title("Feature Interaction Analysis")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Directory with model and SHAP data")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir).resolve()  # Get absolute path
    output_dir = model_dir / "shap_analysis"
    output_dir.mkdir(exist_ok=True)
    
    shap_values, indices = load_shap_data(model_dir)
    vocab = load_vocab(model_dir)
    
    # Create output directory
    output_dir = Path(model_dir) / "shap_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Analyze top features
    print("Analyzing top features by SHAP value...")
    top_terms, top_values = analyze_shap_values(shap_values, vocab, indices)
    plot_top_features(top_terms, top_values, output_dir / "top_features.png")
    
    # Save top features to JSON
    with open(output_dir / "top_features.json", "w") as f:
        json.dump({
            "features": top_terms,
            "shap_values": top_values.tolist()
        }, f, indent=2)
    
    # Analyze feature interactions
    print("\nAnalyzing feature interactions...")
    interaction_terms, correlation = analyze_feature_interactions(shap_values, vocab, indices)
    plot_feature_interactions(interaction_terms, correlation, output_dir / "feature_interactions.png")
    
    # Print vulnerability analysis
    print("\nVulnerability Analysis:")
    print("----------------------")
    print("Most influential features that could be targeted by adversarial attacks:")
    for term, value in zip(top_terms[:10], top_values[:10]):
        print(f"- {term}: {value:.4f}")
    
    # Calculate vulnerability scores
    total_shap = np.abs(shap_values).sum()
    feature_contributions = np.abs(shap_values).sum(axis=0) / total_shap
    vulnerable_threshold = np.percentile(feature_contributions, 90)
    
    print("\nVulnerability Statistics:")
    print(f"- Number of high-impact features (90th percentile): {sum(feature_contributions > vulnerable_threshold)}")
    print(f"- Maximum feature contribution: {feature_contributions.max():.4f}")
    print(f"- Vulnerability concentration (top 10% features): {(feature_contributions[feature_contributions > vulnerable_threshold].sum()):.4f}")
    
    # Ensure shap_analysis directory exists
    shap_dir = output_dir / "shap_analysis"
    shap_dir.mkdir(exist_ok=True)
    
    # Create vulnerability report file
    vuln_report_path = output_dir / "vulnerability_report.json"
    
    # Save vulnerability report
    report = {
        "vulnerability_metrics": {
            "high_impact_features": int(sum(feature_contributions > vulnerable_threshold)),
            "max_feature_contribution": float(feature_contributions.max()),
            "vulnerability_concentration": float(feature_contributions[feature_contributions > vulnerable_threshold].sum())
        },
        "top_vulnerable_features": [
            {"feature": term, "impact": float(value)}
            for term, value in zip(top_terms[:10], top_values[:10])
        ]
    }
    
    with open(vuln_report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()