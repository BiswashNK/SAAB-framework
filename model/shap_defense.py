import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple

class ShapDefenseValidator:
    """Uses SHAP analysis to validate predictions and detect adversarial samples"""
    
    def __init__(self, model_dir: str, 
                 confidence_threshold: float = 0.55,  # Lower threshold to be more sensitive
                 shap_deviation_threshold: float = 0.2):  # Lower threshold to catch more deviations
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        self.shap_deviation_threshold = shap_deviation_threshold
        self.correction_threshold = 0.7  # Slightly lower threshold for corrections
        self.original_weight = 0.3  # Give less weight to potentially adversarial predictions
        
        # Load SHAP baseline statistics
        self._load_shap_baseline()
        
        # Extract vocabulary from SHAP indices
        shap_indices = np.load(self.model_dir / "shap_indices.npy")
        self.vocab = [str(idx) for idx in range(max(shap_indices) + 1)]
        self.vocab_dict = {str(idx): idx for idx in range(len(self.vocab))}
    
    def _load_shap_baseline(self):
        """Load SHAP baseline statistics from analysis"""
        shap_dir = self.model_dir / "shap_analysis"
        
        # Load vulnerability report
        with open(shap_dir / "vulnerability_report.json", "r") as f:
            vuln_report = json.load(f)
        
        # Extract critical features and their baseline impacts
        self.critical_features = {
            item["feature"]: item["impact"]
            for item in vuln_report["top_vulnerable_features"]
        }
        
        # Load feature interaction patterns
        self.vulnerability_metrics = vuln_report["vulnerability_metrics"]
        
        # Calculate baseline statistics
        self.max_feature_impact = self.vulnerability_metrics["max_feature_contribution"]
        self.vulnerability_concentration = self.vulnerability_metrics["vulnerability_concentration"]
    
    def _analyze_feature_distribution(self, code: str) -> Dict[str, float]:
        """Analyze distribution of critical features in input"""
        # Tokenize input
        tokens = code.split()
        
        # Count occurrences of critical features
        feature_counts = {}
        for feature in self.critical_features:
            count = tokens.count(feature)
            if count > 0:
                feature_counts[feature] = count
        
        return feature_counts
    
    def _calculate_feature_impact_score(self, feature_counts: Dict[str, float]) -> float:
        """Calculate impact score based on critical feature distribution"""
        total_impact = 0.0
        for feature, count in feature_counts.items():
            baseline_impact = self.critical_features.get(feature, 0.0)
            total_impact += count * baseline_impact
        
        return total_impact
    
    def check_adversarial(self, code: str, prediction_prob: float) -> Tuple[bool, float, Dict]:
        """
        Check if a sample is likely adversarial based on SHAP patterns
        
        Args:
            code: Input code to check
            prediction_prob: Model's prediction probability
            
        Returns:
            is_adversarial: Whether the sample appears adversarial
            confidence: Confidence in the adversarial detection
            analysis: Detailed analysis of the decision
        """
        # Analyze feature distribution
        feature_counts = self._analyze_feature_distribution(code)
        feature_impact = self._calculate_feature_impact_score(feature_counts)
        
        # Check for suspicious patterns with confidence levels
        suspicious_patterns = []
        confidence_factors = []
        
        # 1. High concentration of critical features
        impact_ratio = feature_impact / self.max_feature_impact
        if impact_ratio > 1.5:
            suspicious_patterns.append("Very high concentration of critical features")
            confidence_factors.append(min(impact_ratio / 2.0, 1.0))
        elif impact_ratio > 1.2:
            suspicious_patterns.append("High concentration of critical features")
            confidence_factors.append(0.7)
        
        # 2. Missing expected critical features
        expected_features = set(self.critical_features.keys())
        found_features = set(feature_counts.keys())
        missing_critical = expected_features - found_features
        missing_ratio = len(missing_critical) / len(expected_features)
        if missing_ratio > 0.7:  # Missing >70% of expected features
            suspicious_patterns.append(
                "Missing many expected critical features"
            )
        
        # 3. Check prediction confidence vs feature distribution
        if prediction_prob > self.confidence_threshold and feature_impact < self.vulnerability_concentration * 0.5:
            suspicious_patterns.append(
                "High confidence despite unusual feature distribution"
            )
        
        # 4. Check for anomalous feature combinations
        if len(feature_counts) > 0:
            avg_impact_per_feature = feature_impact / len(feature_counts)
            if avg_impact_per_feature > self.max_feature_impact * 2:
                suspicious_patterns.append(
                    "Anomalous combination of critical features"
                )
        
            suspicious_patterns.append("Missing many expected critical features")
            confidence_factors.append(min(missing_ratio, 1.0))
            
        # 3. Check prediction confidence vs feature distribution
        feature_distr_ratio = feature_impact / (self.vulnerability_concentration * 0.5)
        if prediction_prob > self.confidence_threshold and feature_distr_ratio < 1.0:
            suspicious_patterns.append("High confidence despite unusual feature distribution")
            confidence_factors.append(1.0 - feature_distr_ratio)
            
        # 4. Check for anomalous feature combinations
        if len(feature_counts) > 0:
            avg_impact_per_feature = feature_impact / len(feature_counts)
            impact_anomaly = avg_impact_per_feature / (self.max_feature_impact * 2)
            if impact_anomaly > 1.0:
                suspicious_patterns.append("Anomalous combination of critical features")
                confidence_factors.append(min(impact_anomaly, 1.0))
                
        # Calculate detection confidence
        detection_confidence = 0.0
        if confidence_factors:
            detection_confidence = sum(confidence_factors) / len(confidence_factors)
            
        # Make final decision - require multiple patterns and sufficient confidence
        is_adversarial = len(suspicious_patterns) >= 2 and detection_confidence > self.shap_deviation_threshold
        
        analysis = {
            "is_adversarial": is_adversarial,
            "confidence": float(detection_confidence),
            "feature_impact_score": float(feature_impact),
            "suspicious_patterns": suspicious_patterns,
            "critical_features_found": list(feature_counts.keys()),
            "missing_critical_features": list(missing_critical),
            "confidence_factors": confidence_factors
        }
        
        return is_adversarial, detection_confidence, analysis

def validate_predictions(model_output_dir: str,
                       test_data_file: str,
                       output_file: str = None):
    """
    Validate model predictions using SHAP-aware defense
    
    Args:
        model_output_dir: Directory containing model and SHAP analysis
        test_data_file: File containing test samples
        output_file: Optional file to save detailed analysis
    """
    # Initialize defender
    defender = ShapDefenseValidator(model_output_dir)
    
    # Load predictions
    with open(Path(model_output_dir) / "predictions.json", "r") as f:
        predictions = json.load(f)
    
    # Load test data
    with open(test_data_file, "r") as f:
        test_data = []
        for line in f:
            sample = json.loads(line)
            if "target" in sample:  # Only include samples with ground truth
                test_data.append(sample)
    
    results = []
    detected_adversarial = 0
    total_samples = len(test_data)
    
    print(f"\nValidating {total_samples} samples using top SHAP features...")
    
    corrected_predictions = []
    for sample, pred in zip(test_data, predictions):
        # Check for adversarial patterns
        is_adversarial, confidence, analysis = defender.check_adversarial(
            sample["func"], 
            pred["prob"]
        )
        
        # Store original prediction
        result = {**analysis, "original_prediction": pred}
        
        # Apply correction if adversarial with high confidence
        if is_adversarial:
            detected_adversarial += 1
            if confidence > defender.correction_threshold:
                # Weighted combination of original and corrected prediction
                corrected_prob = defender.original_weight * pred["prob"] + (1 - defender.original_weight) * (1 - pred["prob"])
                pred["prob"] = corrected_prob
                pred["pred"] = int(corrected_prob > 0.5)
                result["corrected_prediction"] = pred
            else:
                result["correction_skipped"] = "Insufficient confidence"
        
        # Store result with ground truth target
        result = {
            "idx": sample["idx"],
            "original_prediction": {
                "prob": float(pred["prob"]),
                "pred": int(pred["pred"])
            },
            "adversarial_analysis": analysis,
            "ground_truth": int(sample["target"])
        }
        
        # Add corrected prediction if adversarial
        if is_adversarial and confidence > defender.correction_threshold:
            corrected_prob = defender.original_weight * pred["prob"] + (1 - defender.original_weight) * (1 - pred["prob"])
            result["corrected_prediction"] = {
                "prob": float(corrected_prob),
                "pred": int(corrected_prob > 0.5)
            }
        
        results.append(result)
    
    # Calculate accuracy metrics
    total_samples = len(results)
    original_correct = 0
    corrected_correct = 0
    total_corrected = 0
    
    for result in results:
        # Check original prediction accuracy
        if result["original_prediction"]["pred"] == result["ground_truth"]:
            original_correct += 1
            
        # Check corrected prediction accuracy if available
        if result["adversarial_analysis"]["is_adversarial"]:
            if "corrected_prediction" in result:
                total_corrected += 1
                if result["corrected_prediction"]["pred"] == result["ground_truth"]:
                    corrected_correct += 1
    
    original_accuracy = original_correct / total_samples
    corrected_accuracy = corrected_correct / total_corrected if total_corrected > 0 else 0
    final_accuracy = (original_correct + corrected_correct) / total_samples
    
    # Print summary
    print("\nSHAP Defense Analysis:")
    print(f"Total samples analyzed: {total_samples}")
    print(f"Detected adversarial samples: {detected_adversarial}")
    print(f"Adversarial detection rate: {detected_adversarial/total_samples:.2%}")
    print(f"Original model accuracy: {original_accuracy:.2%}")
    print(f"Final accuracy with defense: {final_accuracy:.2%}")
    if total_corrected > 0:
        print(f"Accuracy of corrected predictions: {corrected_accuracy:.2%}")
        print(f"Number of predictions corrected: {total_corrected}")
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "total_samples": total_samples,
                    "detected_adversarial": detected_adversarial,
                    "detection_rate": detected_adversarial/total_samples
                },
                "detailed_results": results
            }, f, indent=2)
        print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="runs/saab_bilstm_svm_asterisk_full",
                       help="Directory containing model and SHAP analysis")
    parser.add_argument("--test_file", required=True,
                       help="File containing test samples")
    parser.add_argument("--output_file", default=None,
                       help="Optional file to save detailed analysis")
    
    args = parser.parse_args()
    
    validate_predictions(
        args.model_dir,
        args.test_file,
        args.output_file
    )