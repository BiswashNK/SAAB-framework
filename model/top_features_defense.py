import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple

class TopFeaturesShapDefender:
    """SHAP defense based on pre-identified important features from training"""
    
    def __init__(self, model_dir: str,
                 confidence_threshold: float = 0.7,  # Lower threshold to be more sensitive
                 feature_impact_threshold: float = 1.2,  # Lower threshold to catch more anomalies 
                 missing_features_threshold: float = 0.4):  # Lower threshold for missing features
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        self.feature_impact_threshold = feature_impact_threshold
        self.missing_features_threshold = missing_features_threshold
        
        # Load top features and their SHAP values
        self._load_top_features()
    
    def _load_top_features(self):
        """Load pre-computed top features and their SHAP values"""
        shap_dir = self.model_dir / "shap_analysis"
        with open(shap_dir / "top_features.json", "r") as f:
            data = json.load(f)
            
        self.top_features = data["features"]
        self.feature_importance = dict(zip(data["features"], data["shap_values"]))
        
        # Calculate baseline statistics
        self.max_feature_impact = max(data["shap_values"])
        self.mean_feature_impact = np.mean(data["shap_values"])
        self.total_importance = sum(data["shap_values"])
    
    def check_adversarial(self, code: str, prediction_prob: float) -> Tuple[bool, Dict]:
        """
        Check if a sample is likely adversarial by analyzing its use of important features
        
        Args:
            code: Input code to check
            prediction_prob: Model's prediction probability
        
        Returns:
            is_adversarial: Whether the sample appears adversarial
            analysis: Detailed analysis of the decision
        """
        # Split code into tokens
        tokens = code.split()
        
        # Analyze presence/absence of top features
        feature_counts = {}
        total_feature_importance = 0.0
        max_feature_importance = 0.0
        
        # Track both present and absent feature impacts
        for feature in self.top_features:
            count = tokens.count(feature)
            importance = self.feature_importance[feature]
            if count > 0:
                feature_counts[feature] = count
                total_feature_importance += importance * count
                max_feature_importance = max(max_feature_importance, importance * count)
        
        # Enhanced detection criteria with confidence scores
        suspicious_patterns = []
        pattern_scores = []
        
        # 1. Check for missing highly important features with weighted scoring
        present_features = set(feature_counts.keys())
        top_20_features = set(self.top_features[:20])  # Expanded feature set
        missing_important = top_20_features - present_features
        
        # Weight missing features by their importance
        missing_importance = sum(self.feature_importance[f] for f in missing_important)
        total_importance = sum(self.feature_importance[f] for f in top_20_features)
        missing_score = missing_importance / total_importance if total_importance > 0 else 0
        
        # Track code characteristics for vulnerability assessment
        has_vuln_indicators = False
        vuln_score = 0.0
        
        if missing_score > 0.4:
            suspicious_patterns.append(
                f"Missing critical features with {missing_score:.2%} total importance"
            )
            pattern_scores.append(min(1.0, missing_score * 1.5))
            
            # Check if missing features indicate vulnerability
            vuln_indicators = set(['buffer', 'malloc', 'strcpy', 'memcpy', 'free'])
            if any(f in vuln_indicators for f in missing_important):
                has_vuln_indicators = True
                vuln_score += 0.4
        
        # 2. Check feature importance distribution
        if total_feature_importance > 0:
            # Compare against expected distribution
            expected_importance = self.mean_feature_impact * len(feature_counts)
            importance_ratio = total_feature_importance / expected_importance
            if importance_ratio > 2.0 or importance_ratio < 0.3:
                suspicious_patterns.append(
                    f"Abnormal feature importance distribution (ratio: {importance_ratio:.2f})"
                )
                pattern_scores.append(min(abs(importance_ratio - 1.0), 1.0))
                
                # Check if feature distribution indicates vulnerability
                if importance_ratio < 0.3 and any(f in feature_counts for f in ['buffer', 'malloc', 'strcpy']):
                    has_vuln_indicators = True
                    vuln_score += 0.3
                    
        # 3. Check for prediction confidence anomalies with improved scoring
        confidence_delta = abs(0.5 - prediction_prob)  # Distance from decision boundary
        found_top_features = len(present_features.intersection(top_20_features))
        expected_confidence = found_top_features / len(top_20_features)
        
        # Calculate confidence mismatch score
        confidence_mismatch = abs(confidence_delta - expected_confidence)
        if confidence_mismatch > 0.3:  # More sensitive threshold
            suspicious_patterns.append(
                f"Confidence ({prediction_prob:.2f}) significantly mismatches feature presence"
            )
            pattern_scores.append(min(1.0, confidence_mismatch * 2.0))  # Stronger confidence signal
            
        # Add pattern score for overall feature importance
        if total_feature_importance > 0:
            importance_ratio = total_feature_importance / (self.mean_feature_impact * len(present_features))
            pattern_scores.append(min(1.0, abs(1.0 - importance_ratio)))
            pattern_scores.append(abs(confidence_delta - expected_confidence))
        
        # 4. Check for feature concentration
        if len(feature_counts) > 0:
            concentration = max_feature_importance / total_feature_importance
            if concentration > 0.5:  # Single feature dominates
                suspicious_patterns.append(
                    f"Suspicious feature concentration: {concentration:.2f}"
                )
                pattern_scores.append(concentration)
        
        # Make final decision with confidence weighting
        is_adversarial = False
        confidence_score = 0.0
        if len(suspicious_patterns) >= 2:  # Require multiple indicators
            # Calculate weighted confidence score
            confidence_score = sum(pattern_scores) / len(pattern_scores)
            is_adversarial = confidence_score > 0.6  # Higher threshold for detection
            
        # Calculate vulnerability likelihood
        vuln_likelihood = min(1.0, vuln_score + (0.3 if has_vuln_indicators else 0))
        
        # Update analysis with more detailed metrics
        analysis = {
            "is_adversarial": is_adversarial,
            "confidence": float(prediction_prob),
            "feature_impact_score": float(total_feature_importance),
            "feature_concentration": float(max_feature_importance / total_feature_importance) if total_feature_importance > 0 else 0.0,
            "missing_features_ratio": float(missing_score),
            "confidence_score": float(confidence_score),
            "suspicious_patterns": suspicious_patterns,
            "pattern_scores": pattern_scores,
            "features_found": {k: v for k, v in feature_counts.items()},
            "missing_top_features": list(missing_important) if len(missing_important) > 0 else [],
            "vulnerability_score": float(vuln_likelihood),
            "has_vulnerability_indicators": has_vuln_indicators
        }
        
        return is_adversarial, analysis

def validate_with_top_features(model_dir: str,
                             test_data_file: str,
                             output_file: str = None):
    """
    Validate predictions using pre-computed top SHAP features with enhanced defense
    
    Args:
        model_dir: Directory containing model and SHAP analysis
        test_data_file: File containing test samples
        output_file: Optional file to save detailed analysis
    """
    # Initialize defender with enhanced detection
    defender = TopFeaturesShapDefender(
        model_dir,
        confidence_threshold=0.6,  # More sensitive detection
        feature_impact_threshold=1.1,  # Lower threshold for anomalies
        missing_features_threshold=0.35  # More sensitive to missing features
    )
    
    # Load predictions 
    with open(Path(model_dir) / "predictions.json", "r") as f:
        raw_predictions = json.load(f)
        predictions = {}
        if isinstance(raw_predictions, list):
            for i, pred in enumerate(raw_predictions):
                if isinstance(pred, (int, float)):
                    predictions[str(i)] = {
                        "prediction": int(round(pred)),
                        "probability": abs(0.5 - pred) + 0.5  # Estimate confidence
                    }
                elif isinstance(pred, dict):
                    # If it already has prediction/probability fields
                    if "prediction" in pred and "probability" in pred:
                        predictions[str(i)] = pred
                    else:
                        # Try to extract values from the dict
                        pred_val = next(iter(pred.values()))
                        predictions[str(i)] = {
                            "prediction": int(round(float(pred_val))),
                            "probability": abs(0.5 - float(pred_val)) + 0.5
                        }
        else:
            # If it's already a dict with the right format
            predictions = raw_predictions
    
    # Load test data and align with predictions
    test_data = []
    test_indices = []
    with open(test_data_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Extract code from AST representation
                code = data["func"]
                idx = data["idx"]
                test_indices.append(idx)
                test_data.append({
                    "code": code,
                    "func": code,
                    "idx": idx,
                    "target": data["target"]
                })
    
    # Create predictions aligned with test data
    aligned_predictions = {}
    for i, idx in enumerate(test_indices):
        if str(idx) in predictions:
            aligned_predictions[str(i)] = predictions[str(idx)]
        else:
            # Default prediction if missing
            aligned_predictions[str(i)] = {
                "prediction": 0,
                "probability": 0.5
            }
    predictions = aligned_predictions
    
    results = []
    detected_adversarial = 0
    corrected_predictions = 0
    total_samples = len(test_data)
    
    print(f"\nValidating {total_samples} samples using top SHAP features with vulnerability assessment...")
    
    for i, sample in enumerate(test_data):
        code = sample['code']
        original_pred = predictions[str(i)]['prediction']
        pred_prob = predictions[str(i)]['probability']
        
        # Run enhanced defense check
        is_adversarial, analysis = defender.check_adversarial(code, pred_prob)
        
        # Correct prediction based on vulnerability indicators and confidence
        final_pred = original_pred
        if is_adversarial:
            detected_adversarial += 1
            
            # Calculate correction confidence based on multiple factors
            vuln_confidence = analysis['vulnerability_score']
            pattern_confidence = analysis['confidence_score']
            missing_features_impact = analysis['missing_features_ratio']
            
            # Weighted decision based on all factors
            correction_weight = (vuln_confidence * 0.4 + 
                              pattern_confidence * 0.4 + 
                              missing_features_impact * 0.2)
            
            if correction_weight > 0.6:  # High confidence in need for correction
                if vuln_confidence > 0.5:  # Strong vulnerability indicators
                    final_pred = 1  # More likely to be vulnerable
                    corrected_predictions += 1
                elif pattern_confidence > 0.8:  # Very strong adversarial patterns
                    final_pred = 1 - original_pred  # Flip prediction
                    corrected_predictions += 1
            
        # Record detailed results
        result = {
            'idx': test_data[i]['idx'],
            'code': test_data[i]['code'],
            'is_adversarial': is_adversarial,
            'original_prediction': int(original_pred),
            'final_prediction': int(final_pred),
            'pred_probability': float(pred_prob),
            'analysis': {
                k: float(v) if isinstance(v, (np.float32, np.float64, np.bool_)) else v
                for k, v in analysis.items()
            }
        }
        results.append(result)
    
    detected_adversarial = 0
    corrected_predictions = 0
    
    detected_adversarial = 0
    corrected_predictions = 0
    results = []
    
    for i, sample in enumerate(test_data):
        code = sample['code']
        original_pred = predictions[str(i)]['prediction']
        pred_prob = predictions[str(i)]['probability']
        
        # Run enhanced defense check
        is_adversarial, analysis = defender.check_adversarial(code, pred_prob)
        
        # Update statistics
        if is_adversarial:
            detected_adversarial += 1
            
        # Get confidence score for prediction adjustment
        confidence_score = analysis.get("confidence_score", 0.0)

        if is_adversarial:
            detected_adversarial += 1
            
            # Factor in multiple metrics for a more balanced decision
            missing_ratio = analysis.get("missing_features_ratio", 0.0)
            feature_concentration = analysis.get("feature_concentration", 0.0)
            impact_score = analysis.get("feature_impact_score", 0.0)
            pattern_scores = analysis.get("pattern_scores", [])
            avg_pattern_score = sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0.0
            
            # Calculate final prediction with all factors
        final_pred = original_pred
        if is_adversarial:
            # Calculate defense metrics with weighted components
            pattern_weight = 0.4
            feature_weight = 0.3
            concentration_weight = 0.2
            missing_weight = 0.1
            
            # Enhanced defense strength calculation with non-linear scaling
            base_strength = (
                pattern_weight * min(1.0, avg_pattern_score * 1.5) +    # Stronger pattern influence
                feature_weight * min(1.0, (impact_score / 5.0)) +       # More aggressive impact normalization
                concentration_weight * (feature_concentration ** 0.8) +  # Non-linear concentration scaling
                missing_weight * missing_ratio
            )
            
            # Apply sigmoid-like scaling to defense strength
            defense_strength = 1.0 / (1.0 + np.exp(-6 * (base_strength - 0.5)))
            # Boost with confidence
            defense_strength = defense_strength * (1.0 + (confidence_score * 0.5))
            
            # Calculate prediction confidence based on multiple factors
            if defense_strength > 0.6:  # Lower threshold for high confidence
                if original_pred == 1:
                    # Already predicted vulnerable - strengthen proportionally
                    defense_confidence = min(0.98, pred_prob * (1.0 + (defense_strength * 0.4)))
                    revised_pred = {"prob": float(defense_confidence), "pred": int(1)}
                else:
                    # Predicted non-vulnerable - more aggressive correction
                    base_confidence = 0.75  # Higher base confidence
                    defense_weight = min(0.9, defense_strength * 1.2)  # Boost defense weight
                    original_weight = max(0.1, 1.0 - defense_weight)
                    defense_confidence = (base_confidence * defense_weight) + (pred_prob * original_weight)
                    revised_pred = {"prob": float(defense_confidence), "pred": int(1)}
            elif defense_strength > 0.5:  # Medium confidence
                # Use weighted average of original and corrected predictions
                defense_weight = (defense_strength - 0.5) * 2  # Scale to 0-1
                original_weight = 1.0 - defense_weight
                if original_pred == 1:
                    # Already vulnerable - minor adjustment
                    defense_confidence = pred_prob * (1.0 + (defense_strength * 0.1))
        # Set initial prediction
        final_pred = original_pred
        
        # Calculate final prediction based on vulnerability and pattern analysis
        if is_adversarial:
            vuln_score = analysis.get("vulnerability_score", 0.0)
            has_vuln = analysis.get("has_vulnerability_indicators", False)
            pattern_confidence = analysis.get("confidence_score", 0.0)
            
            if vuln_score > 0.6 or has_vuln:
                # High confidence in vulnerability
                final_pred = 1
                corrected_predictions += 1
            elif pattern_confidence > 0.8:
                # Strong adversarial patterns - likely wrong
                final_pred = 1 - original_pred
                corrected_predictions += 1
            # else keep original prediction
        
        # Store result
        result = {
            'idx': test_data[i]['idx'],
            'code': test_data[i]['code'],
            'is_adversarial': is_adversarial,
            'original_prediction': int(original_pred),
            'final_prediction': int(final_pred),
            'pred_probability': float(pred_prob),
            'analysis': {
                k: float(v) if isinstance(v, (np.float32, np.float64, np.bool_)) else v
                for k, v in analysis.items()
            }
        }
        results.append(result)
        
        # Update predictions
        predictions[str(i)] = {
            'prediction': int(final_pred),
            'probability': float(pred_prob)
        }
    
    # Print summary
    print("\nSHAP Top Features Defense Analysis:")
    print(f"Total samples analyzed: {total_samples}")
    print(f"Detected adversarial samples: {detected_adversarial}")
    print(f"Adversarial detection rate: {detected_adversarial/total_samples:.2%}")
    
    # Print summary with more detail
    print("\nSHAP Top Features Defense Analysis:")
    print(f"Total samples analyzed: {total_samples}")
    print(f"Detected adversarial: {detected_adversarial}")
    print(f"Corrected predictions: {corrected_predictions}")
    print(f"Detection rate: {(detected_adversarial/total_samples)*100:.2f}%")
    print(f"Correction rate: {(corrected_predictions/total_samples)*100:.2f}%")
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "summary": {
                    "total_samples": total_samples,
                    "detected_adversarial": detected_adversarial,
                    "corrected_predictions": corrected_predictions,
                    "detection_rate": detected_adversarial/total_samples,
                    "correction_rate": corrected_predictions/total_samples
                },
                "detailed_results": results
            }, f, indent=2)
        print(f"\nDetailed analysis saved to: {output_file}")
        
        # Save corrected predictions
        pred_file = str(Path(output_file).parent / "predictions.json")
        with open(pred_file, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Corrected predictions saved to: {pred_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True,
                       help="Directory containing model and SHAP analysis")
    parser.add_argument("--test_file", required=True,
                       help="File containing test samples")
    parser.add_argument("--output_file", default=None,
                       help="Optional file to save detailed analysis")
    
    args = parser.parse_args()
    
    validate_with_top_features(
        args.model_dir,
        args.test_file,
        args.output_file
    )