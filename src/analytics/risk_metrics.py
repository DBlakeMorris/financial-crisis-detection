import numpy as np
from typing import Dict, List, Tuple
from scipy import stats

class RiskMetrics:
    def __init__(self):
        self.metrics_history = []
        self.confidence_intervals = {}
    
    def calculate_risk_metrics(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        confidences: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        metrics = {}
        
        # Classification metrics
        metrics.update(self._calculate_classification_metrics(
            predictions,
            true_values
        ))
        
        # Confidence calibration
        metrics.update(self._calculate_calibration_metrics(
            predictions,
            true_values,
            confidences
        ))
        
        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(
            predictions,
            true_values,
            confidences
        ))
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_classification_metrics(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification performance metrics"""
        accuracy = np.mean(predictions == true_values)
        
        # Calculate per-class metrics
        class_metrics = {}
        for cls in np.unique(true_values):
            mask = true_values == cls
            if np.sum(mask) > 0:
                precision = np.mean(
                    predictions[predictions == cls] == true_values[predictions == cls]
                )
                recall = np.sum(
                    (predictions == cls) & (true_values == cls)
                ) / np.sum(mask)
                f1 = 2 * (precision * recall) / (precision + recall)
                
                class_metrics.update({
                    f'precision_class_{cls}': precision,
                    f'recall_class_{cls}': recall,
                    f'f1_class_{cls}': f1
                })
        
        return {
            'accuracy': accuracy,
            **class_metrics
        }
    
    def _calculate_calibration_metrics(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        confidences: np.ndarray
    ) -> Dict[str, float]:
        """Calculate confidence calibration metrics"""
        # Expected Calibration Error
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if np.sum(in_bin) > 0:
                accuracy_in_bin = np.mean(
                    predictions[in_bin] == true_values[in_bin]
                )
                confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - confidence_in_bin) * (
                    np.sum(in_bin) / len(predictions)
                )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            predictions,
            true_values,
            confidences
        )
        
        return {
            'expected_calibration_error': ece,
            'confidence_intervals': confidence_intervals
        }
    
    def _calculate_risk_adjusted_metrics(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        confidences: np.ndarray
    ) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        # Weight errors by confidence
        weighted_errors = np.abs(
            predictions - true_values
        ) * (1 - confidences)
        
        # Calculate metrics
        risk_adjusted_accuracy = 1 / (1 + np.mean(weighted_errors))
        
        # Cost-sensitive metrics
        cost_matrix = self._get_cost_matrix(len(np.unique(true_values)))
        cost_weighted_errors = np.array([
            cost_matrix[int(true), int(pred)]
            for true, pred in zip(true_values, predictions)
        ])
        
        expected_cost = np.mean(cost_weighted_errors)
        
        return {
            'risk_adjusted_accuracy': risk_adjusted_accuracy,
            'expected_cost': expected_cost
        }
    
    def _calculate_confidence_intervals(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        confidences: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics"""
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        accuracies = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(
                len(predictions),
                size=len(predictions)
            )
            acc = np.mean(
                predictions[idx] == true_values[idx]
            )
            accuracies.append(acc)
            
        ci_lower, ci_upper = np.percentile(
            accuracies,
            [alpha * 100, (1 - alpha) * 100]
        )
        
        return {
            'accuracy_ci': (ci_lower, ci_upper)
        }
    
    def _get_cost_matrix(self, n_classes: int) -> np.ndarray:
        """Create cost matrix for different types of errors"""
        cost_matrix = np.zeros((n_classes, n_classes))
        
        # Penalize more severely for larger prediction errors
        for i in range(n_classes):
            for j in range(n_classes):
                cost_matrix[i, j] = abs(i - j)
                
        # Extra penalty for missing crisis predictions
        cost_matrix[-1, :] *= 2
        
        return cost_matrix
