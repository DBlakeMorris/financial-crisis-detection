import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import scipy.stats as stats
import json
import logging

@dataclass
class ExperimentConfig:
    name: str
    variant_names: List[str]
    metrics: List[str]
    min_samples: int = 1000
    significance_level: float = 0.05

class ABTest:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data = {variant: [] for variant in config.variant_names}
        self.results = {}
        self.logger = logging.getLogger(__name__)
        
    def record_observation(
        self,
        variant: str,
        metrics: Dict[str, float]
    ):
        """Record observation for a variant"""
        if variant not in self.data:
            raise ValueError(f"Unknown variant: {variant}")
            
        self.data[variant].append({
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
    def analyze(self) -> Dict[str, Dict]:
        """Analyze experiment results"""
        if not self._enough_samples():
            self.logger.warning("Not enough samples for analysis")
            return {}
            
        results = {}
        for metric in self.config.metrics:
            metric_results = self._analyze_metric(metric)
            results[metric] = metric_results
            
        self.results = results
        return results
    
    def _analyze_metric(
        self,
        metric: str
    ) -> Dict[str, float]:
        """Analyze single metric"""
        control_data = self._get_metric_data(
            self.config.variant_names[0],
            metric
        )
        
        results = {}
        for variant in self.config.variant_names[1:]:
            variant_data = self._get_metric_data(variant, metric)
            
            # Statistical tests
            t_stat, p_value = stats.ttest_ind(
                control_data,
                variant_data
            )
            
            # Effect size
            effect_size = (
                np.mean(variant_data) -
                np.mean(control_data)
            ) / np.std(control_data)
            
            results[variant] = {
                'mean_difference': np.mean(variant_data) - np.mean(control_data),
                'percent_change': (
                    (np.mean(variant_data) - np.mean(control_data)) /
                    np.mean(control_data) * 100
                ),
                'p_value': p_value,
                'significant': p_value < self.config.significance_level,
                'effect_size': effect_size
            }
            
        return results
    
    def _get_metric_data(
        self,
        variant: str,
        metric: str
    ) -> np.ndarray:
        """Extract metric data for variant"""
        return np.array([
            obs['metrics'][metric]
            for obs in self.data[variant]
            if metric in obs['metrics']
        ])
        
    def _enough_samples(self) -> bool:
        """Check if enough samples are collected"""
        return all(
            len(data) >= self.config.min_samples
            for data in self.data.values()
        )
    
    def get_summary(self) -> str:
        """Generate human-readable summary"""
        if not self.results:
            return "No results available yet"
            
        summary = [f"Experiment: {self.config.name}\n"]
        
        for metric, results in self.results.items():
            summary.append(f"\nMetric: {metric}")
            for variant, stats in results.items():
                summary.append(f"\n{variant}:")
                summary.append(
                    f"  Change: {stats['percent_change']:.2f}%"
                )
                summary.append(
                    f"  Significant: {stats['significant']}"
                    f" (p={stats['p_value']:.4f})"
                )
                summary.append(
                    f"  Effect size: {stats['effect_size']:.2f}"
                )
                
        return "\n".join(summary)
    
    def save_results(self, path: str):
        """Save results to file"""
        with open(path, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
