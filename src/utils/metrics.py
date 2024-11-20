import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from typing import Dict, Tuple

def calculate_financial_metrics(predictions: torch.Tensor, 
                              targets: torch.Tensor,
                              market_data: torch.Tensor) -> Dict[str, float]:
    """Calculate finance-specific metrics"""
    # Convert to numpy
    preds = predictions.cpu().numpy()
    true = targets.cpu().numpy()
    
    # Basic metrics
    auroc = roc_auc_score(true, preds, multi_class='ovr')
    
    # Finance-specific metrics
    precision, recall, thresholds = precision_recall_curve(true, preds)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    # Calculate trading metrics
    trading_metrics = calculate_trading_metrics(preds, true, market_data)
    
    return {
        'auroc': auroc,
        'optimal_threshold': optimal_threshold,
        **trading_metrics
    }

def calculate_trading_metrics(predictions: np.ndarray,
                            targets: np.ndarray,
                            market_data: np.ndarray) -> Dict[str, float]:
    """Calculate trading-specific metrics"""
    # Sharp ratio
    returns = calculate_returns(predictions, market_data)
    sharpe = calculate_sharpe_ratio(returns)
    
    # Maximum drawdown
    max_dd = calculate_max_drawdown(returns)
    
    return {
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'returns': returns.mean()
    }
