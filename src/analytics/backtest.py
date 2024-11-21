import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class BacktestConfig:
    lookback_window: int = 60
    confidence_threshold: float = 0.8
    risk_free_rate: float = 0.02
    transaction_cost: float = 0.001

class BacktestFramework:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.positions = []
        self.returns = []
        self.metrics = {}
    
    def run_backtest(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Run backtest simulation"""
        portfolio_value = self._simulate_trading(
            predictions,
            confidences,
            market_data
        )
        
        self.metrics = self._calculate_metrics(portfolio_value)
        return self.metrics
    
    def _simulate_trading(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        market_data: pd.DataFrame
    ) -> np.ndarray:
        portfolio_value = np.zeros(len(predictions))
        portfolio_value[0] = 1.0  # Start with $1
        
        for t in range(1, len(predictions)):
            if confidences[t] > self.config.confidence_threshold:
                # High confidence prediction
                if predictions[t] >= 4:  # Crisis prediction
                    # Risk-off position
                    returns = -market_data['returns'].iloc[t]
                else:
                    # Risk-on position
                    returns = market_data['returns'].iloc[t]
                    
                # Apply transaction costs
                returns -= self.config.transaction_cost
                
                portfolio_value[t] = portfolio_value[t-1] * (1 + returns)
            else:
                portfolio_value[t] = portfolio_value[t-1]
        
        return portfolio_value
    
    def _calculate_metrics(
        self,
        portfolio_value: np.ndarray
    ) -> Dict[str, float]:
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        
        sharpe = self._calculate_sharpe_ratio(returns)
        sortino = self._calculate_sortino_ratio(returns)
        max_dd = self._calculate_max_drawdown(portfolio_value)
        
        return {
            'total_return': portfolio_value[-1] / portfolio_value[0] - 1,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'win_rate': np.mean(returns > 0)
        }
    
    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray
    ) -> float:
        excess_returns = returns - self.config.risk_free_rate/252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    
    def _calculate_sortino_ratio(
        self,
        returns: np.ndarray
    ) -> float:
        excess_returns = returns - self.config.risk_free_rate/252
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        return np.sqrt(252) * np.mean(excess_returns) / downside_std if downside_std != 0 else 0
    
    def _calculate_max_drawdown(
        self,
        portfolio_value: np.ndarray
    ) -> float:
        peak = portfolio_value[0]
        max_dd = 0
        
        for value in portfolio_value[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
    
    def plot_performance(
        self,
        portfolio_value: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray
    ):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot portfolio value
        ax1.plot(portfolio_value, label='Portfolio Value')
        ax1.set_title('Backtest Performance')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot predictions and confidence
        ax2.scatter(
            range(len(predictions)),
            predictions,
            c=confidences,
            cmap='viridis',
            alpha=0.6
        )
        ax2.set_title('Model Predictions with Confidence')
        ax2.set_ylabel('Risk Level')
        ax2.set_xlabel('Time')
        plt.colorbar(
            ax2.collections[0],
            ax=ax2,
            label='Confidence'
        )
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.close()
