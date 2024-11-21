# Financial Crisis Early Warning System

## Overview
A deep learning system that predicts financial market crisis risk levels by analyzing synthetic financial news text and market indicators. The model demonstrates the potential for multi-modal deep learning in financial risk detection.

## Potential Application Scope
### Text Data Sources:
- Financial News Articles
- Company Performance Reports
- Market Commentary
- Economic Indicators Analysis
- Trading Pattern Reports

### Market Data Sources:
- Market Performance Metrics
- Volatility Indicators
- Trading Volume
- Market Momentum Signals
- Stability Measures

The system processes these inputs through a neural architecture to classify financial risk into 5 distinct levels, from stable conditions to crisis scenarios.

## Current Implementation
### Data Analysis
#### Text Features:
- Basic sentiment patterns in financial news
- Risk-indicative language processing
- Simple market condition descriptions
- Synthetic news text classification
- Risk level associated terminology

#### Market Features:
- Risk-correlated synthetic indicators
- Noise-adjusted market metrics
- Simulated market stability measures
- Basic volatility patterns
- Risk level aligned data points

### Implementation Details
#### Text Data (Synthetic)
- Financial news snippets with risk-indicative language
- Examples: "Company reports strong performance with positive market conditions" or "Markets indicate bearish trends as investors show negative sentiment"
- Processed using DistilBERT for text embedding

#### Market Data (Synthetic)
- Generated market indicators with controlled noise levels
- Features correlate with risk levels
- Normalized market metrics with variable volatility

## Model Architecture
- **Text Processing**: DistilBERT encoder (66.4M parameters)
- **Market Processing**: Custom neural network (25.4K parameters)
- **Fusion Layer**: Combined feature processing (1.2M parameters)
- **Output**: 5-level risk classification (0: Stable to 4: Crisis)

## Performance
Latest test results show:
- Overall Accuracy: 77%
- Per-Risk Level Performance:
  * Level 0 (Stable): 74% F1-score
  * Level 1 (Low Risk): 76% F1-score
  * Level 2 (Medium Risk): 82% F1-score
  * Level 3 (High Risk): 38% F1-score
  * Level 4 (Crisis): 90% F1-score

## Technical Stack
- PyTorch Lightning
- HuggingFace Transformers
- Weights & Biases for experiment tracking
- MPS/GPU acceleration support

## Implemented Advanced Features

### Real-time Processing
- Streaming architecture for market data and news
- Asynchronous processing pipeline
- Performance monitoring and latency tracking
- Buffer management for data streams

### Uncertainty & Temporal Analysis
- Temporal transformers for sequential data
- Bayesian neural networks for uncertainty estimation
- Causal attention mechanisms
- Uncertainty calibration

### Production Infrastructure
- Experiment tracking with W&B
- Configurable architecture (Hydra)
- Performance monitoring
- MPS/GPU acceleration support

## Planned Future Enhancements

### Advanced Analytics
- Order book modeling
- High-frequency data processing
- Cross-document relationship analysis
- Entity-relationship graphs
- Market microstructure analysis

### Production Features
- Kafka/Redis integration
- Online learning capabilities
- Drift detection and monitoring
- Auto-retraining triggers
- Performance degradation alerts

### Explainability & Robustness
- SHAP/LIME integration
- Attention visualization
- Counterfactual explanations
- Stress testing framework
- Market regime change analysis

## Quick Start
```bash
# Install
git clone https://github.com/DBlakeMorris/financial-crisis-detection.git
cd financial-crisis-detection
pip install -r requirements.txt

# Run training and evaluation
python run_experiment.py
