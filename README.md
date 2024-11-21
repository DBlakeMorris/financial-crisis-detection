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

## Quick Start
```bash
# Install
git clone https://github.com/yourusername/financial-crisis-detection.git
cd financial-crisis-detection
pip install -r requirements.txt

# Run training and evaluation
python run_experiment.py
