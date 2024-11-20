# Financial Crisis Early Warning System

## Overview
Avdeep learning system that predicts financial market crisis risks using multi-modal analysis of textual data and market indicators. The system employs advanced NLP techniques and market analysis to provide early warnings of potential financial crises.

## Key Features
- Multi-modal deep learning architecture combining text and market data
- Advanced NLP using DistilBERT for financial text analysis
- Real-time risk level prediction (5 levels from stable to crisis)
- Uncertainty estimation for reliable risk assessment
- Achieves 77% accuracy on synthetic test data
- Particularly strong at detecting severe crisis signals (90% F1-score for highest risk)

## Technical Highlights
- PyTorch Lightning implementation
- Multi-modal fusion architecture
- Uncertainty-aware predictions
- Wandb integration for experiment tracking
- Distributed training support
- MPS/GPU acceleration support

## Model Performance
- Overall Accuracy: 77%
- Weighted Average F1-Score: 0.75
- Crisis Detection (Highest Risk) F1-Score: 0.90
- Balanced performance across risk levels

## Installation
```bash
git clone https://github.com/yourusername/financial-crisis-detection.git
cd financial-crisis-detection
pip install -r requirements.txt
