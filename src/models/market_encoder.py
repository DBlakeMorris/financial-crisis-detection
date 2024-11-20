import torch
import torch.nn as nn
from .attention import MultiHeadHierarchicalAttention

class TemporalMarketEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        self.temporal_attention = MultiHeadHierarchicalAttention(hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
    def forward(self, market_data: torch.Tensor, mask: torch.Tensor = None):
        # Project features
        features = self.feature_projection(market_data)
        
        # Apply temporal attention
        attended = self.temporal_attention(features, features, features, mask)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(attended.unsqueeze(1))
        
        return lstm_out.squeeze(1)
