import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> tuple:
        features = self.encoder(x)
        uncertainty = self.uncertainty_estimator(features)
        return features, uncertainty

class EpistemicUncertaintyHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> tuple:
        x = self.dropout(x)
        logits = self.classifier(x)
        uncertainty = torch.norm(x, dim=-1, keepdim=True)
        return logits, uncertainty
