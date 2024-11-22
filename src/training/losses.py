import torch
import torch.nn as nn
import torch.nn.functional as F

class RiskPredictionLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, predictions, targets):
        """
        Multi-class classification loss with optional class weighting
        """
        return F.cross_entropy(predictions, targets, weight=self.class_weights)