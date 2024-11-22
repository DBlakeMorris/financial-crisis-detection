import torch
from torch.utils.data import Dataset
import pandas as pd

class FinancialCrisisDataset(Dataset):
    def __init__(self, text_data, market_data, labels):
        self.text_data = text_data
        self.market_data = market_data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            "text": self.text_data[idx],
            "market": self.market_data[idx],
            "label": self.labels[idx]
        }