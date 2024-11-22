import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

class DataPreprocessor:
    def __init__(self, tokenizer_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
    def preprocess_text(self, texts):
        """
        Tokenize and prepare text data
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
    def preprocess_market_data(self, market_data):
        """
        Normalize and prepare market indicators
        """
        # Add normalization logic here
        return torch.tensor(market_data, dtype=torch.float32)