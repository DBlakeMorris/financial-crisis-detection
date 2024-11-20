import pytest
import torch
from src.models.document_encoder import DocumentHierarchyEncoder
from src.models.entity_graph import FinancialEntityGraph

def test_document_encoder():
    batch_size = 4
    seq_length = 512
    hidden_dim = 768
    
    # Create dummy config
    class Config:
        hidden_dim = hidden_dim
        num_heads = 8
    
    config = Config()
    
    # Create dummy batch
    batch = {
        'news_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'masks': {
            'news': torch.ones(batch_size, seq_length),
            'filing': torch.ones(batch_size, seq_length),
            'social': torch.ones(batch_size, seq_length)
        },
        'hierarchy_markers': {
            'sentence': torch.ones(batch_size, seq_length),
            'paragraph': torch.ones(batch_size, seq_length),
            'document': torch.ones(batch_size, seq_length)
        }
    }
    
    model = DocumentHierarchyEncoder(config)
    output = model(batch)
    
    assert output.shape == (batch_size, seq_length, hidden_dim)
