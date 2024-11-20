import pytest
from src.data.datamodule import FinancialDataModule

def test_datamodule():
    # Create dummy config
    class Config:
        data = type('', (), {
            'max_seq_length': 512,
            'max_documents': 1000,
            'num_workers': 0
        })
        training = type('', (), {
            'batch_size': 32
        })
    
    config = Config()
    
    datamodule = FinancialDataModule(config)
    datamodule.setup()
    
    # Test batch shape
    batch = next(iter(datamodule.train_dataloader()))
    assert batch['news_ids'].dim() == 2
    assert batch['market_data'].dim() == 2
