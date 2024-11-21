import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger
import torch
from src.models.enhanced_model import EnhancedFinancialModel
from src.data.lite_datamodule import FinancialDataModuleLite

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(config: DictConfig):
    pl.seed_everything(42)
    
    # Initialize model
    model = EnhancedFinancialModel(
        config=config.model,
        learning_rate=config.training.learning_rate
    )
    
    # Initialize data module
    datamodule = FinancialDataModuleLite(config)
    
    # Initialize callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_accuracy',
            mode='max',
            save_top_k=3,
            filename='enhanced-model-{epoch:02d}-{val_accuracy:.3f}'
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=config.training.early_stopping_patience,
            mode='max'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Initialize trainer with safe config access
    trainer_kwargs = {
        'max_epochs': config.training.max_epochs,
        'accelerator': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'devices': 1,
        'callbacks': callbacks,
        'logger': WandbLogger(project=config.wandb.project),
        'gradient_clip_val': config.training.gradient_clip_val
    }
    
    # Safely add accumulate_grad_batches if it exists
    if hasattr(config.training, 'accumulate_grad_batches'):
        trainer_kwargs['accumulate_grad_batches'] = config.training.accumulate_grad_batches
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train model
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train()
