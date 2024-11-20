import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)
from pytorch_lightning.loggers import WandbLogger

from src.models.lite_model import FinancialCrisisLite
from src.data.lite_datamodule import FinancialDataModuleLite

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(config: DictConfig):
    # Create model and data
    model = FinancialCrisisLite(config)
    datamodule = FinancialDataModuleLite(config)
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-model-{epoch:02d}-{train_loss:.3f}',
        monitor='train_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    
    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor='train_loss',
            patience=5,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=WandbLogger(project='financial-crisis-lite'),
        enable_progress_bar=True,
        log_every_n_steps=1,
        default_root_dir='checkpoints'
    )
    
    # Train model
    trainer.fit(model, datamodule)
    
    # Print path to best model checkpoint
    print(f"\nBest model checkpoint: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    train()
