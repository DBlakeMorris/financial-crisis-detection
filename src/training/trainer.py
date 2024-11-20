import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.datamodule import FinancialDataModule
from src.models.document_encoder import DocumentHierarchyEncoder
from src.models.entity_graph import FinancialEntityGraph
from src.models.market_encoder import TemporalMarketEncoder

class MarketRiskPredictor(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize components
        self.document_encoder = DocumentHierarchyEncoder(config.model)
        self.entity_graph = FinancialEntityGraph(config.model)
        self.market_encoder = TemporalMarketEncoder(
            input_dim=config.model.market_dim,
            hidden_dim=config.model.hidden_dim
        )
        
        # Final prediction layers
        self.fusion = nn.Linear(config.model.hidden_dim * 3, config.model.hidden_dim)
        self.predictor = nn.Linear(config.model.hidden_dim, config.model.num_risk_levels)
        
    def forward(self, batch):
        # Get features from each component
        doc_features = self.document_encoder(batch)
        entity_features = self.entity_graph(
            batch['mention_features'],
            batch['mention_locations']
        )
        market_features = self.market_encoder(
            batch['market_data'],
            batch['market_mask']
        )
        
        # Combine features
        combined = torch.cat([
            doc_features,
            entity_features,
            market_features
        ], dim=-1)
        
        # Make prediction
        fused = self.fusion(combined)
        predictions = self.predictor(fused)
        
        return predictions
        
    def training_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = nn.CrossEntropyLoss()(predictions, batch['risk_level'])
        self.log('train_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate
        )
        return optimizer

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(config: DictConfig):
    # Set up data
    datamodule = FinancialDataModule(config)
    
    # Create model
    model = MarketRiskPredictor(config)
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        )
    ]
    
    # Initialize trainer with MPS settings
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator='mps' if torch.backends.mps.is_available() else 'cpu',
        devices=1,  # Use single device for MPS
        callbacks=callbacks,
        logger=WandbLogger()
    )
    
    # Train model
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    train()
