import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from torchmetrics import Accuracy, AUROC, F1Score
import torch.nn.functional as F

class FinancialCrisisLite(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.train_acc = Accuracy(task='multiclass', num_classes=config.model.num_risk_levels)
        self.val_acc = Accuracy(task='multiclass', num_classes=config.model.num_risk_levels)
        self.val_auroc = AUROC(task='multiclass', num_classes=config.model.num_risk_levels)
        self.val_f1 = F1Score(task='multiclass', num_classes=config.model.num_risk_levels)
        self.test_acc = Accuracy(task='multiclass', num_classes=config.model.num_risk_levels)
        self.test_auroc = AUROC(task='multiclass', num_classes=config.model.num_risk_levels)
        self.test_f1 = F1Score(task='multiclass', num_classes=config.model.num_risk_levels)
        
        self.text_encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.hidden_dim = 768
        
        self.market_encoder = nn.Sequential(
            nn.BatchNorm1d(config.model.market_dim),
            nn.Linear(config.model.market_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        combined_dim = self.hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(combined_dim),
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, config.model.num_risk_levels)
        )
        
        self.lr = config.training.learning_rate
        self.save_hyperparameters()
    
    def forward(self, batch):
        # Ensure all inputs are on the same device
        device = next(self.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        text_output = self.text_encoder(
            batch['news_ids'],
            attention_mask=batch['masks']['news']
        )
        text_features = text_output.last_hidden_state.mean(dim=1)
        market_features = self.market_encoder(batch['market_data'])
        combined = torch.cat([text_features, market_features], dim=-1)
        logits = self.fusion(combined)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch['risk_level'])
        self.train_acc(logits.softmax(dim=-1), batch['risk_level'])
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
