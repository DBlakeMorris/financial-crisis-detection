import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel
from typing import Dict, Tuple
from .components.temporal import TemporalTransformer, CausalSelfAttention
from .components.bayesian import UncertaintyCalibration

class EnhancedFinancialModel(pl.LightningModule):
    def __init__(
        self,
        config: dict,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Text encoder with temporal processing
        self.text_encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.temporal_text = TemporalTransformer(
            d_model=768,  # DistilBERT hidden size
            nhead=8,
            num_layers=4
        )
        
        # Market data processing
        self.market_encoder = nn.Sequential(
            nn.Linear(config['market_dim'], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.market_temporal = CausalSelfAttention(
            d_model=256,
            num_heads=8
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(768 + 256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Uncertainty-aware prediction head
        self.uncertainty_head = UncertaintyCalibration(
            input_dim=512,
            output_dim=config['num_risk_levels']
        )
        
        self.learning_rate = learning_rate
        
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # Process text with temporal awareness
        text_output = self.text_encoder(
            batch['news_ids'],
            attention_mask=batch['masks']['news']
        )
        text_features = text_output.last_hidden_state
        temporal_text = self.temporal_text(text_features)
        
        # Global text representation
        text_features = temporal_text.mean(dim=1)  # Pool over sequence
        
        # Process market data with temporal awareness
        market_features = self.market_encoder(batch['market_data'])
        temporal_market = self.market_temporal(market_features.unsqueeze(1))
        market_features = temporal_market.squeeze(1)
        
        # Fusion
        combined = torch.cat([text_features, market_features], dim=-1)
        fused = self.fusion(combined)
        
        # Generate predictions with uncertainty
        outputs = self.uncertainty_head(fused)
        
        return outputs
        
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        outputs = self(batch)
        
        # Calculate losses
        prediction_loss = nn.CrossEntropyLoss()(
            outputs['prediction'],
            batch['risk_level']
        )
        
        # KL divergence loss for Bayesian components
        kl_loss = outputs['kl_div']
        
        # Uncertainty calibration
        uncertainty_loss = self._uncertainty_loss(
            outputs['total_uncertainty'],
            outputs['prediction'],
            batch['risk_level']
        )
        
        # Combined loss
        total_loss = (
            prediction_loss +
            0.1 * kl_loss +
            0.1 * uncertainty_loss
        )
        
        # Log metrics
        self.log_dict({
            'train_loss': total_loss,
            'prediction_loss': prediction_loss,
            'kl_loss': kl_loss,
            'uncertainty_loss': uncertainty_loss,
            'epistemic_uncertainty': outputs['epistemic_uncertainty'].mean(),
            'aleatoric_uncertainty': outputs['aleatoric_uncertainty'].mean()
        })
        
        return total_loss
        
    def _uncertainty_loss(
        self,
        uncertainty: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        # Calculate prediction error
        probs = torch.softmax(predictions, dim=-1)
        errors = torch.abs(
            torch.argmax(probs, dim=-1) -
            targets
        ).float()
        
        # Uncertainty should correlate with errors
        uncertainty = uncertainty.mean(dim=-1)
        return nn.MSELoss()(uncertainty, errors)
        
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        outputs = self(batch)
        
        # Calculate metrics
        predictions = torch.argmax(outputs['prediction'], dim=-1)
        accuracy = (predictions == batch['risk_level']).float().mean()
        
        # Log metrics
        self.log_dict({
            'val_accuracy': accuracy,
            'val_uncertainty': outputs['total_uncertainty'].mean()
        })
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy"
            }
        }
