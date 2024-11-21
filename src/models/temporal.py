import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, Tuple

class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.pos_encoder(x)
        return self.transformer_encoder(x, src_key_padding_mask=mask)

class OrderBookEncoder(nn.Module):
    def __init__(
        self,
        num_levels: int = 10,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.price_encoder = nn.Linear(num_levels, hidden_dim)
        self.volume_encoder = nn.Linear(num_levels, hidden_dim)
        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=hidden_dim * 2
        )
        
    def forward(
        self,
        prices: torch.Tensor,  # [batch, time, levels]
        volumes: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        price_features = self.price_encoder(prices)
        volume_features = self.volume_encoder(volumes)
        features = torch.cat([price_features, volume_features], dim=-1)
        return self.temporal_encoder(features, mask)

class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # Prior distributions
        self.weight_prior = torch.distributions.Normal(0, prior_std)
        self.bias_prior = torch.distributions.Normal(0, prior_std)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        
        out = torch.nn.functional.linear(x, weight, bias)
        
        # Calculate KL divergence
        kl_weight = self._kl_divergence(
            self.weight_mu,
            weight_sigma,
            self.weight_prior
        )
        kl_bias = self._kl_divergence(
            self.bias_mu,
            bias_sigma,
            self.bias_prior
        )
        
        return out, kl_weight + kl_bias
    
    def _kl_divergence(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        prior: torch.distributions.Distribution
    ) -> torch.Tensor:
        q = torch.distributions.Normal(mu, sigma)
        return torch.distributions.kl_divergence(q, prior).sum()

class CausalAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        intervention: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        q = rearrange(
            self.q_proj(query),
            'b n (h d) -> b h n d',
            h=self.num_heads
        )
        k = rearrange(
            self.k_proj(key),
            'b n (h d) -> b h n d',
            h=self.num_heads
        )
        v = rearrange(
            self.v_proj(value),
            'b n (h d) -> b h n d',
            h=self.num_heads
        )
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
        if intervention is not None:
            # Apply causal intervention
            attn_weights = attn_weights * intervention.unsqueeze(1)
            
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, 'b h n d -> b n (h d)')
        
        out = self.out_proj(attn_output)
        return out, attn_weights

class UncertaintyCalibration(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int
    ):
        super().__init__()
        self.epistemic_head = BayesianLinear(hidden_dim, num_classes)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes * 2)  # mean and variance
        )
        
    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Epistemic uncertainty through Bayesian sampling
        epistemic_preds = []
        total_kl = 0
        for _ in range(num_samples):
            pred, kl = self.epistemic_head(x)
            epistemic_preds.append(pred)
            total_kl += kl
            
        epistemic_mean = torch.stack(epistemic_preds).mean(0)
        epistemic_var = torch.stack(epistemic_preds).var(0)
        
        # Aleatoric uncertainty
        aleatoric_params = self.aleatoric_head(x)
        aleatoric_mean = aleatoric_params[..., :aleatoric_params.size(-1)//2]
        aleatoric_var = torch.exp(
            aleatoric_params[..., aleatoric_params.size(-1)//2:]
        )
        
        # Combined prediction and uncertainties
        pred = (epistemic_mean + aleatoric_mean) / 2
        total_uncertainty = epistemic_var + aleatoric_var
        
        return pred, total_uncertainty, total_kl / num_samples
