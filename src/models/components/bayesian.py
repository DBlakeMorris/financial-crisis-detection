import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import math

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
        
        # Initialize parameters
        self.reset_parameters()
        
        # Prior distributions
        self.prior_std = prior_std
        self.kl_div = 0.0
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)
        nn.init.constant_(self.bias_mu, 0.0)
        nn.init.constant_(self.bias_rho, -3)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sample weights and biases
        weight = self._sample_weights()
        bias = self._sample_bias()
        
        # Compute output
        out = F.linear(x, weight, bias)
        
        return out, self.kl_div
        
    def _sample_weights(self) -> torch.Tensor:
        epsilon = torch.randn_like(self.weight_mu)
        sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + sigma * epsilon
        
        # Compute KL divergence
        self.kl_div = self._compute_kl(self.weight_mu, sigma)
        
        return weight
        
    def _sample_bias(self) -> torch.Tensor:
        epsilon = torch.randn_like(self.bias_mu)
        sigma = torch.log1p(torch.exp(self.bias_rho))
        bias = self.bias_mu + sigma * epsilon
        
        # Add to KL divergence
        self.kl_div += self._compute_kl(self.bias_mu, sigma)
        
        return bias
        
    def _compute_kl(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        kl = 0.5 * torch.sum(
            (sigma / self.prior_std) ** 2 +
            (mu / self.prior_std) ** 2 -
            1 - 2 * torch.log(sigma / self.prior_std)
        )
        return kl

class BayesianNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BayesianLinear(
                input_dim if i == 0 else hidden_dim,
                hidden_dim if i < num_layers - 1 else output_dim
            )
            for i in range(num_layers)
        ])
        
    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        kl_divs = []
        
        for _ in range(num_samples):
            current = x
            kl_total = 0.0
            
            for layer in self.layers:
                current, kl = layer(current)
                kl_total += kl
                
                if layer != self.layers[-1]:
                    current = F.relu(current)
                    
            outputs.append(current)
            kl_divs.append(kl_total)
            
        # Stack results
        outputs = torch.stack(outputs)
        kl_divs = torch.stack(kl_divs)
        
        # Compute mean and uncertainty
        mean = outputs.mean(dim=0)
        uncertainty = outputs.var(dim=0)
        
        return {
            'mean': mean,
            'uncertainty': uncertainty,
            'kl_div': kl_divs.mean()
        }

class AleatoricUncertaintyHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.var = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean = self.mean(x)
        log_var = self.var(x)
        
        return {
            'mean': mean,
            'var': torch.exp(log_var),
            'log_var': log_var
        }

class UncertaintyCalibration(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.epistemic = BayesianNetwork(input_dim, input_dim // 2, output_dim)
        self.aleatoric = AleatoricUncertaintyHead(input_dim, output_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        num_samples: int = 10
    ) -> Dict[str, torch.Tensor]:
        # Get epistemic uncertainty
        epistemic_out = self.epistemic(x, num_samples)
        
        # Get aleatoric uncertainty
        aleatoric_out = self.aleatoric(x)
        
        # Combine uncertainties
        total_uncertainty = (
            epistemic_out['uncertainty'] +
            aleatoric_out['var']
        )
        
        return {
            'prediction': (
                epistemic_out['mean'] +
                aleatoric_out['mean']
            ) / 2,
            'epistemic_uncertainty': epistemic_out['uncertainty'],
            'aleatoric_uncertainty': aleatoric_out['var'],
            'total_uncertainty': total_uncertainty,
            'kl_div': epistemic_out['kl_div']
        }
