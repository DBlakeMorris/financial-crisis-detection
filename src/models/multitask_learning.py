import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class MultitaskHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_configs: Dict[str, Dict]
    ):
        super().__init__()
        self.task_configs = task_configs
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: self._create_head(config)
            for task, config in task_configs.items()
        })
        
        # Task weights for loss balancing
        self.task_weights = nn.Parameter(
            torch.ones(len(task_configs))
        )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all task heads"""
        shared_features = self.shared_layer(x)
        
        return {
            task: head(shared_features)
            for task, head in self.task_heads.items()
        }
        
    def _create_head(
        self,
        config: Dict
    ) -> nn.Module:
        """Create task-specific head"""
        if config['type'] == 'classification':
            return nn.Linear(
                self.shared_layer[-3].out_features,
                config['num_classes']
            )
        elif config['type'] == 'regression':
            return nn.Sequential(
                nn.Linear(
                    self.shared_layer[-3].out_features,
                    1
                ),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown task type: {config['type']}")
            
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted losses for all tasks"""
        task_losses = {}
        weighted_losses = []
        
        for i, (task, output) in enumerate(outputs.items()):
            if task not in targets:
                continue
                
            loss = self._compute_task_loss(
                output,
                targets[task],
                self.task_configs[task]['type']
            )
            
            task_losses[task] = loss.item()
            weighted_losses.append(
                self.task_weights[i] * loss
            )
            
        total_loss = sum(weighted_losses)
        
        return total_loss, task_losses
        
    def _compute_task_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        task_type: str
    ) -> torch.Tensor:
        """Compute loss for specific task"""
        if task_type == 'classification':
            return nn.CrossEntropyLoss()(output, target)
        elif task_type == 'regression':
            return nn.MSELoss()(output, target)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
