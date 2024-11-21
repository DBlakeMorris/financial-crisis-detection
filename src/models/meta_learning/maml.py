import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy

class MAML(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        
    def adapt(
        self,
        support_data: Dict[str, torch.Tensor],
        num_steps: int = None
    ) -> nn.Module:
        """Adapt model to new task using support data"""
        if num_steps is None:
            num_steps = self.num_inner_steps
            
        adapted_model = deepcopy(self.model)
        adapted_params = adapted_model.parameters()
        
        for _ in range(num_steps):
            # Forward pass
            outputs = adapted_model(support_data)
            loss = self._compute_loss(outputs, support_data)
            
            # Inner loop update
            grads = torch.autograd.grad(loss, adapted_params)
            adapted_params = [
                p - self.inner_lr * g 
                for p, g in zip(adapted_params, grads)
            ]
            
            # Update model parameters
            for param, new_param in zip(
                adapted_model.parameters(),
                adapted_params
            ):
                param.data = new_param.data
                
        return adapted_model
    
    def meta_train(
        self,
        tasks: List[Dict[str, torch.Tensor]]
    ) -> float:
        """Perform meta-training step"""
        meta_loss = 0
        
        for task in tasks:
            support_data = task['support']
            query_data = task['query']
            
            # Adapt to task
            adapted_model = self.adapt(support_data)
            
            # Evaluate on query set
            outputs = adapted_model(query_data)
            task_loss = self._compute_loss(outputs, query_data)
            meta_loss += task_loss
            
        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item() / len(tasks)
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute task-specific loss"""
        prediction_loss = nn.CrossEntropyLoss()(
            outputs['predictions'],
            data['risk_level']
        )
        
        if 'uncertainty' in outputs:
            uncertainty_loss = self._uncertainty_loss(
                outputs['uncertainty'],
                outputs['predictions'],
                data['risk_level']
            )
            return prediction_loss + 0.1 * uncertainty_loss
            
        return prediction_loss
