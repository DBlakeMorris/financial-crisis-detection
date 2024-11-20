import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class FinancialEntityGraph(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.entity_embeddings = nn.Parameter(
            torch.randn(config.num_entities, config.hidden_dim)
        )
        
        # Graph transformer layers
        self.graph_layers = nn.ModuleList([
            TransformerConv(
                config.hidden_dim,
                config.hidden_dim // config.num_heads,
                heads=config.num_heads,
                dropout=config.dropout,
                edge_dim=config.hidden_dim
            ) for _ in range(config.num_layers)
        ])
        
        # Edge prediction
        self.edge_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )

    def construct_dynamic_graph(self, entity_features, mention_locations):
        N = entity_features.size(0)
        entity_pairs = torch.cat([
            entity_features.unsqueeze(1).repeat(1, N, 1),
            entity_features.unsqueeze(0).repeat(N, 1, 1)
        ], dim=-1)
        
        edge_weights = self.edge_predictor(entity_pairs)
        edges = (edge_weights > 0.5).nonzero()
        edge_features = edge_weights[edges[:, 0], edges[:, 1]]
        
        return edges, edge_features

    def forward(self, mention_features, mention_locations):
        entity_features = self.entity_embeddings
        edges, edge_features = self.construct_dynamic_graph(
            entity_features, mention_locations
        )
        
        for layer in self.graph_layers:
            entity_features = entity_features + layer(
                entity_features, edges, edge_features
            )
            
        return entity_features
