import torch
import torch.nn as nn
from transformers import LongformerModel, RobertaModel, T5EncoderModel
from .attention import MultiHeadHierarchicalAttention

class DocumentHierarchyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Text encoders
        self.news_encoder = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.filing_encoder = RobertaModel.from_pretrained('roberta-large')
        self.social_encoder = T5EncoderModel.from_pretrained('t5-base')
        
        # Hierarchical attention
        self.sentence_attention = MultiHeadHierarchicalAttention(
            config.hidden_dim,
            num_heads=config.num_heads
        )
        self.paragraph_attention = MultiHeadHierarchicalAttention(
            config.hidden_dim,
            num_heads=config.num_heads
        )
        self.document_attention = MultiHeadHierarchicalAttention(
            config.hidden_dim,
            num_heads=config.num_heads
        )

    def forward(self, batch):
        # Process different text types
        news_features = self.news_encoder(
            batch['news_ids'],
            attention_mask=batch['masks']['news']
        ).last_hidden_state
        
        filing_features = self.filing_encoder(
            batch['filing_ids'],
            attention_mask=batch['masks']['filing']
        ).last_hidden_state
        
        social_features = self.social_encoder(
            batch['social_ids'],
            attention_mask=batch['masks']['social']
        ).last_hidden_state
        
        # Build document hierarchy
        sentence_features = self.sentence_attention(
            news_features, filing_features, social_features,
            batch['hierarchy_markers']['sentence']
        )
        
        paragraph_features = self.paragraph_attention(
            sentence_features,
            batch['hierarchy_markers']['paragraph']
        )
        
        document_features = self.document_attention(
            paragraph_features,
            batch['hierarchy_markers']['document']
        )
        
        return document_features
