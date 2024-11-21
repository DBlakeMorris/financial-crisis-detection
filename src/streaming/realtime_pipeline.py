import torch
from typing import Dict, Optional, List, Tuple
import numpy as np
from datetime import datetime
import asyncio
from dataclasses import dataclass
import logging
from collections import deque

@dataclass
class StreamConfig:
    batch_size: int = 32
    buffer_size: int = 1000
    update_interval: float = 0.1
    feature_dim: int = 256
    max_latency: float = 0.05

class RealtimePipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        config: StreamConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Buffers for streaming data
        self.market_buffer = deque(maxlen=config.buffer_size)
        self.text_buffer = deque(maxlen=config.buffer_size)
        
        # Performance monitoring
        self.latency_tracker = deque(maxlen=1000)
        self.prediction_tracker = deque(maxlen=1000)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    async def process_market_data(
        self,
        data: Dict[str, float]
    ):
        """Process incoming market data asynchronously"""
        start_time = datetime.now()
        
        try:
            # Add to buffer
            self.market_buffer.append(data)
            
            # Process if buffer is full
            if len(self.market_buffer) >= self.config.batch_size:
                await self.generate_prediction()
                
            # Track latency
            latency = (datetime.now() - start_time).total_seconds()
            self.latency_tracker.append(latency)
            
            # Check latency threshold
            if latency > self.config.max_latency:
                self.logger.warning(f"High latency detected: {latency:.3f}s")
                
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            raise
            
    async def process_text_data(
        self,
        text: str,
        metadata: Dict[str, str]
    ):
        """Process incoming text data asynchronously"""
        try:
            # Add to buffer
            self.text_buffer.append({
                'text': text,
                'metadata': metadata,
                'timestamp': datetime.now()
            })
            
            # Clean old data
            self._clean_text_buffer()
            
        except Exception as e:
            self.logger.error(f"Error processing text data: {str(e)}")
            raise
            
    async def generate_prediction(self) -> Dict[str, float]:
        """Generate model prediction from current buffers"""
        try:
            # Prepare features
            market_features = self._prepare_market_features()
            text_features = await self._prepare_text_features()
            
            # Combined features
            combined_features = self._combine_features(
                market_features,
                text_features
            )
            
            # Generate prediction
            with torch.no_grad():
                prediction = await self._model_inference(combined_features)
                
            # Track prediction
            self.prediction_tracker.append(prediction)
            
            # Log prediction
            self.logger.info(
                f"Generated prediction: Risk Level {prediction['risk_level']:.2f} "
                f"(Uncertainty: {prediction['uncertainty']:.3f})"
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {str(e)}")
            raise
            
    def _prepare_market_features(self) -> torch.Tensor:
        """Prepare market data features"""
        features = []
        for data in list(self.market_buffer)[-self.config.batch_size:]:
            features.append([
                data.get(k, 0.0) for k in [
                    'price', 'volume', 'volatility',
                    'momentum', 'sentiment'
                ]
            ])
        
        return torch.tensor(
            features,
            dtype=torch.float32,
            device=self.device
        )
        
    async def _prepare_text_features(self) -> torch.Tensor:
        """Prepare text features asynchronously"""
        texts = [
            item['text']
            for item in list(self.text_buffer)[-self.config.batch_size:]
        ]
        
        # Process in smaller batches if needed
        batch_size = min(8, len(texts))  # Smaller batch size for text processing
        features = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_features = await self._process_text_batch(batch_texts)
            features.append(batch_features)
            
        return torch.cat(features, dim=0)
        
    async def _process_text_batch(
        self,
        texts: List[str]
    ) -> torch.Tensor:
        """Process a batch of texts asynchronously"""
        # This would be replaced with actual text processing
        # Currently using dummy features
        features = torch.randn(
            len(texts),
            self.config.feature_dim,
            device=self.device
        )
        return features
        
    def _combine_features(
        self,
        market_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Combine market and text features"""
        # Ensure same batch size
        min_batch = min(
            market_features.size(0),
            text_features.size(0)
        )
        market_features = market_features[:min_batch]
        text_features = text_features[:min_batch]
        
        return torch.cat([market_features, text_features], dim=-1)
        
    async def _model_inference(
        self,
        features: torch.Tensor
    ) -> Dict[str, float]:
        """Run model inference asynchronously"""
        outputs = self.model(features)
        
        return {
            'risk_level': outputs['risk_logits'].mean().item(),
            'uncertainty': outputs['uncertainty'].mean().item(),
            'timestamp': datetime.now().isoformat()
        }
        
    def _clean_text_buffer(self):
        """Clean old text data from buffer"""
        current_time = datetime.now()
        self.text_buffer = deque(
            [
                item for item in self.text_buffer
                if (current_time - item['timestamp']).total_seconds() < 3600
            ],
            maxlen=self.config.buffer_size
        )
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get pipeline performance statistics"""
        return {
            'mean_latency': np.mean(self.latency_tracker),
            'max_latency': np.max(self.latency_tracker),
            'buffer_utilization': len(self.market_buffer) / self.config.buffer_size,
            'prediction_rate': len(self.prediction_tracker) / 1000,
            'mean_risk_level': np.mean([
                p['risk_level'] for p in self.prediction_tracker
            ]),
            'mean_uncertainty': np.mean([
                p['uncertainty'] for p in self.prediction_tracker
            ])
        }
