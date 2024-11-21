import asyncio
from realtime_pipeline import RealtimePipeline, StreamConfig
from typing import Dict
import numpy as np
import time

async def simulate_market_data(pipeline: RealtimePipeline):
    """Simulate market data stream"""
    while True:
        data = {
            'price': np.random.normal(100, 5),
            'volume': np.random.exponential(1000),
            'volatility': abs(np.random.normal(0, 0.1)),
            'momentum': np.random.normal(0, 1),
            'sentiment': np.random.uniform(-1, 1)
        }
        await pipeline.process_market_data(data)
        await asyncio.sleep(0.1)  # Simulate 10Hz market data

async def simulate_text_data(pipeline: RealtimePipeline):
    """Simulate text data stream"""
    texts = [
        "Market showing strong upward momentum",
        "Investors concerned about economic outlook",
        "Central bank announces policy change",
        "Major market volatility expected"
    ]
    
    while True:
        text = np.random.choice(texts)
        metadata = {'source': 'news', 'priority': 'high'}
        await pipeline.process_text_data(text, metadata)
        await asyncio.sleep(1.0)  # Simulate 1Hz text data

async def monitor_performance(pipeline: RealtimePipeline):
    """Monitor pipeline performance"""
    while True:
        stats = pipeline.get_performance_stats()
        print(f"Performance Stats: {stats}")
        await asyncio.sleep(5.0)  # Update every 5 seconds

async def main():
    config = StreamConfig()
    pipeline = RealtimePipeline(model=None, config=config)  # Replace None with actual model
    
    # Create tasks
    market_task = asyncio.create_task(simulate_market_data(pipeline))
    text_task = asyncio.create_task(simulate_text_data(pipeline))
    monitor_task = asyncio.create_task(monitor_performance(pipeline))
    
    try:
        # Run until interrupted
        await asyncio.gather(
            market_task,
            text_task,
            monitor_task
        )
    except KeyboardInterrupt:
        print("Shutting down pipeline...")
    finally:
        # Cleanup
        market_task.cancel()
        text_task.cancel()
        monitor_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
