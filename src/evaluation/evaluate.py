import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
import argparse
from src.models.lite_model import FinancialCrisisLite
from src.data.lite_datamodule import FinancialDataModuleLite
from omegaconf import OmegaConf

def move_batch_to_device(batch, device):
    """Recursively move batch to device"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(item, device) for item in batch]
    return batch

def evaluate_model(model, dataloader, device='mps'):
    model.eval()
    model = model.to(device)
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move everything to the same device
            batch = move_batch_to_device(batch, device)
            
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['risk_level'].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()

    # Create basic config
    config = {
        'model': {
            'hidden_dim': 768,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'num_entities': 1000,
            'market_dim': 32,
            'num_risk_levels': 5,
            'max_seq_length': 512
        },
        'training': {
            'learning_rate': 1e-4,
            'batch_size': 8
        },
        'data': {
            'max_seq_length': 512,
            'num_workers': 0  # Set to 0 to avoid device issues
        }
    }
    
    config = OmegaConf.create(config)
    
    # Initialize WandB
    wandb.init(project="financial-crisis-lite-eval")
    
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = FinancialCrisisLite.load_from_checkpoint(args.checkpoint_path, config=config)
    model = model.to(device)
    
    # Prepare data
    datamodule = FinancialDataModuleLite(config)
    datamodule.setup(stage='test')
    
    # Evaluate
    print("Starting evaluation...")
    preds, labels, probs = evaluate_model(model, datamodule.test_dataloader(), device=device)
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(labels, preds))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Log to wandb
    wandb.log({
        "test_accuracy": (preds == labels).mean(),
        "confusion_matrix": wandb.Image('confusion_matrix.png'),
        "classification_report": wandb.Table(
            dataframe=pd.DataFrame(
                classification_report(labels, preds, output_dict=True)
            ).transpose()
        )
    })
    
    # Print additional metrics
    print(f"\nTest Accuracy: {(preds == labels).mean():.4f}")
    print("\nConfusion Matrix saved as confusion_matrix.png")
    
    wandb.finish()

if __name__ == "__main__":
    main()
