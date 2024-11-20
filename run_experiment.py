import subprocess
import os
from pathlib import Path
import glob

def main():
    # Set environment variable for tokenizer
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train the model
    print("Starting training...")
    subprocess.run(["python", "src/training/lite_trainer.py"], env=os.environ.copy())
    
    # Find the latest checkpoint file
    checkpoint_files = glob.glob('checkpoints/**/best-model*.ckpt', recursive=True)
    
    if not checkpoint_files:
        print("No checkpoint files found in checkpoints directory")
        return
    
    # Get the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"\nFound checkpoint: {latest_checkpoint}")
    
    # Run evaluation
    print("\nStarting evaluation...")
    subprocess.run([
        "python", 
        "src/evaluation/evaluate.py",
        "--checkpoint_path", latest_checkpoint
    ], env=os.environ.copy())

if __name__ == "__main__":
    main()
