#!/usr/bin/env python3
"""
Quick Start Example for DT-GCNN MLX
Demonstrates basic usage with synthetic data
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from pathlib import Path

# Import DT-GCNN components
from src.models import create_model, DTGCNN_CONFIGS
from src.losses import TripletLoss, BatchHardMiner
from src.data import create_sample_data
from src.training import DT_GCNN_Trainer

def main():
    print("🚀 DT-GCNN MLX Quick Start Demo")
    print("=" * 50)
    
    # 1. Check MLX device
    device = mx.default_device()
    print(f"✓ Using device: {device}")
    print(f"✓ Metal GPU available: {'gpu' in str(device)}")
    
    # 2. Create synthetic data
    print("\n📊 Generating synthetic data...")
    data_dir = Path("demo_data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate 1000 triplets with 4 classes
    create_sample_data(
        output_dir=data_dir,
        num_samples=1000,
        num_classes=4,
        sequence_length=128,
        vocab_size=5000
    )
    print("✓ Data generated successfully")
    
    # 3. Initialize model
    print("\n🏗️ Creating DT-GCNN model...")
    model = create_model(
        config_name="small",  # Use small config for quick demo
        vocab_size=5000,
        num_classes=4
    )
    print(f"✓ Model created with {sum(p.size for p in model.parameters().values())/1e6:.2f}M parameters")
    
    # 4. Setup losses
    print("\n📉 Setting up losses...")
    triplet_loss = TripletLoss(
        margin=0.3,
        distance_metric="euclidean",
        dynamic_margin=True
    )
    miner = BatchHardMiner()
    print("✓ Triplet loss and miner configured")
    
    # 5. Initialize trainer
    print("\n🎯 Initializing trainer...")
    trainer = DT_GCNN_Trainer(
        model=model,
        triplet_loss=triplet_loss,
        miner=miner,
        learning_rate=1e-3,
        weight_decay=0.01,
        num_classes=4,
        classification_weight=1.0,
        triplet_weight=1.0
    )
    print("✓ Trainer ready")
    
    # 6. Quick training demo
    print("\n🏃 Running quick training demo (5 steps)...")
    print("-" * 50)
    
    # Generate dummy batch for demo
    batch_size = 32
    seq_length = 128
    
    for step in range(5):
        # Create random batch (in real training, use DataLoader)
        anchor = mx.random.randint(0, 5000, [batch_size, seq_length])
        positive = mx.random.randint(0, 5000, [batch_size, seq_length])
        negative = mx.random.randint(0, 5000, [batch_size, seq_length])
        labels = mx.random.randint(0, 4, [batch_size])
        
        # Training step
        metrics = trainer.train_step(anchor, positive, negative, labels)
        
        print(f"Step {step+1}/5:")
        print(f"  Total Loss: {metrics['loss']:.4f}")
        print(f"  Classification Loss: {metrics['classification_loss']:.4f}")
        print(f"  Triplet Loss: {metrics['triplet_loss']:.4f}")
        if 'accuracy' in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
    
    print("-" * 50)
    print("\n✅ Quick start demo completed!")
    print("\n📚 Next steps:")
    print("1. Train on real data: python src/training/train.py --data-dir your_data")
    print("2. Run benchmarks: python examples/performance_benchmark.py")
    print("3. See examples: python examples/simple_classification.py")
    print("4. Read docs: docs/training_guide.md")

if __name__ == "__main__":
    main()