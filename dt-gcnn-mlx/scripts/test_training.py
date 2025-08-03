#!/usr/bin/env python3
"""
Test script to verify training system functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
import numpy as np
from models.dt_gcnn import DT_GCNN
from training.trainer import DT_GCNN_Trainer, TrainingConfig
from training.data_loader import create_sample_data, create_data_loaders
from training.utils import TrainingMonitor, ModelProfiler
import tempfile
import shutil


def test_model_creation():
    """Test model creation and forward pass"""
    print("Testing model creation...")
    
    model = DT_GCNN(
        num_vertices=1723,
        embedding_dim=128,
        num_classes=10
    )
    
    # Test forward pass
    batch_size = 4
    dummy_coords = mx.random.normal((batch_size, 1723, 3))
    dummy_features = mx.random.normal((batch_size, 1723, 6))
    
    embeddings, logits = model(dummy_coords, dummy_features)
    
    assert embeddings.shape == (batch_size, 128)
    assert logits.shape == (batch_size, 10)
    
    print("✓ Model creation successful")
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Logits shape: {logits.shape}")
    
    return model


def test_data_loading():
    """Test data loading pipeline"""
    print("\nTesting data loading...")
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample data
        create_sample_data(temp_dir, num_samples=50)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            temp_dir,
            batch_size=8,
            augment=True
        )
        
        # Test iteration
        batch = next(iter(train_loader))
        
        assert 'coordinates' in batch
        assert 'features' in batch
        assert 'labels' in batch
        
        print("✓ Data loading successful")
        print(f"  - Batch coordinates shape: {batch['coordinates'].shape}")
        print(f"  - Batch features shape: {batch['features'].shape}")
        print(f"  - Batch labels shape: {batch['labels'].shape}")
        
        if 'triplet_indices' in batch:
            print(f"  - Triplet indices available")
        
        return train_loader, val_loader


def test_training_step(model, train_loader):
    """Test single training step"""
    print("\nTesting training step...")
    
    config = TrainingConfig(
        batch_size=8,
        learning_rate=0.001,
        num_epochs=1
    )
    
    trainer = DT_GCNN_Trainer(model, config)
    
    # Get a batch
    batch = next(iter(train_loader))
    
    # Run training step
    loss, metrics = trainer.train_step(batch)
    
    print("✓ Training step successful")
    print(f"  - Loss: {metrics['loss']:.4f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    
    if 'triplet_loss' in metrics:
        print(f"  - Triplet loss: {metrics['triplet_loss']:.4f}")
    
    return trainer


def test_validation(trainer, val_loader):
    """Test validation loop"""
    print("\nTesting validation...")
    
    val_metrics = trainer.validate(val_loader)
    
    print("✓ Validation successful")
    print(f"  - Val loss: {val_metrics['loss']:.4f}")
    print(f"  - Val accuracy: {val_metrics['accuracy']:.4f}")


def test_checkpoint_saving(trainer):
    """Test checkpoint saving and loading"""
    print("\nTesting checkpointing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update checkpoint directory
        trainer.config.checkpoint_dir = temp_dir
        trainer.checkpoint_dir = Path(temp_dir)
        
        # Save checkpoint
        metrics = {'train_loss': 0.5, 'val_loss': 0.6}
        trainer.save_checkpoint(0, metrics, is_best=True)
        
        # Check if files exist
        checkpoint_file = Path(temp_dir) / "checkpoint_epoch_0000.npz"
        best_file = Path(temp_dir) / "best_model.npz"
        
        assert checkpoint_file.exists()
        assert best_file.exists()
        
        # Test loading
        loaded_metrics = trainer.load_checkpoint(str(best_file))
        
        print("✓ Checkpointing successful")
        print(f"  - Checkpoint saved to: {checkpoint_file}")
        print(f"  - Best model saved to: {best_file}")


def test_memory_usage():
    """Test memory profiling"""
    print("\nTesting memory usage...")
    
    if mx.metal.is_available():
        initial_memory = mx.metal.get_active_memory() / 1e6
        print(f"✓ Metal GPU available")
        print(f"  - Initial memory: {initial_memory:.2f} MB")
        
        # Create large model
        model = DT_GCNN(
            num_vertices=1723,
            embedding_dim=512,
            num_classes=10
        )
        
        # Initialize
        dummy = mx.zeros((1, 1723, 3))
        _ = model(dummy, mx.zeros((1, 1723, 6)))
        
        after_model = mx.metal.get_active_memory() / 1e6
        print(f"  - After model creation: {after_model:.2f} MB")
        print(f"  - Model memory usage: {after_model - initial_memory:.2f} MB")
    else:
        print("✓ Running on CPU (Metal not available)")


def test_full_training():
    """Test complete training loop"""
    print("\nTesting full training loop...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample data
        data_dir = Path(temp_dir) / "data"
        create_sample_data(str(data_dir), num_samples=100)
        
        # Create output directory
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()
        
        # Create small model and config
        model = DT_GCNN(
            num_vertices=1723,
            embedding_dim=64,
            num_classes=10
        )
        
        config = TrainingConfig(
            batch_size=8,
            num_epochs=3,
            learning_rate=0.001,
            checkpoint_dir=str(output_dir / "checkpoints"),
            verbose=False
        )
        
        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(
            str(data_dir),
            batch_size=config.batch_size
        )
        
        # Create trainer
        trainer = DT_GCNN_Trainer(model, config)
        
        # Run training
        history = trainer.train(train_loader, val_loader)
        
        print("✓ Full training successful")
        print(f"  - Completed {len(history['train_loss'])} epochs")
        print(f"  - Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  - Final val loss: {history['val_loss'][-1]:.4f}")


def main():
    """Run all tests"""
    print("=" * 50)
    print("DT-GCNN Training System Test Suite")
    print("=" * 50)
    
    try:
        # Test components
        model = test_model_creation()
        train_loader, val_loader = test_data_loading()
        trainer = test_training_step(model, train_loader)
        test_validation(trainer, val_loader)
        test_checkpoint_saving(trainer)
        test_memory_usage()
        test_full_training()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()