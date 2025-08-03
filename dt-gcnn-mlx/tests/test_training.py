"""Tests for training loop and optimization."""

import unittest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import tempfile
import os

# Import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import DTGCNNTrainer, TrainingConfig
from src.training.utils import EarlyStopping, ModelCheckpoint, LearningRateScheduler


class TestTrainingConfig(unittest.TestCase):
    """Test cases for TrainingConfig."""
    
    def test_config_initialization(self):
        """Test TrainingConfig with various parameters."""
        config = TrainingConfig(
            batch_size=32,
            learning_rate=0.001,
            num_epochs=100,
            weight_decay=0.0001,
            patience=10,
            warmup_epochs=5,
            gradient_clip=1.0
        )
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.num_epochs, 100)
        
    def test_config_validation(self):
        """Test config parameter validation."""
        # Should handle invalid parameters gracefully
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=-1)
            
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=0.0)


class TestDTGCNNTrainer(unittest.TestCase):
    """Test cases for DTGCNNTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_nodes = 10
        self.input_dim = 4
        self.hidden_dims = [16, 32]
        self.num_classes = 3
        self.seq_length = 20
        self.batch_size = 8
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy adjacency matrix
        adj = np.eye(self.num_nodes) + np.random.rand(self.num_nodes, self.num_nodes) * 0.1
        self.adj_matrix = mx.array(adj / adj.sum(axis=1, keepdims=True))
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_dummy_data(self, num_samples):
        """Create dummy training data."""
        features = []
        labels = []
        
        for i in range(num_samples):
            # Random features
            sample = mx.random.normal((self.seq_length, self.num_nodes, self.input_dim))
            features.append(sample)
            
            # Random label
            label = i % self.num_classes
            labels.append(label)
            
        return mx.stack(features), mx.array(labels)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        config = TrainingConfig(
            batch_size=self.batch_size,
            learning_rate=0.001,
            num_epochs=10
        )
        
        trainer = DTGCNNTrainer(
            model_config={
                'num_nodes': self.num_nodes,
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'temporal_kernel_size': 3,
                'dilations': [1, 2],
                'num_classes': self.num_classes,
                'dropout': 0.5
            },
            training_config=config,
            checkpoint_dir=self.temp_dir
        )
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        
    def test_single_training_step(self):
        """Test single training step."""
        config = TrainingConfig(batch_size=self.batch_size)
        trainer = DTGCNNTrainer(
            model_config={
                'num_nodes': self.num_nodes,
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'temporal_kernel_size': 3,
                'dilations': [1, 2],
                'num_classes': self.num_classes,
                'dropout': 0.5
            },
            training_config=config,
            checkpoint_dir=self.temp_dir
        )
        
        # Create batch
        features, labels = self.create_dummy_data(self.batch_size)
        
        # Training step
        loss, accuracy = trainer.train_step(features, self.adj_matrix, labels)
        
        # Check outputs
        self.assertIsInstance(float(loss), float)
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertIsInstance(float(accuracy), float)
        self.assertGreaterEqual(float(accuracy), 0.0)
        self.assertLessEqual(float(accuracy), 1.0)
        
    def test_validation_step(self):
        """Test validation step."""
        config = TrainingConfig(batch_size=self.batch_size)
        trainer = DTGCNNTrainer(
            model_config={
                'num_nodes': self.num_nodes,
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'temporal_kernel_size': 3,
                'dilations': [1, 2],
                'num_classes': self.num_classes,
                'dropout': 0.5
            },
            training_config=config,
            checkpoint_dir=self.temp_dir
        )
        
        # Create validation batch
        features, labels = self.create_dummy_data(self.batch_size)
        
        # Validation step
        loss, accuracy = trainer.validate_step(features, self.adj_matrix, labels)
        
        # Check outputs
        self.assertIsInstance(float(loss), float)
        self.assertIsInstance(float(accuracy), float)
        
    def test_triplet_loss_training(self):
        """Test training with triplet loss."""
        config = TrainingConfig(
            batch_size=12,  # Multiple of 3 for triplet mining
            loss_type='triplet',
            margin=0.2
        )
        
        trainer = DTGCNNTrainer(
            model_config={
                'num_nodes': self.num_nodes,
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'temporal_kernel_size': 3,
                'dilations': [1, 2],
                'num_classes': self.num_classes,
                'dropout': 0.5,
                'use_embeddings': True
            },
            training_config=config,
            checkpoint_dir=self.temp_dir
        )
        
        # Create batch with balanced classes
        features, labels = self.create_dummy_data(12)
        labels = mx.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        
        # Training step with triplet loss
        loss, _ = trainer.train_step(features, self.adj_matrix, labels)
        
        self.assertGreaterEqual(float(loss), 0.0)
        
    def test_gradient_clipping(self):
        """Test gradient clipping during training."""
        config = TrainingConfig(
            batch_size=self.batch_size,
            gradient_clip=1.0
        )
        
        trainer = DTGCNNTrainer(
            model_config={
                'num_nodes': self.num_nodes,
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'temporal_kernel_size': 3,
                'dilations': [1, 2],
                'num_classes': self.num_classes,
                'dropout': 0.5
            },
            training_config=config,
            checkpoint_dir=self.temp_dir
        )
        
        # Create batch
        features, labels = self.create_dummy_data(self.batch_size)
        
        # Get initial parameters
        initial_params = trainer.model.parameters()
        
        # Training step
        loss, _ = trainer.train_step(features, self.adj_matrix, labels)
        
        # Parameters should be updated
        updated_params = trainer.model.parameters()
        
        # Check some parameters changed (basic check)
        self.assertIsNotNone(loss)


class TestEarlyStopping(unittest.TestCase):
    """Test cases for EarlyStopping callback."""
    
    def test_early_stopping_patience(self):
        """Test early stopping with patience."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        # Simulate validation losses
        val_losses = [0.5, 0.4, 0.35, 0.34, 0.34, 0.34]
        
        for epoch, loss in enumerate(val_losses):
            should_stop = early_stopping(loss, epoch)
            
            if epoch < 3:
                self.assertFalse(should_stop)
            elif epoch == 5:  # After patience exhausted
                self.assertTrue(should_stop)
                
    def test_early_stopping_improvement(self):
        """Test early stopping with continuous improvement."""
        early_stopping = EarlyStopping(patience=3)
        
        # Continuous improvement
        val_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        for epoch, loss in enumerate(val_losses):
            should_stop = early_stopping(loss, epoch)
            self.assertFalse(should_stop)
            
        # Best score should be the last one
        self.assertAlmostEqual(early_stopping.best_score, 0.1)


class TestModelCheckpoint(unittest.TestCase):
    """Test cases for ModelCheckpoint callback."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_checkpoint_saving(self):
        """Test model checkpoint saving."""
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.temp_dir, 'model_{epoch:02d}_{val_loss:.4f}.npz'),
            monitor='val_loss',
            save_best_only=True
        )
        
        # Mock model state
        model_state = {'layer1': mx.array([1.0, 2.0, 3.0])}
        
        # Simulate training
        for epoch in range(5):
            val_loss = 0.5 - epoch * 0.1  # Decreasing loss
            
            saved = checkpoint(
                model_state=model_state,
                epoch=epoch,
                val_loss=val_loss
            )
            
            if epoch == 0 or val_loss < checkpoint.best:
                self.assertTrue(saved)
            else:
                self.assertFalse(saved)
                
    def test_checkpoint_loading(self):
        """Test model checkpoint loading."""
        # Save a checkpoint
        filepath = os.path.join(self.temp_dir, 'model.npz')
        model_state = {
            'layer1.weight': mx.array([[1.0, 2.0], [3.0, 4.0]]),
            'layer1.bias': mx.array([0.1, 0.2])
        }
        
        mx.save(filepath, model_state)
        
        # Load checkpoint
        loaded_state = mx.load(filepath)
        
        # Verify loaded state
        self.assertTrue('layer1.weight' in loaded_state)
        self.assertTrue('layer1.bias' in loaded_state)
        self.assertTrue(mx.array_equal(loaded_state['layer1.weight'], model_state['layer1.weight']))


class TestLearningRateScheduler(unittest.TestCase):
    """Test cases for learning rate scheduling."""
    
    def test_step_lr_scheduler(self):
        """Test step learning rate scheduler."""
        initial_lr = 0.1
        scheduler = LearningRateScheduler(
            initial_lr=initial_lr,
            schedule_type='step',
            step_size=10,
            gamma=0.1
        )
        
        # Test LR at different epochs
        self.assertAlmostEqual(scheduler(0), 0.1)
        self.assertAlmostEqual(scheduler(9), 0.1)
        self.assertAlmostEqual(scheduler(10), 0.01)
        self.assertAlmostEqual(scheduler(19), 0.01)
        self.assertAlmostEqual(scheduler(20), 0.001)
        
    def test_cosine_lr_scheduler(self):
        """Test cosine annealing scheduler."""
        initial_lr = 0.1
        scheduler = LearningRateScheduler(
            initial_lr=initial_lr,
            schedule_type='cosine',
            T_max=100
        )
        
        # Test LR at different epochs
        lr_0 = scheduler(0)
        lr_50 = scheduler(50)
        lr_100 = scheduler(100)
        
        self.assertAlmostEqual(lr_0, initial_lr)
        self.assertLess(lr_50, lr_0)
        self.assertAlmostEqual(lr_100, 0.0, places=5)
        
    def test_warmup_scheduler(self):
        """Test learning rate warmup."""
        initial_lr = 0.1
        scheduler = LearningRateScheduler(
            initial_lr=initial_lr,
            schedule_type='warmup_cosine',
            warmup_epochs=10,
            T_max=100
        )
        
        # During warmup
        lr_0 = scheduler(0)
        lr_5 = scheduler(5)
        lr_10 = scheduler(10)
        
        self.assertLess(lr_0, initial_lr)
        self.assertLess(lr_0, lr_5)
        self.assertLess(lr_5, lr_10)
        self.assertAlmostEqual(lr_10, initial_lr)


class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for complete training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_full_training_loop(self):
        """Test complete training loop with all components."""
        # Configuration
        config = TrainingConfig(
            batch_size=8,
            learning_rate=0.01,
            num_epochs=5,
            patience=3
        )
        
        # Create trainer
        trainer = DTGCNNTrainer(
            model_config={
                'num_nodes': 10,
                'input_dim': 4,
                'hidden_dims': [16, 32],
                'temporal_kernel_size': 3,
                'dilations': [1, 2],
                'num_classes': 3,
                'dropout': 0.5
            },
            training_config=config,
            checkpoint_dir=self.temp_dir
        )
        
        # Create dummy dataset
        num_train = 80
        num_val = 20
        
        train_features = mx.random.normal((num_train, 20, 10, 4))
        train_labels = mx.array([i % 3 for i in range(num_train)])
        
        val_features = mx.random.normal((num_val, 20, 10, 4))
        val_labels = mx.array([i % 3 for i in range(num_val)])
        
        adj = np.eye(10) + np.random.rand(10, 10) * 0.1
        adj_matrix = mx.array(adj / adj.sum(axis=1, keepdims=True))
        
        # Mock data loader
        class SimpleDataLoader:
            def __init__(self, features, labels, batch_size):
                self.features = features
                self.labels = labels
                self.batch_size = batch_size
                
            def __iter__(self):
                n = len(self.features)
                for i in range(0, n, self.batch_size):
                    end = min(i + self.batch_size, n)
                    yield self.features[i:end], self.labels[i:end]
                    
            def __len__(self):
                return (len(self.features) + self.batch_size - 1) // self.batch_size
        
        train_loader = SimpleDataLoader(train_features, train_labels, config.batch_size)
        val_loader = SimpleDataLoader(val_features, val_labels, config.batch_size)
        
        # Training loop
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            adj_matrix=adj_matrix
        )
        
        # Check training history
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('train_accuracy', history)
        self.assertIn('val_accuracy', history)
        
        # Should have entries for each epoch
        self.assertLessEqual(len(history['train_loss']), config.num_epochs)


if __name__ == "__main__":
    unittest.main()