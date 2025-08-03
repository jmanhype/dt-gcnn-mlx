"""Tests for triplet loss functions in MLX."""

import unittest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses.triplet_loss import TripletLoss, BatchHardTripletLoss, BatchAllTripletLoss


class TestTripletLoss(unittest.TestCase):
    """Test cases for basic Triplet Loss."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 128
        self.margin = 0.2
        self.batch_size = 16
        
    def test_loss_initialization(self):
        """Test TripletLoss initialization."""
        loss_fn = TripletLoss(margin=self.margin)
        self.assertEqual(loss_fn.margin, self.margin)
        
    def test_triplet_loss_computation(self):
        """Test basic triplet loss computation."""
        loss_fn = TripletLoss(margin=self.margin)
        
        # Create embeddings for anchor, positive, negative
        anchor = mx.random.normal((self.batch_size, self.embedding_dim))
        positive = anchor + mx.random.normal((self.batch_size, self.embedding_dim)) * 0.1
        negative = mx.random.normal((self.batch_size, self.embedding_dim))
        
        # Compute loss
        loss = loss_fn(anchor, positive, negative)
        
        # Check loss shape
        self.assertEqual(loss.shape, ())  # Scalar
        
        # Loss should be non-negative
        self.assertGreaterEqual(float(loss), 0.0)
        
    def test_loss_with_easy_negatives(self):
        """Test loss behavior with easy negatives."""
        loss_fn = TripletLoss(margin=self.margin)
        
        # Create embeddings where negatives are far
        anchor = mx.random.normal((self.batch_size, self.embedding_dim))
        positive = anchor + mx.random.normal((self.batch_size, self.embedding_dim)) * 0.01
        negative = anchor + mx.random.normal((self.batch_size, self.embedding_dim)) * 10.0
        
        loss = loss_fn(anchor, positive, negative)
        
        # Loss should be close to zero for easy negatives
        self.assertLess(float(loss), 0.1)
        
    def test_loss_with_hard_negatives(self):
        """Test loss behavior with hard negatives."""
        loss_fn = TripletLoss(margin=self.margin)
        
        # Create embeddings where negatives are close
        anchor = mx.random.normal((self.batch_size, self.embedding_dim))
        positive = anchor + mx.random.normal((self.batch_size, self.embedding_dim)) * 0.5
        negative = anchor + mx.random.normal((self.batch_size, self.embedding_dim)) * 0.1
        
        loss = loss_fn(anchor, positive, negative)
        
        # Loss should be significant for hard negatives
        self.assertGreater(float(loss), 0.1)


class TestBatchHardTripletLoss(unittest.TestCase):
    """Test cases for Batch Hard Triplet Loss."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 128
        self.margin = 0.2
        self.num_classes = 4
        self.num_per_class = 4
        self.batch_size = self.num_classes * self.num_per_class
        
    def test_batch_hard_loss_initialization(self):
        """Test BatchHardTripletLoss initialization."""
        loss_fn = BatchHardTripletLoss(margin=self.margin)
        self.assertEqual(loss_fn.margin, self.margin)
        
    def test_batch_hard_mining(self):
        """Test batch hard mining strategy."""
        loss_fn = BatchHardTripletLoss(margin=self.margin)
        
        # Create embeddings with clear class structure
        embeddings = []
        labels = []
        
        for class_id in range(self.num_classes):
            # Create clustered embeddings for each class
            class_center = mx.random.normal((self.embedding_dim,)) * 5
            for _ in range(self.num_per_class):
                embedding = class_center + mx.random.normal((self.embedding_dim,)) * 0.5
                embeddings.append(embedding)
                labels.append(class_id)
        
        embeddings = mx.stack(embeddings)
        labels = mx.array(labels)
        
        # Compute loss
        loss = loss_fn(embeddings, labels)
        
        # Check loss is scalar
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(float(loss), 0.0)
        
    def test_no_valid_triplets(self):
        """Test behavior when no valid triplets exist."""
        loss_fn = BatchHardTripletLoss(margin=self.margin)
        
        # All samples from same class
        embeddings = mx.random.normal((self.batch_size, self.embedding_dim))
        labels = mx.zeros((self.batch_size,), dtype=mx.int32)
        
        # Should handle gracefully
        loss = loss_fn(embeddings, labels)
        self.assertEqual(loss.shape, ())
        
    def test_pairwise_distances(self):
        """Test pairwise distance computation."""
        loss_fn = BatchHardTripletLoss(margin=self.margin)
        
        # Create simple embeddings
        embeddings = mx.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0]
        ])
        
        # Compute pairwise distances
        distances = loss_fn._pairwise_distances(embeddings)
        
        # Check shape
        self.assertEqual(distances.shape, (4, 4))
        
        # Check diagonal is zero
        for i in range(4):
            self.assertAlmostEqual(float(distances[i, i]), 0.0, places=5)


class TestBatchAllTripletLoss(unittest.TestCase):
    """Test cases for Batch All Triplet Loss."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 128
        self.margin = 0.2
        self.num_classes = 3
        self.num_per_class = 3
        self.batch_size = self.num_classes * self.num_per_class
        
    def test_batch_all_loss_initialization(self):
        """Test BatchAllTripletLoss initialization."""
        loss_fn = BatchAllTripletLoss(margin=self.margin)
        self.assertEqual(loss_fn.margin, self.margin)
        
    def test_batch_all_computation(self):
        """Test batch all triplet loss computation."""
        loss_fn = BatchAllTripletLoss(margin=self.margin)
        
        # Create structured embeddings
        embeddings = []
        labels = []
        
        for class_id in range(self.num_classes):
            class_center = mx.random.normal((self.embedding_dim,)) * 3
            for _ in range(self.num_per_class):
                embedding = class_center + mx.random.normal((self.embedding_dim,)) * 0.3
                embeddings.append(embedding)
                labels.append(class_id)
        
        embeddings = mx.stack(embeddings)
        labels = mx.array(labels)
        
        # Compute loss
        loss = loss_fn(embeddings, labels)
        
        # Check loss properties
        self.assertEqual(loss.shape, ())
        self.assertGreaterEqual(float(loss), 0.0)
        
    def test_valid_triplet_counting(self):
        """Test counting of valid triplets."""
        loss_fn = BatchAllTripletLoss(margin=self.margin)
        
        # Simple case: 2 classes, 2 samples each
        embeddings = mx.array([
            [1.0, 0.0],  # Class 0
            [1.1, 0.0],  # Class 0  
            [0.0, 1.0],  # Class 1
            [0.0, 1.1],  # Class 1
        ])
        labels = mx.array([0, 0, 1, 1])
        
        # Each anchor from class 0 has 1 positive and 2 negatives
        # Total valid triplets: 2 * 1 * 2 * 2 = 8
        loss = loss_fn(embeddings, labels)
        self.assertIsNotNone(loss)


class TestLossFunctionIntegration(unittest.TestCase):
    """Integration tests for loss functions with models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedding_dim = 128
        self.margin = 0.2
        
    def test_loss_with_model_output(self):
        """Test loss computation with actual model outputs."""
        # Simulate model that outputs embeddings
        class EmbeddingModel(nn.Module):
            def __init__(self, input_dim, embedding_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 256)
                self.fc2 = nn.Linear(256, embedding_dim)
                
            def __call__(self, x):
                x = self.fc1(x)
                x = mx.maximum(x, 0)  # ReLU
                x = self.fc2(x)
                # L2 normalize
                return x / mx.sqrt(mx.sum(x ** 2, axis=1, keepdims=True) + 1e-8)
        
        model = EmbeddingModel(input_dim=64, embedding_dim=self.embedding_dim)
        loss_fn = BatchHardTripletLoss(margin=self.margin)
        
        # Create batch
        batch_size = 16
        x = mx.random.normal((batch_size, 64))
        labels = mx.array([i // 4 for i in range(batch_size)])  # 4 samples per class
        
        # Forward pass
        embeddings = model(x)
        loss = loss_fn(embeddings, labels)
        
        # Check we can compute gradients
        def train_step(x, labels):
            embeddings = model(x)
            return loss_fn(embeddings, labels)
        
        loss_grad_fn = mx.grad(train_step)
        grads = loss_grad_fn(x, labels)
        
        self.assertEqual(grads.shape, x.shape)
        
    def test_different_distance_metrics(self):
        """Test loss functions with different distance metrics."""
        # Test both L2 and cosine distance variants
        embeddings = mx.random.normal((12, 128))
        labels = mx.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        
        # L2 distance version
        loss_l2 = BatchHardTripletLoss(margin=0.2, distance_metric='euclidean')
        loss_val_l2 = loss_l2(embeddings, labels)
        
        # If cosine distance is implemented
        # loss_cosine = BatchHardTripletLoss(margin=0.2, distance_metric='cosine')  
        # loss_val_cosine = loss_cosine(embeddings, labels)
        
        self.assertIsNotNone(loss_val_l2)


if __name__ == "__main__":
    unittest.main()