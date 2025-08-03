"""Tests for data loading and preprocessing pipeline."""

import unittest
import mlx.core as mx
import numpy as np
import tempfile
import os
import json

# Import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.graph_dataset import GraphDataset, GraphBatchSampler
from src.data.preprocessing import (
    normalize_adjacency_matrix,
    compute_laplacian,
    temporal_signal_augmentation,
    create_sliding_windows
)


class TestGraphDataset(unittest.TestCase):
    """Test cases for GraphDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_samples = 100
        self.num_nodes = 20
        self.num_features = 8
        self.seq_length = 50
        self.num_classes = 5
        
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def create_dummy_dataset(self):
        """Create dummy dataset files."""
        # Create adjacency matrix
        adj = np.eye(self.num_nodes) + np.random.rand(self.num_nodes, self.num_nodes) * 0.1
        adj = (adj + adj.T) / 2  # Make symmetric
        np.save(os.path.join(self.temp_dir, 'adjacency.npy'), adj)
        
        # Create node features
        features = []
        labels = []
        
        for i in range(self.num_samples):
            # Time series data for each sample
            sample_features = np.random.randn(self.seq_length, self.num_nodes, self.num_features)
            features.append(sample_features)
            
            # Label for classification
            labels.append(i % self.num_classes)
        
        np.save(os.path.join(self.temp_dir, 'features.npy'), np.array(features))
        np.save(os.path.join(self.temp_dir, 'labels.npy'), np.array(labels))
        
        # Create metadata
        metadata = {
            'num_samples': self.num_samples,
            'num_nodes': self.num_nodes,
            'num_features': self.num_features,
            'seq_length': self.seq_length,
            'num_classes': self.num_classes
        }
        
        with open(os.path.join(self.temp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
            
    def test_dataset_loading(self):
        """Test loading dataset from files."""
        self.create_dummy_dataset()
        
        dataset = GraphDataset(
            data_dir=self.temp_dir,
            transform=None,
            cache_size=10
        )
        
        # Check dataset properties
        self.assertEqual(len(dataset), self.num_samples)
        self.assertEqual(dataset.num_nodes, self.num_nodes)
        self.assertEqual(dataset.num_features, self.num_features)
        
    def test_dataset_indexing(self):
        """Test dataset indexing and data retrieval."""
        self.create_dummy_dataset()
        
        dataset = GraphDataset(self.temp_dir)
        
        # Test single item access
        features, adj, label = dataset[0]
        
        # Check shapes
        self.assertEqual(features.shape, (self.seq_length, self.num_nodes, self.num_features))
        self.assertEqual(adj.shape, (self.num_nodes, self.num_nodes))
        self.assertIsInstance(label, (int, mx.array))
        
    def test_dataset_batching(self):
        """Test batch creation from dataset."""
        self.create_dummy_dataset()
        
        dataset = GraphDataset(self.temp_dir)
        batch_size = 8
        
        # Get batch
        batch_indices = list(range(batch_size))
        batch_data = dataset.get_batch(batch_indices)
        
        features_batch, adj_batch, labels_batch = batch_data
        
        # Check batch shapes
        self.assertEqual(features_batch.shape[0], batch_size)
        self.assertEqual(labels_batch.shape[0], batch_size)
        
    def test_data_augmentation(self):
        """Test data augmentation in dataset."""
        self.create_dummy_dataset()
        
        def augment_fn(features, adj, label):
            # Simple augmentation: add noise
            features = features + mx.random.normal(features.shape) * 0.01
            return features, adj, label
        
        dataset = GraphDataset(self.temp_dir, transform=augment_fn)
        
        # Get same sample twice
        features1, _, _ = dataset[0]
        features2, _, _ = dataset[0]
        
        # Should be different due to augmentation
        diff = mx.sum(mx.abs(features1 - features2))
        self.assertGreater(float(diff), 0.0)


class TestGraphBatchSampler(unittest.TestCase):
    """Test cases for GraphBatchSampler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_samples = 100
        self.num_classes = 5
        self.samples_per_class = 4
        
        # Create labels
        self.labels = np.array([i % self.num_classes for i in range(self.num_samples)])
        
    def test_balanced_batch_sampler(self):
        """Test balanced batch sampling."""
        sampler = GraphBatchSampler(
            labels=self.labels,
            num_classes=self.num_classes,
            samples_per_class=self.samples_per_class,
            num_batches=10
        )
        
        batch_size = self.num_classes * self.samples_per_class
        
        for batch_indices in sampler:
            # Check batch size
            self.assertEqual(len(batch_indices), batch_size)
            
            # Check class balance
            batch_labels = self.labels[batch_indices]
            unique, counts = np.unique(batch_labels, return_counts=True)
            
            # Each class should have equal samples
            for count in counts:
                self.assertEqual(count, self.samples_per_class)
                
    def test_sampler_exhaustion(self):
        """Test sampler behavior when exhausted."""
        num_batches = 5
        sampler = GraphBatchSampler(
            labels=self.labels,
            num_classes=self.num_classes,
            samples_per_class=self.samples_per_class,
            num_batches=num_batches
        )
        
        # Count batches
        batch_count = sum(1 for _ in sampler)
        self.assertEqual(batch_count, num_batches)


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_nodes = 15
        
    def test_adjacency_normalization(self):
        """Test adjacency matrix normalization."""
        # Create random adjacency matrix
        adj = np.random.rand(self.num_nodes, self.num_nodes)
        adj = (adj + adj.T) / 2  # Make symmetric
        
        # Normalize
        adj_norm = normalize_adjacency_matrix(mx.array(adj))
        
        # Check properties
        self.assertEqual(adj_norm.shape, (self.num_nodes, self.num_nodes))
        
        # Row sums should be close to 1
        row_sums = mx.sum(adj_norm, axis=1)
        for i in range(self.num_nodes):
            self.assertAlmostEqual(float(row_sums[i]), 1.0, places=5)
            
    def test_laplacian_computation(self):
        """Test graph Laplacian computation."""
        # Create adjacency matrix
        adj = np.eye(self.num_nodes)
        
        # Add some edges
        adj[0, 1] = adj[1, 0] = 1
        adj[1, 2] = adj[2, 1] = 1
        
        # Compute Laplacian
        laplacian = compute_laplacian(mx.array(adj), normalized=True)
        
        # Check shape
        self.assertEqual(laplacian.shape, (self.num_nodes, self.num_nodes))
        
        # Check symmetry
        diff = mx.sum(mx.abs(laplacian - laplacian.T))
        self.assertAlmostEqual(float(diff), 0.0, places=5)
        
    def test_temporal_augmentation(self):
        """Test temporal signal augmentation."""
        seq_length = 30
        num_features = 8
        
        # Create temporal signal
        signal = mx.random.normal((seq_length, self.num_nodes, num_features))
        
        # Apply augmentation
        augmented = temporal_signal_augmentation(
            signal,
            noise_level=0.1,
            time_shift=True,
            scale_range=(0.9, 1.1)
        )
        
        # Check shape preserved
        self.assertEqual(augmented.shape, signal.shape)
        
        # Check signal is modified
        diff = mx.sum(mx.abs(augmented - signal))
        self.assertGreater(float(diff), 0.0)
        
    def test_sliding_windows(self):
        """Test sliding window creation."""
        total_length = 100
        window_size = 20
        stride = 10
        
        # Create long time series
        data = mx.random.normal((total_length, self.num_nodes, 8))
        
        # Create sliding windows
        windows = create_sliding_windows(data, window_size, stride)
        
        # Check number of windows
        expected_windows = (total_length - window_size) // stride + 1
        self.assertEqual(len(windows), expected_windows)
        
        # Check window shapes
        for window in windows:
            self.assertEqual(window.shape, (window_size, self.num_nodes, 8))


class TestDataPipeline(unittest.TestCase):
    """Integration tests for complete data pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_end_to_end_pipeline(self):
        """Test complete data loading and preprocessing pipeline."""
        # Create test data
        num_samples = 50
        num_nodes = 10
        num_features = 4
        seq_length = 25
        num_classes = 3
        
        # Generate and save data
        adj = np.eye(num_nodes) + np.random.rand(num_nodes, num_nodes) * 0.2
        adj = (adj + adj.T) / 2
        
        features = np.random.randn(num_samples, seq_length, num_nodes, num_features)
        labels = np.array([i % num_classes for i in range(num_samples)])
        
        np.save(os.path.join(self.temp_dir, 'adjacency.npy'), adj)
        np.save(os.path.join(self.temp_dir, 'features.npy'), features)
        np.save(os.path.join(self.temp_dir, 'labels.npy'), labels)
        
        metadata = {
            'num_samples': num_samples,
            'num_nodes': num_nodes,
            'num_features': num_features,
            'seq_length': seq_length,
            'num_classes': num_classes
        }
        
        with open(os.path.join(self.temp_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # Create dataset with preprocessing
        def preprocess_fn(features, adj, label):
            # Normalize adjacency
            adj = normalize_adjacency_matrix(adj)
            
            # Add noise augmentation
            if np.random.rand() > 0.5:
                features = temporal_signal_augmentation(features, noise_level=0.05)
                
            return features, adj, label
        
        dataset = GraphDataset(self.temp_dir, transform=preprocess_fn)
        
        # Create batch sampler
        sampler = GraphBatchSampler(
            labels=labels,
            num_classes=num_classes,
            samples_per_class=3,
            num_batches=5
        )
        
        # Process batches
        for batch_indices in sampler:
            batch_data = dataset.get_batch(batch_indices)
            features_batch, adj_batch, labels_batch = batch_data
            
            # Verify batch
            self.assertEqual(features_batch.shape[0], num_classes * 3)
            self.assertEqual(features_batch.shape[1], seq_length)
            self.assertEqual(features_batch.shape[2], num_nodes)
            self.assertEqual(features_batch.shape[3], num_features)
            
    def test_memory_efficient_loading(self):
        """Test memory-efficient data loading."""
        # This would test lazy loading, memory mapping, etc.
        # For now, just verify basic functionality
        pass


if __name__ == "__main__":
    unittest.main()