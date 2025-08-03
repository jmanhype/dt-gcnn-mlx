"""Tests for DT-GCNN model architecture in MLX."""

import unittest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Tuple

# Import from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dt_gcnn import DTGCNN, GraphConvLayer, DilatedTemporalConvLayer


class TestGraphConvLayer(unittest.TestCase):
    """Test cases for Graph Convolutional Layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 16
        self.out_features = 32
        self.num_nodes = 10
        
        # Create sample adjacency matrix (normalized)
        adj = np.eye(self.num_nodes) + np.random.rand(self.num_nodes, self.num_nodes) * 0.1
        self.adj_matrix = mx.array(adj / adj.sum(axis=1, keepdims=True))
        
    def test_layer_initialization(self):
        """Test GraphConvLayer initialization."""
        layer = GraphConvLayer(self.in_features, self.out_features)
        
        # Check weight shapes
        self.assertEqual(layer.weight.shape, (self.in_features, self.out_features))
        if hasattr(layer, 'bias'):
            self.assertEqual(layer.bias.shape, (self.out_features,))
    
    def test_forward_pass(self):
        """Test forward pass of GraphConvLayer."""
        layer = GraphConvLayer(self.in_features, self.out_features)
        
        # Create input tensor [batch_size, num_nodes, in_features]
        batch_size = 4
        x = mx.random.normal((batch_size, self.num_nodes, self.in_features))
        
        # Forward pass
        output = layer(x, self.adj_matrix)
        
        # Check output shape
        expected_shape = (batch_size, self.num_nodes, self.out_features)
        self.assertEqual(output.shape, expected_shape)
        
    def test_gradient_flow(self):
        """Test gradient flow through GraphConvLayer."""
        layer = GraphConvLayer(self.in_features, self.out_features)
        
        def loss_fn(x, adj):
            output = layer(x, adj)
            return mx.mean(output ** 2)
        
        # Create input
        x = mx.random.normal((2, self.num_nodes, self.in_features))
        
        # Compute gradients
        loss_grad_fn = mx.grad(loss_fn)
        grads = loss_grad_fn(x, self.adj_matrix)
        
        # Check gradient shape matches input
        self.assertEqual(grads.shape, x.shape)


class TestDilatedTemporalConvLayer(unittest.TestCase):
    """Test cases for Dilated Temporal Convolutional Layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_channels = 32
        self.out_channels = 64
        self.kernel_size = 3
        self.dilation = 2
        self.seq_length = 20
        
    def test_layer_initialization(self):
        """Test DilatedTemporalConvLayer initialization."""
        layer = DilatedTemporalConvLayer(
            self.in_channels, 
            self.out_channels, 
            self.kernel_size, 
            self.dilation
        )
        
        # Check conv layer exists
        self.assertIsInstance(layer.conv, nn.Conv1d)
        
    def test_forward_pass(self):
        """Test forward pass of DilatedTemporalConvLayer."""
        layer = DilatedTemporalConvLayer(
            self.in_channels, 
            self.out_channels, 
            self.kernel_size, 
            self.dilation
        )
        
        # Create input tensor [batch_size, seq_length, in_channels]
        batch_size = 4
        x = mx.random.normal((batch_size, self.seq_length, self.in_channels))
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        # Output length depends on padding
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[2], self.out_channels)
        
    def test_receptive_field(self):
        """Test receptive field calculation."""
        dilations = [1, 2, 4, 8]
        kernel_size = 3
        
        for dilation in dilations:
            layer = DilatedTemporalConvLayer(
                self.in_channels, 
                self.out_channels, 
                kernel_size, 
                dilation
            )
            
            # Receptive field = (kernel_size - 1) * dilation + 1
            expected_rf = (kernel_size - 1) * dilation + 1
            
            # Can test this by checking the actual convolution behavior
            x = mx.zeros((1, expected_rf + 10, self.in_channels))
            x[:, expected_rf // 2, :] = 1.0  # Single spike
            
            output = layer(x)
            # Output should show influence at expected positions


class TestDTGCNN(unittest.TestCase):
    """Test cases for complete DT-GCNN model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.num_nodes = 15
        self.input_dim = 8
        self.hidden_dims = [16, 32, 64]
        self.temporal_kernel_size = 3
        self.dilations = [1, 2, 4]
        self.num_classes = 5
        self.seq_length = 30
        
        # Create normalized adjacency matrix
        adj = np.eye(self.num_nodes) + np.random.rand(self.num_nodes, self.num_nodes) * 0.2
        self.adj_matrix = mx.array(adj / adj.sum(axis=1, keepdims=True))
        
    def test_model_initialization(self):
        """Test DTGCNN initialization."""
        model = DTGCNN(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            temporal_kernel_size=self.temporal_kernel_size,
            dilations=self.dilations,
            num_classes=self.num_classes,
            dropout=0.5
        )
        
        # Check layer counts
        self.assertEqual(len(model.graph_convs), len(self.hidden_dims))
        self.assertEqual(len(model.temporal_convs), len(self.dilations))
        
    def test_forward_pass(self):
        """Test forward pass of complete model."""
        model = DTGCNN(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            temporal_kernel_size=self.temporal_kernel_size,
            dilations=self.dilations,
            num_classes=self.num_classes,
            dropout=0.5
        )
        
        # Create input tensor [batch_size, seq_length, num_nodes, input_dim]
        batch_size = 2
        x = mx.random.normal((batch_size, self.seq_length, self.num_nodes, self.input_dim))
        
        # Forward pass
        output = model(x, self.adj_matrix, training=True)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.num_classes))
        
    def test_embedding_extraction(self):
        """Test feature embedding extraction."""
        model = DTGCNN(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            temporal_kernel_size=self.temporal_kernel_size,
            dilations=self.dilations,
            num_classes=self.num_classes,
            dropout=0.5
        )
        
        batch_size = 2
        x = mx.random.normal((batch_size, self.seq_length, self.num_nodes, self.input_dim))
        
        # Get embeddings
        embeddings = model.get_embeddings(x, self.adj_matrix)
        
        # Check embedding dimension
        self.assertEqual(len(embeddings.shape), 2)
        self.assertEqual(embeddings.shape[0], batch_size)
        
    def test_model_with_different_configs(self):
        """Test model with various configurations."""
        configs = [
            {"hidden_dims": [32, 64], "dilations": [1, 2]},
            {"hidden_dims": [16, 32, 64, 128], "dilations": [1, 2, 4, 8]},
            {"hidden_dims": [64], "dilations": [1]},
        ]
        
        for config in configs:
            model = DTGCNN(
                num_nodes=self.num_nodes,
                input_dim=self.input_dim,
                hidden_dims=config["hidden_dims"],
                temporal_kernel_size=self.temporal_kernel_size,
                dilations=config["dilations"],
                num_classes=self.num_classes,
                dropout=0.3
            )
            
            x = mx.random.normal((1, self.seq_length, self.num_nodes, self.input_dim))
            output = model(x, self.adj_matrix, training=False)
            
            self.assertEqual(output.shape, (1, self.num_classes))
            
    def test_training_vs_eval_mode(self):
        """Test model behavior in training vs evaluation mode."""
        model = DTGCNN(
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            temporal_kernel_size=self.temporal_kernel_size,
            dilations=self.dilations,
            num_classes=self.num_classes,
            dropout=0.5
        )
        
        x = mx.random.normal((2, self.seq_length, self.num_nodes, self.input_dim))
        
        # Training mode (with dropout)
        model.eval()  # First set to eval
        model.train()  # Then to train
        output_train = model(x, self.adj_matrix, training=True)
        
        # Eval mode (no dropout)
        model.eval()
        output_eval = model(x, self.adj_matrix, training=False)
        
        # Both should have same shape
        self.assertEqual(output_train.shape, output_eval.shape)


if __name__ == "__main__":
    unittest.main()