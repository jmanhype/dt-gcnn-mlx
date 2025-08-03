"""
DT-GCNN Model Architecture in MLX

This module implements the Dual-Transformer Graph Convolutional Network (DT-GCNN)
for text classification using Apple's MLX framework.

Architecture Overview:
1. Embedding Layer: Token to vector conversion
2. Bidirectional GRU Encoder: Manual forward/backward implementation
3. 1D CNN Feature Extractor: With proper channel handling (channels-last)
4. Global Max Pooling: Aggregate features across sequence
5. Projection Head: With L2 normalization for representation learning
6. Classification Head: Final output layer

All components follow MLX conventions with lazy initialization and channels-last format.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple, List
import math


class GRUCell(nn.Module):
    """Single GRU cell implementation for MLX."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input-to-hidden weights for all gates
        self.W_ir = nn.Linear(input_size, hidden_size, bias=True)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=True)
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        
        # Hidden-to-hidden weights for all gates
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=True)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=True)
        
    def __call__(self, x: mx.array, h_prev: mx.array) -> mx.array:
        """
        Forward pass of GRU cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h_prev: Previous hidden state of shape (batch_size, hidden_size)
            
        Returns:
            h_new: New hidden state of shape (batch_size, hidden_size)
        """
        # Reset gate
        r = mx.sigmoid(self.W_ir(x) + self.W_hr(h_prev))
        
        # Update gate
        z = mx.sigmoid(self.W_iz(x) + self.W_hz(h_prev))
        
        # New gate
        n = mx.tanh(self.W_in(x) + r * self.W_hn(h_prev))
        
        # Update hidden state
        h_new = (1 - z) * n + z * h_prev
        
        return h_new


class BidirectionalGRU(nn.Module):
    """Bidirectional GRU implementation with manual forward/backward passes."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create forward and backward GRU cells for each layer
        self.forward_cells = []
        self.backward_cells = []
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else 2 * hidden_size
            self.forward_cells.append(GRUCell(layer_input_size, hidden_size))
            self.backward_cells.append(GRUCell(layer_input_size, hidden_size))
            
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of bidirectional GRU.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            output: Concatenated forward and backward outputs of shape 
                   (batch_size, seq_len, 2 * hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through each layer
        layer_input = x
        for layer_idx in range(self.num_layers):
            # Forward pass
            h_forward = mx.zeros((batch_size, self.hidden_size))
            forward_outputs = []
            
            for t in range(seq_len):
                h_forward = self.forward_cells[layer_idx](layer_input[:, t, :], h_forward)
                forward_outputs.append(h_forward)
            
            # Backward pass
            h_backward = mx.zeros((batch_size, self.hidden_size))
            backward_outputs = []
            
            for t in range(seq_len - 1, -1, -1):
                h_backward = self.backward_cells[layer_idx](layer_input[:, t, :], h_backward)
                backward_outputs.append(h_backward)
            
            # Reverse backward outputs to match sequence order
            backward_outputs = backward_outputs[::-1]
            
            # Concatenate forward and backward outputs
            forward_tensor = mx.stack(forward_outputs, axis=1)
            backward_tensor = mx.stack(backward_outputs, axis=1)
            layer_output = mx.concatenate([forward_tensor, backward_tensor], axis=2)
            
            # Use concatenated output as input to next layer
            layer_input = layer_output
        
        return layer_output


class Conv1DBlock(nn.Module):
    """1D Convolutional block with proper channel handling for MLX."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding)
        self.bn = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of convolutional block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
            
        Returns:
            output: Tensor of shape (batch_size, new_seq_len, out_channels)
        """
        # Conv1d in MLX expects (batch_size, seq_len, channels)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GlobalMaxPool1D(nn.Module):
    """Global max pooling over sequence dimension."""
    
    def __call__(self, x: mx.array) -> mx.array:
        """
        Apply global max pooling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, channels)
            
        Returns:
            output: Tensor of shape (batch_size, channels)
        """
        return mx.max(x, axis=1)


class ProjectionHead(nn.Module):
    """Projection head with L2 normalization for representation learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm(output_dim)
        
    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass of projection head.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            output: L2-normalized tensor of shape (batch_size, output_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        
        # L2 normalization
        x = x / (mx.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        
        return x


class DTGCNN(nn.Module):
    """
    Dual-Transformer Graph Convolutional Network (DT-GCNN) for text classification.
    
    This model combines bidirectional GRU encoding with 1D CNN feature extraction
    and includes both a projection head for representation learning and a 
    classification head for final predictions.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Hidden dimension for GRU
        num_classes: Number of output classes
        num_filters: Number of CNN filters
        filter_sizes: List of CNN filter sizes
        num_gru_layers: Number of bidirectional GRU layers
        projection_dim: Dimension of projection head output
        dropout_rate: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_filters: int = 100,
        filter_sizes: List[int] = [3, 4, 5],
        num_gru_layers: int = 2,
        projection_dim: int = 128,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_gru_layers = num_gru_layers
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        # Bidirectional GRU encoder
        self.bigru = BidirectionalGRU(embedding_dim, hidden_dim, num_gru_layers)
        self.gru_dropout = nn.Dropout(dropout_rate)
        
        # 1D CNN feature extractors with different kernel sizes
        self.conv_blocks = []
        for kernel_size in filter_sizes:
            conv_block = Conv1DBlock(
                in_channels=2 * hidden_dim,  # Bidirectional output
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Same padding
            )
            self.conv_blocks.append(conv_block)
        
        # Global max pooling
        self.global_pool = GlobalMaxPool1D()
        
        # Calculate total feature dimension after concatenation
        total_cnn_features = num_filters * len(filter_sizes)
        total_features = total_cnn_features + 2 * hidden_dim  # CNN + GRU features
        
        # Projection head for representation learning
        self.projection_head = ProjectionHead(
            input_dim=total_features,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.BatchNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        # Initialize embedding with normal distribution
        embedding_std = 1.0 / math.sqrt(self.embedding_dim)
        self.embedding.weight = mx.random.normal(
            shape=(self.vocab_size, self.embedding_dim),
            scale=embedding_std
        )
        
    def __call__(
        self, 
        input_ids: mx.array,
        return_embeddings: bool = False
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Forward pass of DT-GCNN model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            return_embeddings: Whether to return projection embeddings
            
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
            embeddings: Projection embeddings if requested (batch_size, projection_dim)
        """
        # Embedding lookup
        x = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        x = self.embedding_dropout(x)
        
        # Bidirectional GRU encoding
        gru_output = self.bigru(x)  # (batch_size, seq_len, 2 * hidden_dim)
        gru_output = self.gru_dropout(gru_output)
        
        # Apply CNN blocks with different kernel sizes
        cnn_outputs = []
        for conv_block in self.conv_blocks:
            conv_out = conv_block(gru_output)
            pooled = self.global_pool(conv_out)
            cnn_outputs.append(pooled)
        
        # Concatenate all CNN outputs
        cnn_features = mx.concatenate(cnn_outputs, axis=1)
        
        # Global max pooling on GRU output for additional features
        gru_pooled = self.global_pool(gru_output)
        
        # Concatenate all features
        combined_features = mx.concatenate([cnn_features, gru_pooled], axis=1)
        
        # Classification logits
        logits = self.classifier(combined_features)
        
        # Projection embeddings (if requested)
        embeddings = None
        if return_embeddings:
            embeddings = self.projection_head(combined_features)
        
        return logits, embeddings
    
    def get_embeddings(self, input_ids: mx.array) -> mx.array:
        """
        Get normalized embeddings from the projection head.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            embeddings: L2-normalized embeddings of shape (batch_size, projection_dim)
        """
        _, embeddings = self(input_ids, return_embeddings=True)
        return embeddings