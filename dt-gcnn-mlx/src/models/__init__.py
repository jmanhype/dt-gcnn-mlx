"""
DT-GCNN Models Module for MLX

This module provides the Dual-Transformer Graph Convolutional Network (DT-GCNN)
implementation for text classification using Apple's MLX framework.
"""

from .dt_gcnn import (
    DTGCNN,
    BidirectionalGRU,
    GRUCell,
    Conv1DBlock,
    GlobalMaxPool1D,
    ProjectionHead
)

# For backward compatibility
ModelConfig = dict

__all__ = [
    'DTGCNN',
    'BidirectionalGRU',
    'GRUCell', 
    'Conv1DBlock',
    'GlobalMaxPool1D',
    'ProjectionHead',
    'ModelConfig'
]

# Model configuration presets
MODEL_CONFIGS = {
    'small': {
        'embedding_dim': 128,
        'hidden_dim': 64,
        'num_filters': 50,
        'filter_sizes': [3, 4, 5],
        'num_gru_layers': 1,
        'projection_dim': 64
    },
    'base': {
        'embedding_dim': 256,
        'hidden_dim': 128,
        'num_filters': 100,
        'filter_sizes': [3, 4, 5],
        'num_gru_layers': 2,
        'projection_dim': 128
    },
    'large': {
        'embedding_dim': 512,
        'hidden_dim': 256,
        'num_filters': 200,
        'filter_sizes': [2, 3, 4, 5],
        'num_gru_layers': 3,
        'projection_dim': 256
    }
}

def create_model(preset: str, vocab_size: int, num_classes: int, **kwargs):
    """
    Create a DT-GCNN model with a predefined configuration.
    
    Args:
        preset: Configuration preset ('small', 'base', or 'large')
        vocab_size: Size of the vocabulary
        num_classes: Number of output classes
        **kwargs: Additional arguments to override preset values
        
    Returns:
        DTGCNN: Configured model instance
    """
    if preset not in MODEL_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[preset].copy()
    config.update(kwargs)
    
    return DTGCNN(
        vocab_size=vocab_size,
        num_classes=num_classes,
        **config
    )