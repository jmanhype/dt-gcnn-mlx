"""
Utility functions for DT-GCNN model management in MLX.

This module provides helper functions for model initialization,
parameter management, and model inspection.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple, Any
import json


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: MLX model instance
        
    Returns:
        Dictionary with parameter counts by category
    """
    total_params = 0
    trainable_params = 0
    param_details = {}
    
    # Recursively count parameters
    for name, param in model.named_parameters():
        param_count = param.size
        total_params += param_count
        trainable_params += param_count  # In MLX, all params are trainable by default
        
        # Group by module type
        module_name = name.split('.')[0]
        if module_name not in param_details:
            param_details[module_name] = 0
        param_details[module_name] += param_count
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': 0,
        'by_module': param_details
    }


def get_model_summary(model: nn.Module, input_shape: Tuple[int, int]) -> str:
    """
    Generate a summary of the model architecture.
    
    Args:
        model: MLX model instance
        input_shape: Expected input shape (batch_size, seq_len)
        
    Returns:
        String containing model summary
    """
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("Model: DT-GCNN")
    summary_lines.append("=" * 80)
    
    # Model configuration
    if hasattr(model, 'vocab_size'):
        summary_lines.append(f"Vocabulary Size: {model.vocab_size:,}")
        summary_lines.append(f"Embedding Dimension: {model.embedding_dim}")
        summary_lines.append(f"Hidden Dimension: {model.hidden_dim}")
        summary_lines.append(f"Number of Classes: {model.num_classes}")
        summary_lines.append(f"Number of Filters: {model.num_filters}")
        summary_lines.append(f"Filter Sizes: {model.filter_sizes}")
        summary_lines.append(f"GRU Layers: {model.num_gru_layers}")
        summary_lines.append(f"Projection Dimension: {model.projection_dim}")
        summary_lines.append(f"Dropout Rate: {model.dropout_rate}")
    
    summary_lines.append("-" * 80)
    
    # Parameter count
    param_info = count_parameters(model)
    summary_lines.append(f"Total Parameters: {param_info['total']:,}")
    summary_lines.append(f"Trainable Parameters: {param_info['trainable']:,}")
    summary_lines.append(f"Non-trainable Parameters: {param_info['non_trainable']:,}")
    
    summary_lines.append("-" * 80)
    summary_lines.append("Parameters by Module:")
    for module, count in param_info['by_module'].items():
        summary_lines.append(f"  {module}: {count:,}")
    
    summary_lines.append("=" * 80)
    
    return "\n".join(summary_lines)


def save_model_config(model: nn.Module, filepath: str):
    """
    Save model configuration to a JSON file.
    
    Args:
        model: MLX model instance
        filepath: Path to save configuration
    """
    config = {
        'model_type': 'DT-GCNN',
        'vocab_size': model.vocab_size,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'num_classes': model.num_classes,
        'num_filters': model.num_filters,
        'filter_sizes': model.filter_sizes,
        'num_gru_layers': model.num_gru_layers,
        'projection_dim': model.projection_dim,
        'dropout_rate': model.dropout_rate
    }
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_model_config(filepath: str) -> Dict[str, Any]:
    """
    Load model configuration from a JSON file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Dictionary containing model configuration
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def initialize_embeddings_from_pretrained(
    model: nn.Module,
    pretrained_embeddings: mx.array,
    word_to_idx: Dict[str, int],
    freeze: bool = False
) -> None:
    """
    Initialize embedding layer with pretrained embeddings.
    
    Args:
        model: MLX model instance with embedding layer
        pretrained_embeddings: Pretrained embedding matrix
        word_to_idx: Vocabulary mapping
        freeze: Whether to freeze embedding weights
    """
    vocab_size, embedding_dim = pretrained_embeddings.shape
    
    # Ensure dimensions match
    assert model.embedding.weight.shape == pretrained_embeddings.shape, \
        f"Embedding shapes don't match: {model.embedding.weight.shape} vs {pretrained_embeddings.shape}"
    
    # Copy pretrained embeddings
    model.embedding.weight = pretrained_embeddings
    
    # MLX doesn't have a direct freeze mechanism, but we can track this
    # for training logic to skip embedding updates if needed
    model.embedding._freeze = freeze


def compute_output_shape(model: nn.Module, input_shape: Tuple[int, int]) -> Dict[str, Tuple]:
    """
    Compute output shapes for different stages of the model.
    
    Args:
        model: MLX model instance
        input_shape: Input shape (batch_size, seq_len)
        
    Returns:
        Dictionary mapping layer names to output shapes
    """
    batch_size, seq_len = input_shape
    
    shapes = {
        'input': input_shape,
        'embedding': (batch_size, seq_len, model.embedding_dim),
        'bigru': (batch_size, seq_len, 2 * model.hidden_dim),
    }
    
    # CNN output shapes (after global pooling)
    for i, kernel_size in enumerate(model.filter_sizes):
        shapes[f'cnn_{kernel_size}'] = (batch_size, model.num_filters)
    
    # Combined features shape
    total_cnn_features = model.num_filters * len(model.filter_sizes)
    total_features = total_cnn_features + 2 * model.hidden_dim
    shapes['combined_features'] = (batch_size, total_features)
    
    # Final outputs
    shapes['logits'] = (batch_size, model.num_classes)
    shapes['embeddings'] = (batch_size, model.projection_dim)
    
    return shapes


def create_model_checkpoint(model: nn.Module) -> Dict[str, Any]:
    """
    Create a checkpoint dictionary containing model state and configuration.
    
    Args:
        model: MLX model instance
        
    Returns:
        Dictionary containing model state and configuration
    """
    checkpoint = {
        'model_state': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'hidden_dim': model.hidden_dim,
            'num_classes': model.num_classes,
            'num_filters': model.num_filters,
            'filter_sizes': model.filter_sizes,
            'num_gru_layers': model.num_gru_layers,
            'projection_dim': model.projection_dim,
            'dropout_rate': model.dropout_rate
        }
    }
    
    return checkpoint


def load_model_from_checkpoint(checkpoint: Dict[str, Any], model_class) -> nn.Module:
    """
    Load a model from a checkpoint dictionary.
    
    Args:
        checkpoint: Checkpoint dictionary
        model_class: Model class to instantiate
        
    Returns:
        Loaded model instance
    """
    # Create model with saved configuration
    model = model_class(**checkpoint['model_config'])
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state'])
    
    return model