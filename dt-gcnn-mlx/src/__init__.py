"""
DT-GCNN MLX: Dynamic Triplet GRU-CNN for Text Classification
Optimized for Apple Silicon using MLX Framework
"""

__version__ = "0.1.0"
__author__ = "DT-GCNN MLX Team"

# Import main components
from .models import DTGCNN, create_model, ModelConfig
from .losses import TripletLoss, BatchHardMiner, OnlineTripletMiner
from .data import SentencePieceTokenizer, TripletDataset, create_data_loader
from .training import DT_GCNN_Trainer, train_model

__all__ = [
    # Models
    "DTGCNN",
    "create_model",
    "ModelConfig",
    
    # Losses
    "TripletLoss",
    "BatchHardMiner",
    "OnlineTripletMiner",
    
    # Data
    "SentencePieceTokenizer",
    "TripletDataset",
    "create_data_loader",
    
    # Training
    "DT_GCNN_Trainer",
    "train_model",
]