"""
Data loading utilities for DT-GCNN MLX
"""

import json
import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple


class TripletDataset:
    """Dataset for triplet learning with MLX."""
    
    def __init__(self, data_path: str):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to triplets JSON file
        """
        self.data_path = Path(data_path)
        self.triplets = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load triplet data from JSON file."""
        with open(self.data_path, 'r') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """Get a triplet sample."""
        triplet = self.triplets[idx]
        
        return {
            "anchor": mx.array(triplet["anchor"], dtype=mx.int32),
            "positive": mx.array(triplet["positive"], dtype=mx.int32), 
            "negative": mx.array(triplet["negative"], dtype=mx.int32),
            "label": mx.array(triplet["label"], dtype=mx.int32)
        }


def create_data_loader(
    dataset: TripletDataset,
    batch_size: int = 32,
    shuffle: bool = True
) -> Iterator[Dict[str, mx.array]]:
    """
    Create a simple data loader for the dataset.
    
    Args:
        dataset: TripletDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Yields:
        Batched data dictionaries
    """
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        
        # Collect batch data
        batch_anchors = []
        batch_positives = []
        batch_negatives = []
        batch_labels = []
        
        for idx in batch_indices:
            sample = dataset[idx]
            batch_anchors.append(sample["anchor"])
            batch_positives.append(sample["positive"])
            batch_negatives.append(sample["negative"])
            batch_labels.append(sample["label"])
        
        # Stack into batches
        yield {
            "anchor": mx.stack(batch_anchors),
            "positive": mx.stack(batch_positives),
            "negative": mx.stack(batch_negatives),
            "label": mx.stack(batch_labels)
        }