"""
Sample data generation for DT-GCNN MLX demos
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def create_sample_data(
    output_dir: Path,
    num_samples: int = 1000,
    num_classes: int = 4,
    sequence_length: int = 128,
    vocab_size: int = 5000
) -> None:
    """
    Generate synthetic triplet data for demo purposes.
    
    Args:
        output_dir: Directory to save the data
        num_samples: Number of triplet samples to generate
        num_classes: Number of classes
        sequence_length: Length of each sequence
        vocab_size: Vocabulary size for token generation
    """
    print(f"Generating {num_samples} triplet samples...")
    
    triplets = []
    
    for i in range(num_samples):
        # Generate class label
        class_id = np.random.randint(0, num_classes)
        
        # Generate anchor sequence
        anchor = np.random.randint(1, vocab_size, sequence_length).tolist()
        
        # Generate positive (similar to anchor, same class)
        positive = anchor.copy()
        # Add some noise but keep it similar
        noise_positions = np.random.choice(sequence_length, size=sequence_length//4, replace=False)
        for pos in noise_positions:
            positive[pos] = np.random.randint(1, vocab_size)
        
        # Generate negative (different class)
        negative = np.random.randint(1, vocab_size, sequence_length).tolist()
        
        triplet = {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "label": class_id
        }
        triplets.append(triplet)
    
    # Save to JSON file
    output_file = output_dir / "triplets.json"
    with open(output_file, 'w') as f:
        json.dump(triplets, f, indent=2)
    
    print(f"✓ Saved {len(triplets)} triplets to {output_file}")


def load_sample_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load triplet data from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)