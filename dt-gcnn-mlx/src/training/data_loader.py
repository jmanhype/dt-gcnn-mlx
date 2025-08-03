"""
Data loader for DT-GCNN training
Handles mesh data loading, augmentation, and batching
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import json
import h5py
import logging
from dataclasses import dataclass
import random
from collections import defaultdict


@dataclass
class MeshSample:
    """Single mesh sample"""
    coordinates: np.ndarray  # (num_vertices, 3)
    features: np.ndarray     # (num_vertices, 6) - coordinates + normals
    label: int
    subject_id: str
    mesh_id: str


class MeshAugmentation:
    """Data augmentation for 3D meshes"""
    
    def __init__(
        self,
        rotation_range: float = 30.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.01,
        flip_probability: float = 0.5
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.flip_probability = flip_probability
    
    def __call__(self, sample: MeshSample) -> MeshSample:
        """Apply augmentations to mesh sample"""
        coords = sample.coordinates.copy()
        normals = sample.features[:, 3:6].copy()
        
        # Random rotation
        if self.rotation_range > 0:
            coords, normals = self._random_rotation(coords, normals)
        
        # Random scaling
        if self.scale_range[0] < 1.0 or self.scale_range[1] > 1.0:
            scale = np.random.uniform(*self.scale_range)
            coords *= scale
        
        # Random noise
        if self.noise_std > 0:
            coords += np.random.normal(0, self.noise_std, coords.shape)
        
        # Random flip
        if np.random.random() < self.flip_probability:
            coords[:, 0] *= -1  # Flip x-axis
            normals[:, 0] *= -1
        
        # Reconstruct features
        features = np.concatenate([coords, normals], axis=1)
        
        return MeshSample(
            coordinates=coords,
            features=features,
            label=sample.label,
            subject_id=sample.subject_id,
            mesh_id=sample.mesh_id
        )
    
    def _random_rotation(
        self,
        coords: np.ndarray,
        normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random rotation around y-axis"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.radians(angle)
        
        # Rotation matrix around y-axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        # Apply rotation
        coords_rotated = coords @ rotation_matrix.T
        normals_rotated = normals @ rotation_matrix.T
        
        return coords_rotated, normals_rotated


class TripletSampler:
    """Samples triplets for triplet loss"""
    
    def __init__(self, samples: List[MeshSample], num_triplets_per_batch: int = 16):
        self.samples = samples
        self.num_triplets_per_batch = num_triplets_per_batch
        
        # Group samples by label
        self.label_to_indices = defaultdict(list)
        for idx, sample in enumerate(samples):
            self.label_to_indices[sample.label].append(idx)
        
        # Filter out classes with less than 2 samples
        self.valid_labels = [
            label for label, indices in self.label_to_indices.items()
            if len(indices) >= 2
        ]
    
    def sample_triplets(
        self,
        batch_indices: List[int]
    ) -> Optional[Tuple[List[int], List[int], List[int]]]:
        """Sample triplet indices for a batch"""
        if len(self.valid_labels) < 2:
            return None
        
        anchor_indices = []
        positive_indices = []
        negative_indices = []
        
        # Sample triplets
        for _ in range(self.num_triplets_per_batch):
            # Select anchor from batch
            anchor_idx = random.choice(batch_indices)
            anchor_label = self.samples[anchor_idx].label
            
            # Skip if not enough positive samples
            positive_candidates = [
                idx for idx in self.label_to_indices[anchor_label]
                if idx != anchor_idx
            ]
            if not positive_candidates:
                continue
            
            # Select positive (same class, different sample)
            positive_idx = random.choice(positive_candidates)
            
            # Select negative (different class)
            negative_label = random.choice([
                label for label in self.valid_labels
                if label != anchor_label
            ])
            negative_idx = random.choice(self.label_to_indices[negative_label])
            
            anchor_indices.append(anchor_idx)
            positive_indices.append(positive_idx)
            negative_indices.append(negative_idx)
        
        if not anchor_indices:
            return None
        
        return anchor_indices, positive_indices, negative_indices


class MeshDataset:
    """Dataset for mesh data"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        augmentation: Optional[MeshAugmentation] = None,
        cache_data: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augmentation = augmentation
        self.cache_data = cache_data
        
        # Load metadata
        self.samples = self._load_metadata()
        
        # Cache for loaded data
        self._cache = {} if cache_data else None
        
        # Setup triplet sampler for training
        self.triplet_sampler = TripletSampler(self.samples) if split == 'train' else None
        
        logging.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_metadata(self) -> List[MeshSample]:
        """Load dataset metadata"""
        metadata_file = self.data_dir / f"{self.split}_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        samples = []
        for item in metadata['samples']:
            sample = MeshSample(
                coordinates=None,  # Loaded on demand
                features=None,
                label=item['label'],
                subject_id=item['subject_id'],
                mesh_id=item['mesh_id']
            )
            samples.append(sample)
        
        return samples
    
    def _load_mesh_data(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load mesh data from file"""
        if self._cache and idx in self._cache:
            return self._cache[idx]
        
        sample = self.samples[idx]
        
        # Load from HDF5 file
        data_file = self.data_dir / f"{self.split}_data.h5"
        
        with h5py.File(data_file, 'r') as f:
            group = f[f"{sample.subject_id}/{sample.mesh_id}"]
            coordinates = group['coordinates'][:]
            normals = group['normals'][:]
        
        features = np.concatenate([coordinates, normals], axis=1)
        
        if self._cache is not None:
            self._cache[idx] = (coordinates, features)
        
        return coordinates, features
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> MeshSample:
        """Get a single sample"""
        sample = self.samples[idx]
        
        # Load mesh data
        coordinates, features = self._load_mesh_data(idx)
        
        # Create sample
        mesh_sample = MeshSample(
            coordinates=coordinates,
            features=features,
            label=sample.label,
            subject_id=sample.subject_id,
            mesh_id=sample.mesh_id
        )
        
        # Apply augmentation if training
        if self.augmentation and self.split == 'train':
            mesh_sample = self.augmentation(mesh_sample)
        
        return mesh_sample


class MeshDataLoader:
    """Data loader for mesh dataset"""
    
    def __init__(
        self,
        dataset: MeshDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Create indices
        self.indices = list(range(len(dataset)))
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches"""
        # Shuffle indices if needed
        if self.shuffle:
            random.shuffle(self.indices)
        
        # Generate batches
        for batch_start in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[batch_start:batch_start + self.batch_size]
            
            # Skip incomplete batch if drop_last
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            
            # Load batch samples
            batch_samples = [self.dataset[idx] for idx in batch_indices]
            
            # Stack into arrays
            coordinates = np.stack([s.coordinates for s in batch_samples])
            features = np.stack([s.features for s in batch_samples])
            labels = np.array([s.label for s in batch_samples])
            
            # Convert to MLX arrays
            batch = {
                'coordinates': mx.array(coordinates),
                'features': mx.array(features),
                'labels': mx.array(labels)
            }
            
            # Add triplet indices if training
            if self.dataset.triplet_sampler and self.dataset.split == 'train':
                triplet_indices = self.dataset.triplet_sampler.sample_triplets(batch_indices)
                
                if triplet_indices:
                    batch['triplet_indices'] = tuple(
                        mx.array(indices) for indices in triplet_indices
                    )
            
            yield batch


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    cache_data: bool = False
) -> Tuple[MeshDataLoader, MeshDataLoader, MeshDataLoader]:
    """Create train, validation, and test data loaders"""
    
    # Create augmentation
    augmentation = MeshAugmentation() if augment else None
    
    # Create datasets
    train_dataset = MeshDataset(
        data_dir=data_dir,
        split='train',
        augmentation=augmentation,
        cache_data=cache_data
    )
    
    val_dataset = MeshDataset(
        data_dir=data_dir,
        split='val',
        augmentation=None,  # No augmentation for validation
        cache_data=cache_data
    )
    
    test_dataset = MeshDataset(
        data_dir=data_dir,
        split='test',
        augmentation=None,
        cache_data=cache_data
    )
    
    # Create data loaders
    train_loader = MeshDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = MeshDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = MeshDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def create_sample_data(output_dir: str, num_samples: int = 1000):
    """Create sample dataset for testing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Parameters
    num_vertices = 1723
    num_classes = 10
    
    # Create splits
    splits = {
        'train': int(0.7 * num_samples),
        'val': int(0.15 * num_samples),
        'test': int(0.15 * num_samples)
    }
    
    for split, split_size in splits.items():
        # Create metadata
        metadata = {'samples': []}
        
        # Create HDF5 file
        h5_file = output_dir / f"{split}_data.h5"
        
        with h5py.File(h5_file, 'w') as f:
            for i in range(split_size):
                subject_id = f"subject_{i:04d}"
                mesh_id = f"mesh_{i:06d}"
                label = i % num_classes
                
                # Generate random mesh data
                coordinates = np.random.randn(num_vertices, 3) * 10
                normals = np.random.randn(num_vertices, 3)
                normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
                
                # Save to HDF5
                group = f.create_group(f"{subject_id}/{mesh_id}")
                group.create_dataset('coordinates', data=coordinates)
                group.create_dataset('normals', data=normals)
                
                # Add to metadata
                metadata['samples'].append({
                    'subject_id': subject_id,
                    'mesh_id': mesh_id,
                    'label': label
                })
        
        # Save metadata
        metadata_file = output_dir / f"{split}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Created {split} split with {split_size} samples")


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data("./sample_data", num_samples=100)
    
    # Test data loader
    train_loader, val_loader, test_loader = create_data_loaders(
        "./sample_data",
        batch_size=16,
        augment=True
    )
    
    # Test iteration
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  Coordinates shape: {batch['coordinates'].shape}")
        print(f"  Features shape: {batch['features'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")
        
        if 'triplet_indices' in batch:
            anchor, positive, negative = batch['triplet_indices']
            print(f"  Triplet indices: {len(anchor)} triplets")
        
        if i >= 2:
            break