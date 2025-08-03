"""
Mining strategies for efficient triplet selection in DT-GCNN.

This module provides various mining strategies for selecting informative
triplets during training, including batch-hard mining, online selection,
and offline generation.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Tuple, Dict, Optional, Literal
from dataclasses import dataclass
import numpy as np
from .triplet_loss import pairwise_distance, euclidean_distance, cosine_distance


@dataclass
class MiningStatistics:
    """Statistics from triplet mining."""
    num_triplets: int
    num_hard_triplets: int
    num_semi_hard_triplets: int
    num_easy_triplets: int
    avg_positive_distance: float
    avg_negative_distance: float
    num_anchors_with_positives: int
    num_anchors_with_negatives: int
    mining_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "num_triplets": self.num_triplets,
            "num_hard_triplets": self.num_hard_triplets,
            "num_semi_hard_triplets": self.num_semi_hard_triplets,
            "num_easy_triplets": self.num_easy_triplets,
            "hard_ratio": self.num_hard_triplets / max(self.num_triplets, 1),
            "avg_positive_distance": self.avg_positive_distance,
            "avg_negative_distance": self.avg_negative_distance,
            "num_anchors_with_positives": self.num_anchors_with_positives,
            "num_anchors_with_negatives": self.num_anchors_with_negatives,
            "mining_time_ms": self.mining_time_ms
        }


class BatchHardMiner:
    """
    Batch-hard mining strategy for triplet selection.
    
    This miner selects the hardest positive and hardest negative for each anchor
    within a batch. It's one of the most effective mining strategies for metric learning.
    
    Example:
        >>> miner = BatchHardMiner(margin=0.5)
        >>> embeddings = model(batch)  # [32, 128]
        >>> labels = batch['labels']   # [32]
        >>> triplets, stats = miner.mine(embeddings, labels)
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        squared: bool = False,
        filter_invalid: bool = True
    ):
        """
        Initialize batch-hard miner.
        
        Args:
            margin: Margin for filtering semi-hard negatives
            metric: Distance metric to use
            squared: Use squared Euclidean distance
            filter_invalid: Filter out invalid triplets
        """
        self.margin = margin
        self.metric = metric
        self.squared = squared
        self.filter_invalid = filter_invalid
        
    def mine(
        self,
        embeddings: mx.array,
        labels: mx.array
    ) -> Tuple[List[Tuple[int, int, int]], MiningStatistics]:
        """
        Mine hard triplets from batch.
        
        Args:
            embeddings: Batch embeddings [N, D]
            labels: Batch labels [N]
            
        Returns:
            List of triplet indices and mining statistics
        """
        import time
        start_time = time.time()
        
        n_samples = embeddings.shape[0]
        
        # Compute pairwise distances
        distances = pairwise_distance(embeddings, self.metric, self.squared)
        
        # Create label equality matrix
        labels_equal = mx.equal(labels[:, None], labels[None, :])
        labels_not_equal = mx.logical_not(labels_equal)
        
        # For each anchor, find hardest positive and negative
        triplets = []
        pos_distances = []
        neg_distances = []
        
        for i in range(n_samples):
            # Find all positives for this anchor
            positive_mask = labels_equal[i]
            positive_mask = mx.where(mx.arange(n_samples) != i, positive_mask, False)
            
            # Find all negatives for this anchor
            negative_mask = labels_not_equal[i]
            
            # Check if anchor has any positives and negatives
            has_positives = mx.any(positive_mask)
            has_negatives = mx.any(negative_mask)
            
            if has_positives and has_negatives:
                # Get distances to positives and negatives
                pos_dists = mx.where(positive_mask, distances[i], -mx.inf)
                neg_dists = mx.where(negative_mask, distances[i], mx.inf)
                
                # Find hardest positive (largest distance)
                hardest_positive_idx = mx.argmax(pos_dists).item()
                hardest_positive_dist = distances[i, hardest_positive_idx].item()
                
                # Find hardest negative (smallest distance)
                hardest_negative_idx = mx.argmin(neg_dists).item()
                hardest_negative_dist = distances[i, hardest_negative_idx].item()
                
                # Add triplet
                triplets.append((i, hardest_positive_idx, hardest_negative_idx))
                pos_distances.append(hardest_positive_dist)
                neg_distances.append(hardest_negative_dist)
        
        # Compute statistics
        if triplets:
            pos_distances = np.array(pos_distances)
            neg_distances = np.array(neg_distances)
            
            # Classify triplets
            hard_mask = pos_distances > neg_distances
            semi_hard_mask = (neg_distances > pos_distances) & (neg_distances < pos_distances + self.margin)
            easy_mask = neg_distances >= pos_distances + self.margin
            
            stats = MiningStatistics(
                num_triplets=len(triplets),
                num_hard_triplets=int(np.sum(hard_mask)),
                num_semi_hard_triplets=int(np.sum(semi_hard_mask)),
                num_easy_triplets=int(np.sum(easy_mask)),
                avg_positive_distance=float(np.mean(pos_distances)),
                avg_negative_distance=float(np.mean(neg_distances)),
                num_anchors_with_positives=len(triplets),
                num_anchors_with_negatives=len(triplets),
                mining_time_ms=(time.time() - start_time) * 1000
            )
        else:
            stats = MiningStatistics(
                num_triplets=0,
                num_hard_triplets=0,
                num_semi_hard_triplets=0,
                num_easy_triplets=0,
                avg_positive_distance=0.0,
                avg_negative_distance=0.0,
                num_anchors_with_positives=0,
                num_anchors_with_negatives=0,
                mining_time_ms=(time.time() - start_time) * 1000
            )
        
        return triplets, stats


class OnlineTripletMiner:
    """
    Online triplet mining with various selection strategies.
    
    This miner supports multiple strategies for selecting triplets during training:
    - all: Select all valid triplets
    - hard: Select only hard triplets
    - semi-hard: Select only semi-hard triplets
    - random: Random sampling of triplets
    
    Example:
        >>> miner = OnlineTripletMiner(strategy="semi-hard", margin=0.5)
        >>> triplets, stats = miner.mine(embeddings, labels)
    """
    
    def __init__(
        self,
        strategy: Literal["all", "hard", "semi-hard", "random"] = "semi-hard",
        margin: float = 0.5,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        max_triplets_per_anchor: Optional[int] = None,
        negative_selection_ratio: float = 1.0
    ):
        """
        Initialize online triplet miner.
        
        Args:
            strategy: Triplet selection strategy
            margin: Margin for semi-hard selection
            metric: Distance metric
            max_triplets_per_anchor: Maximum triplets per anchor
            negative_selection_ratio: Ratio of negatives to consider
        """
        self.strategy = strategy
        self.margin = margin
        self.metric = metric
        self.max_triplets_per_anchor = max_triplets_per_anchor
        self.negative_selection_ratio = negative_selection_ratio
        
    def mine(
        self,
        embeddings: mx.array,
        labels: mx.array
    ) -> Tuple[List[Tuple[int, int, int]], MiningStatistics]:
        """
        Mine triplets online based on strategy.
        
        Args:
            embeddings: Batch embeddings [N, D]
            labels: Batch labels [N]
            
        Returns:
            List of triplet indices and statistics
        """
        import time
        start_time = time.time()
        
        n_samples = embeddings.shape[0]
        
        # Compute pairwise distances
        if self.metric == "euclidean":
            distances = euclidean_distance(embeddings, embeddings, squared=False)
        else:
            distances = cosine_distance(embeddings, embeddings)
        
        # Create masks for positive and negative pairs
        indices = mx.arange(n_samples)
        labels_equal = mx.equal(labels[:, None], labels[None, :])
        
        # Mask out diagonal (same sample)
        valid_positive_mask = labels_equal & (indices[:, None] != indices[None, :])
        valid_negative_mask = ~labels_equal
        
        triplets = []
        all_pos_distances = []
        all_neg_distances = []
        
        for anchor_idx in range(n_samples):
            # Get positive indices for this anchor
            positive_indices = mx.where(valid_positive_mask[anchor_idx])[0]
            negative_indices = mx.where(valid_negative_mask[anchor_idx])[0]
            
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
                
            # Sample negatives if ratio < 1.0
            if self.negative_selection_ratio < 1.0:
                n_negatives = max(1, int(len(negative_indices) * self.negative_selection_ratio))
                perm = mx.random.permutation(len(negative_indices))[:n_negatives]
                negative_indices = negative_indices[perm]
            
            # Get distances
            anchor_embed = embeddings[anchor_idx:anchor_idx+1]
            pos_distances = distances[anchor_idx, positive_indices]
            neg_distances = distances[anchor_idx, negative_indices]
            
            # Generate triplets based on strategy
            if self.strategy == "all":
                # All valid triplets
                for pos_idx in positive_indices:
                    for neg_idx in negative_indices:
                        triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))
                        all_pos_distances.append(distances[anchor_idx, pos_idx].item())
                        all_neg_distances.append(distances[anchor_idx, neg_idx].item())
                        
            elif self.strategy == "hard":
                # Only hard triplets
                for pos_idx in positive_indices:
                    pos_dist = distances[anchor_idx, pos_idx]
                    hard_negatives = negative_indices[neg_distances < pos_dist]
                    
                    for neg_idx in hard_negatives:
                        triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))
                        all_pos_distances.append(pos_dist.item())
                        all_neg_distances.append(distances[anchor_idx, neg_idx].item())
                        
            elif self.strategy == "semi-hard":
                # Semi-hard triplets
                for pos_idx in positive_indices:
                    pos_dist = distances[anchor_idx, pos_idx]
                    
                    # Semi-hard: pos_dist < neg_dist < pos_dist + margin
                    mask = (neg_distances > pos_dist) & (neg_distances < pos_dist + self.margin)
                    semi_hard_negatives = negative_indices[mask]
                    
                    if len(semi_hard_negatives) == 0:
                        # Fall back to hardest negative if no semi-hard
                        hard_negatives = negative_indices[neg_distances < pos_dist + self.margin]
                        if len(hard_negatives) > 0:
                            neg_idx = hard_negatives[mx.argmax(distances[anchor_idx, hard_negatives])]
                            triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))
                            all_pos_distances.append(pos_dist.item())
                            all_neg_distances.append(distances[anchor_idx, neg_idx].item())
                    else:
                        for neg_idx in semi_hard_negatives:
                            triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))
                            all_pos_distances.append(pos_dist.item())
                            all_neg_distances.append(distances[anchor_idx, neg_idx].item())
                            
            elif self.strategy == "random":
                # Random sampling
                n_positives = len(positive_indices)
                n_negatives = len(negative_indices)
                n_triplets = min(n_positives * n_negatives, 
                               self.max_triplets_per_anchor or n_positives)
                
                for _ in range(n_triplets):
                    pos_idx = positive_indices[mx.random.randint(0, n_positives)]
                    neg_idx = negative_indices[mx.random.randint(0, n_negatives)]
                    triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))
                    all_pos_distances.append(distances[anchor_idx, pos_idx].item())
                    all_neg_distances.append(distances[anchor_idx, neg_idx].item())
            
            # Limit triplets per anchor if specified
            if self.max_triplets_per_anchor and len(triplets) > self.max_triplets_per_anchor:
                # Keep only the hardest triplets
                triplet_hardness = [all_pos_distances[i] - all_neg_distances[i] 
                                  for i in range(len(all_pos_distances))]
                sorted_indices = np.argsort(triplet_hardness)[::-1][:self.max_triplets_per_anchor]
                
                triplets = [triplets[i] for i in sorted_indices]
                all_pos_distances = [all_pos_distances[i] for i in sorted_indices]
                all_neg_distances = [all_neg_distances[i] for i in sorted_indices]
        
        # Compute statistics
        if triplets:
            pos_distances = np.array(all_pos_distances)
            neg_distances = np.array(all_neg_distances)
            
            hard_mask = pos_distances > neg_distances
            semi_hard_mask = (neg_distances > pos_distances) & (neg_distances < pos_distances + self.margin)
            easy_mask = neg_distances >= pos_distances + self.margin
            
            # Count unique anchors
            unique_anchors = set(t[0] for t in triplets)
            
            stats = MiningStatistics(
                num_triplets=len(triplets),
                num_hard_triplets=int(np.sum(hard_mask)),
                num_semi_hard_triplets=int(np.sum(semi_hard_mask)),
                num_easy_triplets=int(np.sum(easy_mask)),
                avg_positive_distance=float(np.mean(pos_distances)),
                avg_negative_distance=float(np.mean(neg_distances)),
                num_anchors_with_positives=len(unique_anchors),
                num_anchors_with_negatives=len(unique_anchors),
                mining_time_ms=(time.time() - start_time) * 1000
            )
        else:
            stats = MiningStatistics(
                num_triplets=0,
                num_hard_triplets=0,
                num_semi_hard_triplets=0,
                num_easy_triplets=0,
                avg_positive_distance=0.0,
                avg_negative_distance=0.0,
                num_anchors_with_positives=0,
                num_anchors_with_negatives=0,
                mining_time_ms=(time.time() - start_time) * 1000
            )
            
        return triplets, stats


class OfflineTripletGenerator:
    """
    Generate triplets offline before training.
    
    This is useful for creating balanced triplet datasets or when you want
    to precompute triplets for faster training.
    
    Example:
        >>> generator = OfflineTripletGenerator(embeddings_func=model.encode)
        >>> triplet_dataset = generator.generate(data, labels, n_triplets=10000)
    """
    
    def __init__(
        self,
        embeddings_func,
        margin: float = 0.5,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        strategy: Literal["random", "hard", "semi-hard", "balanced"] = "balanced"
    ):
        """
        Initialize offline generator.
        
        Args:
            embeddings_func: Function to compute embeddings
            margin: Margin for semi-hard selection
            metric: Distance metric
            strategy: Generation strategy
        """
        self.embeddings_func = embeddings_func
        self.margin = margin
        self.metric = metric
        self.strategy = strategy
        
    def generate(
        self,
        data: mx.array,
        labels: mx.array,
        n_triplets: int,
        batch_size: int = 256
    ) -> List[Tuple[int, int, int]]:
        """
        Generate triplets offline.
        
        Args:
            data: Input data
            labels: Data labels
            n_triplets: Number of triplets to generate
            batch_size: Batch size for embedding computation
            
        Returns:
            List of triplet indices
        """
        # Compute all embeddings
        n_samples = data.shape[0]
        embeddings = []
        
        for i in range(0, n_samples, batch_size):
            batch = data[i:i+batch_size]
            batch_embeddings = self.embeddings_func(batch)
            embeddings.append(batch_embeddings)
            
        embeddings = mx.concatenate(embeddings, axis=0)
        
        # Compute pairwise distances
        distances = pairwise_distance(embeddings, self.metric)
        
        # Create label groups
        unique_labels = mx.unique(labels)
        label_to_indices = {}
        
        for label in unique_labels:
            label_to_indices[label.item()] = mx.where(labels == label)[0]
        
        # Generate triplets based on strategy
        triplets = []
        
        if self.strategy == "balanced":
            # Ensure equal representation of all classes
            triplets_per_class = n_triplets // len(unique_labels)
            
            for label in unique_labels:
                label = label.item()
                class_indices = label_to_indices[label]
                
                if len(class_indices) < 2:
                    continue
                    
                # Get negative indices
                negative_mask = labels != label
                negative_indices = mx.where(negative_mask)[0]
                
                for _ in range(triplets_per_class):
                    # Random anchor and positive from same class
                    anchor_idx, positive_idx = np.random.choice(
                        class_indices.tolist(), size=2, replace=False
                    )
                    
                    # Select negative based on distance
                    anchor_to_negatives = distances[anchor_idx, negative_indices]
                    
                    if self.strategy == "hard":
                        # Hardest negative
                        neg_idx = negative_indices[mx.argmin(anchor_to_negatives)]
                    else:
                        # Random negative
                        neg_idx = np.random.choice(negative_indices.tolist())
                        
                    triplets.append((anchor_idx, positive_idx, neg_idx.item()))
                    
        else:
            # Other strategies
            online_miner = OnlineTripletMiner(
                strategy=self.strategy if self.strategy != "balanced" else "all",
                margin=self.margin,
                metric=self.metric
            )
            triplets, _ = online_miner.mine(embeddings, labels)
            
            # Sample if we got too many
            if len(triplets) > n_triplets:
                indices = np.random.choice(len(triplets), n_triplets, replace=False)
                triplets = [triplets[i] for i in indices]
                
        return triplets[:n_triplets]