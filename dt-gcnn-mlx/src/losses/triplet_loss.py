"""
Triplet Loss implementation for DT-GCNN using MLX.

This module provides triplet loss functions with various distance metrics
and dynamic margin adjustment capabilities.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Literal, Tuple, Dict
import numpy as np


def euclidean_distance(x: mx.array, y: mx.array, squared: bool = False) -> mx.array:
    """
    Compute pairwise Euclidean distance between embeddings.
    
    Args:
        x: First set of embeddings [N, D]
        y: Second set of embeddings [M, D]
        squared: If True, return squared distances
        
    Returns:
        Distance matrix [N, M]
        
    Example:
        >>> x = mx.random.normal((10, 128))
        >>> y = mx.random.normal((5, 128))
        >>> dist = euclidean_distance(x, y)
        >>> dist.shape
        (10, 5)
    """
    # Compute squared distances: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    x_norm = mx.sum(x * x, axis=1, keepdims=True)
    y_norm = mx.sum(y * y, axis=1, keepdims=True)
    
    # Compute pairwise distances
    dist = x_norm - 2.0 * mx.matmul(x, y.T) + y_norm.T
    
    # Ensure non-negative distances (numerical stability)
    dist = mx.maximum(dist, 0.0)
    
    if not squared:
        # Add small epsilon for numerical stability
        dist = mx.sqrt(dist + 1e-8)
    
    return dist


def cosine_distance(x: mx.array, y: mx.array) -> mx.array:
    """
    Compute pairwise cosine distance between embeddings.
    
    Args:
        x: First set of embeddings [N, D]
        y: Second set of embeddings [M, D]
        
    Returns:
        Distance matrix [N, M] where values are in [0, 2]
        
    Example:
        >>> x = mx.random.normal((10, 128))
        >>> y = mx.random.normal((5, 128))
        >>> dist = cosine_distance(x, y)
        >>> dist.shape
        (10, 5)
    """
    # Normalize embeddings
    x_norm = x / (mx.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    y_norm = y / (mx.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    sim = mx.matmul(x_norm, y_norm.T)
    
    # Convert to distance: dist = 1 - similarity
    # Range is [0, 2] where 0 means identical, 2 means opposite
    dist = 1.0 - sim
    
    return dist


def pairwise_distance(
    embeddings: mx.array,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    squared: bool = False
) -> mx.array:
    """
    Compute pairwise distances between all embeddings.
    
    Args:
        embeddings: Embeddings tensor [N, D]
        metric: Distance metric to use
        squared: For euclidean, whether to return squared distances
        
    Returns:
        Pairwise distance matrix [N, N]
        
    Example:
        >>> embeddings = mx.random.normal((32, 128))
        >>> distances = pairwise_distance(embeddings)
        >>> distances.shape
        (32, 32)
    """
    if metric == "euclidean":
        return euclidean_distance(embeddings, embeddings, squared=squared)
    elif metric == "cosine":
        return cosine_distance(embeddings, embeddings)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def triplet_margin_loss(
    anchor: mx.array,
    positive: mx.array,
    negative: mx.array,
    margin: float = 1.0,
    p: int = 2,
    swap: bool = False,
    reduction: Literal["mean", "sum", "none"] = "mean"
) -> mx.array:
    """
    Compute triplet margin loss.
    
    Loss = max(d(a,p) - d(a,n) + margin, 0)
    
    Args:
        anchor: Anchor embeddings [N, D]
        positive: Positive embeddings [N, D]
        negative: Negative embeddings [N, D]
        margin: Margin value
        p: Norm degree for distance computation (1 or 2)
        swap: If True, use swap correction
        reduction: How to reduce the loss
        
    Returns:
        Triplet loss value
        
    Example:
        >>> anchor = mx.random.normal((32, 128))
        >>> positive = mx.random.normal((32, 128))
        >>> negative = mx.random.normal((32, 128))
        >>> loss = triplet_margin_loss(anchor, positive, negative)
    """
    # Compute distances
    if p == 1:
        dist_ap = mx.sum(mx.abs(anchor - positive), axis=1)
        dist_an = mx.sum(mx.abs(anchor - negative), axis=1)
    elif p == 2:
        dist_ap = mx.sum((anchor - positive) ** 2, axis=1)
        dist_an = mx.sum((anchor - negative) ** 2, axis=1)
    else:
        raise ValueError(f"p must be 1 or 2, got {p}")
    
    # Apply swap if enabled
    if swap:
        dist_pn = mx.sum((positive - negative) ** 2, axis=1) if p == 2 else mx.sum(mx.abs(positive - negative), axis=1)
        dist_an = mx.minimum(dist_an, dist_pn)
    
    # Compute loss
    losses = mx.maximum(dist_ap - dist_an + margin, 0.0)
    
    # Apply reduction
    if reduction == "mean":
        return mx.mean(losses)
    elif reduction == "sum":
        return mx.sum(losses)
    else:
        return losses


class TripletLoss(nn.Module):
    """
    Triplet loss module with configurable distance metrics and margin adjustment.
    
    This implementation supports:
    - Multiple distance metrics (Euclidean, Cosine)
    - Dynamic margin adjustment
    - Hard negative mining integration
    - Swap correction
    
    Example:
        >>> loss_fn = TripletLoss(margin=0.5, metric="euclidean")
        >>> # With pre-selected triplets
        >>> loss = loss_fn(anchor, positive, negative)
        >>> 
        >>> # With embeddings and labels (requires miner)
        >>> embeddings = model(batch)
        >>> loss = loss_fn.compute_loss(embeddings, labels, miner)
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        p: int = 2,
        swap: bool = False,
        reduction: Literal["mean", "sum", "none"] = "mean",
        dynamic_margin: bool = False,
        margin_scheduler: Optional[Dict] = None
    ):
        """
        Initialize triplet loss.
        
        Args:
            margin: Base margin value
            metric: Distance metric to use
            p: Norm degree for Euclidean distance
            swap: Enable swap correction
            reduction: Loss reduction method
            dynamic_margin: Enable dynamic margin adjustment
            margin_scheduler: Configuration for margin scheduling
        """
        super().__init__()
        self.margin = margin
        self.metric = metric
        self.p = p
        self.swap = swap
        self.reduction = reduction
        self.dynamic_margin = dynamic_margin
        self.margin_scheduler = margin_scheduler or {}
        
        # Dynamic margin state
        self._current_margin = margin
        self._step = 0
        
    def update_margin(self, step: Optional[int] = None) -> float:
        """Update margin based on training progress."""
        if not self.dynamic_margin:
            return self.margin
            
        if step is not None:
            self._step = step
        else:
            self._step += 1
            
        scheduler_type = self.margin_scheduler.get("type", "linear")
        
        if scheduler_type == "linear":
            start = self.margin_scheduler.get("start_margin", self.margin)
            end = self.margin_scheduler.get("end_margin", self.margin * 2)
            steps = self.margin_scheduler.get("steps", 10000)
            
            alpha = min(self._step / steps, 1.0)
            self._current_margin = start + alpha * (end - start)
            
        elif scheduler_type == "exponential":
            gamma = self.margin_scheduler.get("gamma", 0.999)
            self._current_margin = self.margin * (gamma ** self._step)
            
        elif scheduler_type == "cosine":
            min_margin = self.margin_scheduler.get("min_margin", 0.1)
            period = self.margin_scheduler.get("period", 10000)
            
            alpha = 0.5 * (1 + mx.cos(mx.pi * (self._step % period) / period))
            self._current_margin = min_margin + alpha * (self.margin - min_margin)
            
        return self._current_margin
    
    def __call__(
        self,
        anchor: mx.array,
        positive: mx.array,
        negative: mx.array
    ) -> mx.array:
        """
        Compute triplet loss for pre-selected triplets.
        
        Args:
            anchor: Anchor embeddings [N, D]
            positive: Positive embeddings [N, D]
            negative: Negative embeddings [N, D]
            
        Returns:
            Loss value
        """
        margin = self.update_margin()
        
        if self.metric == "euclidean":
            return triplet_margin_loss(
                anchor, positive, negative,
                margin=margin,
                p=self.p,
                swap=self.swap,
                reduction=self.reduction
            )
        else:
            # For cosine distance
            dist_ap = cosine_distance(anchor.reshape(anchor.shape[0], 1, -1), 
                                     positive.reshape(positive.shape[0], 1, -1))[:, 0]
            dist_an = cosine_distance(anchor.reshape(anchor.shape[0], 1, -1), 
                                     negative.reshape(negative.shape[0], 1, -1))[:, 0]
            
            losses = mx.maximum(dist_ap - dist_an + margin, 0.0)
            
            if self.reduction == "mean":
                return mx.mean(losses)
            elif self.reduction == "sum":
                return mx.sum(losses)
            else:
                return losses
    
    def compute_loss(
        self,
        embeddings: mx.array,
        labels: mx.array,
        miner: Optional["BatchHardMiner"] = None
    ) -> Tuple[mx.array, Dict]:
        """
        Compute triplet loss with mining.
        
        Args:
            embeddings: All embeddings in batch [N, D]
            labels: Labels for embeddings [N]
            miner: Triplet miner instance
            
        Returns:
            Loss value and mining statistics
        """
        if miner is None:
            raise ValueError("Miner required for compute_loss")
            
        # Mine triplets
        triplets, stats = miner.mine(embeddings, labels)
        
        if len(triplets) == 0:
            # No valid triplets found
            return mx.array(0.0), stats
            
        # Extract indices
        anchor_idx = mx.array([t[0] for t in triplets])
        positive_idx = mx.array([t[1] for t in triplets])
        negative_idx = mx.array([t[2] for t in triplets])
        
        # Get embeddings
        anchor = embeddings[anchor_idx]
        positive = embeddings[positive_idx]
        negative = embeddings[negative_idx]
        
        # Compute loss
        loss = self(anchor, positive, negative)
        
        return loss, stats