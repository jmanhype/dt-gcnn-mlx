"""
Losses package for DT-GCNN.

This module provides various loss functions for training DT-GCNN models,
with a focus on triplet loss and hard mining strategies.
"""

from .triplet_loss import (
    TripletLoss,
    triplet_margin_loss,
    pairwise_distance,
    cosine_distance,
    euclidean_distance,
)
from .mining_strategies import (
    BatchHardMiner,
    OnlineTripletMiner,
    OfflineTripletGenerator,
    MiningStatistics,
)

__all__ = [
    "TripletLoss",
    "triplet_margin_loss",
    "pairwise_distance",
    "cosine_distance",
    "euclidean_distance",
    "BatchHardMiner",
    "OnlineTripletMiner",
    "OfflineTripletGenerator",
    "MiningStatistics",
]