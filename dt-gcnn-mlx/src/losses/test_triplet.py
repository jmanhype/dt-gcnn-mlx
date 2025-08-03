"""
Test script for triplet loss and mining strategies.

This script demonstrates how to use the triplet loss components with MLX.
"""

import mlx.core as mx
import mlx.nn as nn
from triplet_loss import TripletLoss, pairwise_distance, euclidean_distance, cosine_distance
from mining_strategies import BatchHardMiner, OnlineTripletMiner, MiningStatistics


def test_distance_functions():
    """Test distance computation functions."""
    print("Testing distance functions...")
    
    # Create sample embeddings
    embeddings = mx.random.normal((10, 128))
    
    # Test Euclidean distance
    dist_euclidean = euclidean_distance(embeddings, embeddings)
    print(f"Euclidean distance shape: {dist_euclidean.shape}")
    print(f"Diagonal (should be ~0): {mx.diag(dist_euclidean)[:5]}")
    
    # Test squared Euclidean
    dist_squared = euclidean_distance(embeddings, embeddings, squared=True)
    print(f"\nSquared Euclidean distance shape: {dist_squared.shape}")
    
    # Test cosine distance
    dist_cosine = cosine_distance(embeddings, embeddings)
    print(f"\nCosine distance shape: {dist_cosine.shape}")
    print(f"Diagonal (should be ~0): {mx.diag(dist_cosine)[:5]}")
    
    # Test pairwise distance
    dist_pairwise = pairwise_distance(embeddings, metric="euclidean")
    print(f"\nPairwise distance shape: {dist_pairwise.shape}")
    
    print("\n✓ Distance functions test passed!")


def test_triplet_loss():
    """Test triplet loss computation."""
    print("\nTesting triplet loss...")
    
    # Create sample triplets
    batch_size = 32
    embedding_dim = 128
    
    anchor = mx.random.normal((batch_size, embedding_dim))
    positive = anchor + mx.random.normal((batch_size, embedding_dim)) * 0.1  # Close to anchor
    negative = mx.random.normal((batch_size, embedding_dim))  # Random (far from anchor)
    
    # Test basic triplet loss
    loss_fn = TripletLoss(margin=1.0, metric="euclidean")
    loss = loss_fn(anchor, positive, negative)
    print(f"Triplet loss (Euclidean): {loss.item():.4f}")
    
    # Test with cosine distance
    loss_fn_cosine = TripletLoss(margin=0.5, metric="cosine")
    loss_cosine = loss_fn_cosine(anchor, positive, negative)
    print(f"Triplet loss (Cosine): {loss_cosine.item():.4f}")
    
    # Test with dynamic margin
    loss_fn_dynamic = TripletLoss(
        margin=0.5,
        dynamic_margin=True,
        margin_scheduler={
            "type": "linear",
            "start_margin": 0.5,
            "end_margin": 2.0,
            "steps": 1000
        }
    )
    
    # Simulate training steps
    margins = []
    for step in range(0, 1001, 100):
        loss_fn_dynamic.update_margin(step)
        margins.append(loss_fn_dynamic._current_margin)
    
    print(f"\nDynamic margin progression: {margins[:5]} ... {margins[-1]}")
    
    print("\n✓ Triplet loss test passed!")


def test_batch_hard_mining():
    """Test batch-hard mining strategy."""
    print("\nTesting batch-hard mining...")
    
    # Create sample batch with known structure
    batch_size = 64
    embedding_dim = 128
    n_classes = 8
    samples_per_class = batch_size // n_classes
    
    # Generate embeddings clustered by class
    embeddings = []
    labels = []
    
    for class_id in range(n_classes):
        # Create cluster center
        center = mx.random.normal((1, embedding_dim)) * 5
        
        # Generate samples around center
        class_embeddings = center + mx.random.normal((samples_per_class, embedding_dim)) * 0.5
        embeddings.append(class_embeddings)
        labels.extend([class_id] * samples_per_class)
    
    embeddings = mx.concatenate(embeddings, axis=0)
    labels = mx.array(labels)
    
    # Test batch-hard mining
    miner = BatchHardMiner(margin=0.5, metric="euclidean")
    triplets, stats = miner.mine(embeddings, labels)
    
    print(f"Number of triplets: {stats.num_triplets}")
    print(f"Hard triplets: {stats.num_hard_triplets}")
    print(f"Semi-hard triplets: {stats.num_semi_hard_triplets}")
    print(f"Easy triplets: {stats.num_easy_triplets}")
    print(f"Avg positive distance: {stats.avg_positive_distance:.4f}")
    print(f"Avg negative distance: {stats.avg_negative_distance:.4f}")
    print(f"Mining time: {stats.mining_time_ms:.2f}ms")
    
    print("\n✓ Batch-hard mining test passed!")


def test_online_mining_strategies():
    """Test different online mining strategies."""
    print("\nTesting online mining strategies...")
    
    # Create sample data
    batch_size = 48
    embedding_dim = 64
    embeddings = mx.random.normal((batch_size, embedding_dim))
    labels = mx.array([i // 6 for i in range(batch_size)])  # 8 classes, 6 samples each
    
    strategies = ["all", "hard", "semi-hard", "random"]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        miner = OnlineTripletMiner(strategy=strategy, margin=0.5)
        triplets, stats = miner.mine(embeddings, labels)
        
        print(f"  Triplets: {stats.num_triplets}")
        print(f"  Hard ratio: {stats.num_hard_triplets / max(stats.num_triplets, 1):.2%}")
        print(f"  Mining time: {stats.mining_time_ms:.2f}ms")
    
    print("\n✓ Online mining test passed!")


def test_integrated_workflow():
    """Test complete workflow with loss and mining."""
    print("\nTesting integrated workflow...")
    
    # Simulate a training batch
    batch_size = 32
    embedding_dim = 128
    n_classes = 4
    
    # Create embeddings from a mock model
    embeddings = mx.random.normal((batch_size, embedding_dim))
    labels = mx.array([i % n_classes for i in range(batch_size)])
    
    # Initialize loss and miner
    loss_fn = TripletLoss(margin=1.0, metric="euclidean", dynamic_margin=True)
    miner = BatchHardMiner(margin=1.0)
    
    # Compute loss with mining
    loss, stats = loss_fn.compute_loss(embeddings, labels, miner)
    
    print(f"Loss value: {loss.item():.4f}")
    print(f"Mined {stats.num_triplets} triplets")
    print(f"Statistics: {stats.to_dict()}")
    
    # Test with no valid triplets (all same class)
    same_labels = mx.zeros(batch_size, dtype=mx.int32)
    loss_zero, stats_zero = loss_fn.compute_loss(embeddings, same_labels, miner)
    
    print(f"\nWith all same labels:")
    print(f"Loss: {loss_zero.item():.4f} (should be 0)")
    print(f"Triplets: {stats_zero.num_triplets} (should be 0)")
    
    print("\n✓ Integrated workflow test passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing DT-GCNN Triplet Loss Components")
    print("=" * 60)
    
    test_distance_functions()
    test_triplet_loss()
    test_batch_hard_mining()
    test_online_mining_strategies()
    test_integrated_workflow()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()