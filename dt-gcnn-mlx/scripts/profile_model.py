#!/usr/bin/env python3
"""
Profile DT-GCNN model performance
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
from models.dt_gcnn import DT_GCNN
from training.utils import ModelProfiler
import argparse


def main():
    parser = argparse.ArgumentParser(description="Profile DT-GCNN model")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for profiling')
    parser.add_argument('--num-vertices', type=int, default=1723,
                        help='Number of vertices')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--output', type=str, default='./profile_report.txt',
                        help='Output path for profile report')
    
    args = parser.parse_args()
    
    print("Creating model...")
    model = DT_GCNN(
        num_vertices=args.num_vertices,
        embedding_dim=args.embedding_dim,
        num_classes=10
    )
    
    # Initialize model
    dummy_coords = mx.zeros((1, args.num_vertices, 3))
    dummy_features = mx.zeros((1, args.num_vertices, 6))
    _ = model(dummy_coords, dummy_features)
    
    print(f"Profiling with batch size {args.batch_size}...")
    profile_results = ModelProfiler.profile_model(
        model,
        batch_size=args.batch_size,
        num_vertices=args.num_vertices
    )
    
    # Create report
    ModelProfiler.create_profile_report(profile_results, args.output)
    
    print(f"\nProfile Report:")
    print(f"  Device: {profile_results['device']}")
    print(f"  Forward pass: {profile_results['avg_forward_time_ms']:.2f} ms")
    print(f"  Throughput: {profile_results['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"  Memory: {profile_results['memory_usage_mb']:.2f} MB")
    print(f"  Parameters: {profile_results['total_parameters']:,}")
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()