"""Performance benchmarking for DT-GCNN on Apple Silicon."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import time
import psutil
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dt_gcnn import DTGCNN
from src.losses.triplet_loss import BatchHardTripletLoss


class PerformanceBenchmark:
    """Benchmark suite for DT-GCNN performance testing."""
    
    def __init__(self):
        self.results = defaultdict(dict)
        
    def measure_memory(self):
        """Measure current memory usage."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
        
    def benchmark_forward_pass(self, batch_sizes, seq_lengths, num_nodes_list):
        """Benchmark forward pass with different configurations."""
        print("\n1. Forward Pass Benchmarks")
        print("=" * 60)
        
        results = []
        
        for num_nodes in num_nodes_list:
            for seq_length in seq_lengths:
                for batch_size in batch_sizes:
                    # Create model
                    model = DTGCNN(
                        num_nodes=num_nodes,
                        input_dim=8,
                        hidden_dims=[32, 64, 128],
                        temporal_kernel_size=3,
                        dilations=[1, 2, 4],
                        num_classes=10,
                        dropout=0.0  # No dropout for benchmarking
                    )
                    
                    # Create dummy data
                    adj = np.eye(num_nodes) + np.random.rand(num_nodes, num_nodes) * 0.1
                    adj = adj / adj.sum(axis=1, keepdims=True)
                    adj_matrix = mx.array(adj)
                    
                    features = mx.random.normal((batch_size, seq_length, num_nodes, 8))
                    
                    # Warmup
                    for _ in range(3):
                        _ = model(features, adj_matrix, training=False)
                    mx.eval(model.parameters())
                    
                    # Measure time
                    start_time = time.time()
                    num_iterations = 10
                    
                    for _ in range(num_iterations):
                        output = model(features, adj_matrix, training=False)
                        mx.eval(output)
                        
                    elapsed_time = (time.time() - start_time) / num_iterations
                    throughput = batch_size / elapsed_time
                    
                    # Measure memory
                    memory_used = self.measure_memory()
                    
                    result = {
                        'num_nodes': num_nodes,
                        'seq_length': seq_length,
                        'batch_size': batch_size,
                        'time_per_batch': elapsed_time,
                        'samples_per_second': throughput,
                        'memory_mb': memory_used
                    }
                    
                    results.append(result)
                    
                    print(f"Nodes: {num_nodes:3d}, Seq: {seq_length:3d}, Batch: {batch_size:3d} | "
                          f"Time: {elapsed_time*1000:6.2f}ms | "
                          f"Throughput: {throughput:6.1f} samples/s | "
                          f"Memory: {memory_used:6.1f} MB")
                    
        self.results['forward_pass'] = results
        return results
        
    def benchmark_backward_pass(self, batch_sizes):
        """Benchmark backward pass (training) performance."""
        print("\n2. Backward Pass (Training) Benchmarks")
        print("=" * 60)
        
        results = []
        num_nodes = 20
        seq_length = 30
        
        for batch_size in batch_sizes:
            # Create model
            model = DTGCNN(
                num_nodes=num_nodes,
                input_dim=8,
                hidden_dims=[32, 64, 128],
                temporal_kernel_size=3,
                dilations=[1, 2, 4],
                num_classes=10,
                dropout=0.3
            )
            
            # Create dummy data
            adj = np.eye(num_nodes) + np.random.rand(num_nodes, num_nodes) * 0.1
            adj = adj / adj.sum(axis=1, keepdims=True)
            adj_matrix = mx.array(adj)
            
            features = mx.random.normal((batch_size, seq_length, num_nodes, 8))
            labels = mx.array(np.random.randint(0, 10, batch_size))
            
            # Define loss function
            def loss_fn(model, features, adj, labels):
                logits = model(features, adj, training=True)
                return mx.mean(nn.losses.cross_entropy(logits, labels))
                
            # Warmup
            for _ in range(3):
                _, grads = mx.value_and_grad(loss_fn)(model, features, adj_matrix, labels)
                mx.eval(grads)
                
            # Measure time
            start_time = time.time()
            num_iterations = 10
            
            for _ in range(num_iterations):
                loss, grads = mx.value_and_grad(loss_fn)(model, features, adj_matrix, labels)
                mx.eval(loss, grads)
                
            elapsed_time = (time.time() - start_time) / num_iterations
            throughput = batch_size / elapsed_time
            
            # Measure memory
            memory_used = self.measure_memory()
            
            result = {
                'batch_size': batch_size,
                'time_per_iteration': elapsed_time,
                'samples_per_second': throughput,
                'memory_mb': memory_used
            }
            
            results.append(result)
            
            print(f"Batch: {batch_size:3d} | "
                  f"Time: {elapsed_time*1000:6.2f}ms | "
                  f"Throughput: {throughput:6.1f} samples/s | "
                  f"Memory: {memory_used:6.1f} MB")
                  
        self.results['backward_pass'] = results
        return results
        
    def benchmark_triplet_loss(self, batch_sizes):
        """Benchmark triplet loss computation."""
        print("\n3. Triplet Loss Benchmarks")
        print("=" * 60)
        
        results = []
        embedding_dim = 128
        
        for batch_size in batch_sizes:
            # Create embeddings and labels
            embeddings = mx.random.normal((batch_size, embedding_dim))
            # Ensure balanced classes for triplet mining
            num_classes = min(4, batch_size // 4)
            labels = mx.array([i % num_classes for i in range(batch_size)])
            
            # Create loss function
            triplet_loss = BatchHardTripletLoss(margin=0.3)
            
            # Warmup
            for _ in range(5):
                loss = triplet_loss(embeddings, labels)
                mx.eval(loss)
                
            # Measure time
            start_time = time.time()
            num_iterations = 50
            
            for _ in range(num_iterations):
                loss = triplet_loss(embeddings, labels)
                mx.eval(loss)
                
            elapsed_time = (time.time() - start_time) / num_iterations
            
            result = {
                'batch_size': batch_size,
                'time_ms': elapsed_time * 1000,
                'triplets_per_second': (batch_size * batch_size) / elapsed_time
            }
            
            results.append(result)
            
            print(f"Batch: {batch_size:3d} | "
                  f"Time: {elapsed_time*1000:6.2f}ms | "
                  f"Triplets/s: {result['triplets_per_second']:8.1f}")
                  
        self.results['triplet_loss'] = results
        return results
        
    def benchmark_model_sizes(self):
        """Benchmark different model sizes."""
        print("\n4. Model Size Benchmarks")
        print("=" * 60)
        
        results = []
        configurations = [
            {'name': 'Small', 'hidden_dims': [16, 32], 'dilations': [1, 2]},
            {'name': 'Medium', 'hidden_dims': [32, 64, 128], 'dilations': [1, 2, 4]},
            {'name': 'Large', 'hidden_dims': [64, 128, 256, 512], 'dilations': [1, 2, 4, 8]},
            {'name': 'XLarge', 'hidden_dims': [128, 256, 512, 1024], 'dilations': [1, 2, 4, 8, 16]}
        ]
        
        num_nodes = 20
        batch_size = 16
        seq_length = 30
        
        for config in configurations:
            # Create model
            model = DTGCNN(
                num_nodes=num_nodes,
                input_dim=8,
                hidden_dims=config['hidden_dims'],
                temporal_kernel_size=3,
                dilations=config['dilations'],
                num_classes=10,
                dropout=0.0
            )
            
            # Count parameters
            total_params = sum(p.size for p in model.parameters().values())
            param_memory = total_params * 4 / 1024 / 1024  # MB (float32)
            
            # Create dummy data
            adj = np.eye(num_nodes)
            adj_matrix = mx.array(adj)
            features = mx.random.normal((batch_size, seq_length, num_nodes, 8))
            
            # Measure forward pass time
            start_time = time.time()
            num_iterations = 20
            
            for _ in range(num_iterations):
                output = model(features, adj_matrix, training=False)
                mx.eval(output)
                
            elapsed_time = (time.time() - start_time) / num_iterations
            
            result = {
                'name': config['name'],
                'hidden_dims': config['hidden_dims'],
                'total_parameters': total_params,
                'param_memory_mb': param_memory,
                'inference_time_ms': elapsed_time * 1000,
                'throughput': batch_size / elapsed_time
            }
            
            results.append(result)
            
            print(f"{config['name']:8s} | "
                  f"Params: {total_params:10,d} | "
                  f"Memory: {param_memory:6.1f} MB | "
                  f"Time: {elapsed_time*1000:6.2f}ms | "
                  f"Throughput: {result['throughput']:6.1f} samples/s")
                  
        self.results['model_sizes'] = results
        return results
        
    def plot_results(self):
        """Generate visualization plots for benchmark results."""
        print("\n5. Generating Performance Plots...")
        
        # Plot 1: Throughput vs Batch Size
        if 'forward_pass' in self.results:
            plt.figure(figsize=(10, 6))
            
            # Group by number of nodes
            data_by_nodes = defaultdict(list)
            for result in self.results['forward_pass']:
                if result['seq_length'] == 30:  # Fix sequence length
                    data_by_nodes[result['num_nodes']].append(result)
                    
            for num_nodes, results in sorted(data_by_nodes.items()):
                batch_sizes = [r['batch_size'] for r in results]
                throughputs = [r['samples_per_second'] for r in results]
                plt.plot(batch_sizes, throughputs, marker='o', 
                        label=f'{num_nodes} nodes')
                
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (samples/second)')
            plt.title('DT-GCNN Inference Throughput on Apple Silicon')
            plt.legend()
            plt.grid(True)
            plt.savefig('throughput_vs_batch_size.png')
            print("Saved throughput_vs_batch_size.png")
            
        # Plot 2: Model Size vs Performance
        if 'model_sizes' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            model_names = [r['name'] for r in self.results['model_sizes']]
            params = [r['total_parameters'] for r in self.results['model_sizes']]
            throughputs = [r['throughput'] for r in self.results['model_sizes']]
            
            # Parameters bar chart
            ax1.bar(model_names, params, color='skyblue')
            ax1.set_ylabel('Total Parameters')
            ax1.set_title('Model Size Comparison')
            ax1.tick_params(axis='x', rotation=45)
            
            # Throughput bar chart
            ax2.bar(model_names, throughputs, color='lightcoral')
            ax2.set_ylabel('Throughput (samples/s)')
            ax2.set_title('Inference Speed Comparison')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('model_size_comparison.png')
            print("Saved model_size_comparison.png")
            
    def generate_report(self):
        """Generate a comprehensive benchmark report."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY REPORT")
        print("=" * 60)
        
        # System information
        print("\nSystem Information:")
        print(f"- Platform: Apple Silicon (MLX)")
        print(f"- Python: {sys.version.split()[0]}")
        print(f"- MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'Unknown'}")
        print(f"- Available memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
        
        # Key findings
        if 'forward_pass' in self.results:
            max_throughput = max(r['samples_per_second'] for r in self.results['forward_pass'])
            print(f"\nPeak Inference Throughput: {max_throughput:.1f} samples/second")
            
        if 'model_sizes' in self.results:
            large_model = next(r for r in self.results['model_sizes'] if r['name'] == 'Large')
            print(f"\nLarge Model Performance:")
            print(f"- Parameters: {large_model['total_parameters']:,}")
            print(f"- Inference time: {large_model['inference_time_ms']:.2f} ms/batch")
            print(f"- Throughput: {large_model['throughput']:.1f} samples/s")
            
        print("\nBenchmark completed successfully!")


def main():
    """Run the complete benchmark suite."""
    print("DT-GCNN Performance Benchmark on Apple Silicon")
    print("=" * 60)
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    
    # 1. Forward pass benchmarks
    batch_sizes = [1, 4, 8, 16, 32, 64]
    seq_lengths = [20, 30, 50]
    num_nodes_list = [10, 20, 30]
    
    benchmark.benchmark_forward_pass(batch_sizes, seq_lengths, num_nodes_list)
    
    # 2. Backward pass benchmarks
    batch_sizes_train = [4, 8, 16, 32]
    benchmark.benchmark_backward_pass(batch_sizes_train)
    
    # 3. Triplet loss benchmarks
    batch_sizes_triplet = [8, 16, 32, 64, 128]
    benchmark.benchmark_triplet_loss(batch_sizes_triplet)
    
    # 4. Model size benchmarks
    benchmark.benchmark_model_sizes()
    
    # Generate plots
    benchmark.plot_results()
    
    # Generate final report
    benchmark.generate_report()
    
    # Save raw results
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(dict(benchmark.results), f, indent=2)
    print("\nRaw results saved to benchmark_results.json")


if __name__ == "__main__":
    main()