"""Performance benchmarking for DT-GCNN on Apple Silicon."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import psutil
import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DTGCNN, create_model
from src.losses.triplet_loss import TripletLoss


class PerformanceBenchmark:
    """Benchmark suite for DT-GCNN performance testing."""
    
    def __init__(self):
        self.results = defaultdict(dict)
        
    def measure_memory(self):
        """Measure current memory usage."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
        
    def benchmark_forward_pass(self, batch_sizes, seq_lengths, vocab_sizes):
        """Benchmark forward pass with different configurations."""
        print("\n1. Forward Pass Benchmarks")
        print("=" * 60)
        
        results = []
        
        for vocab_size in vocab_sizes:
            for seq_length in seq_lengths:
                for batch_size in batch_sizes:
                    # Create model
                    model = create_model(
                        preset="small",
                        vocab_size=vocab_size,
                        num_classes=4
                    )
                    
                    # Create dummy data
                    input_data = mx.random.randint(0, vocab_size, [batch_size, seq_length])
                    
                    # Warmup
                    for _ in range(3):
                        logits, _ = model(input_data, return_embeddings=False)
                        mx.eval(logits)
                    
                    # Measure time
                    start_time = time.time()
                    num_iterations = 10
                    
                    for _ in range(num_iterations):
                        logits, _ = model(input_data, return_embeddings=False)
                        mx.eval(logits)
                        
                    elapsed_time = (time.time() - start_time) / num_iterations
                    throughput = batch_size / elapsed_time
                    
                    # Measure memory
                    memory_used = self.measure_memory()
                    
                    result = {
                        'vocab_size': vocab_size,
                        'seq_length': seq_length,
                        'batch_size': batch_size,
                        'time_per_batch': elapsed_time,
                        'samples_per_second': throughput,
                        'memory_mb': memory_used
                    }
                    
                    results.append(result)
                    
                    print(f"Vocab: {vocab_size:5d}, Seq: {seq_length:3d}, Batch: {batch_size:3d} | "
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
        vocab_size = 5000
        seq_length = 128
        
        for batch_size in batch_sizes:
            # Create model
            model = create_model(
                preset="small",
                vocab_size=vocab_size,
                num_classes=4
            )
            
            optimizer = optim.Adam(learning_rate=0.001)
            
            # Create dummy data
            input_data = mx.random.randint(0, vocab_size, [batch_size, seq_length])
            labels = mx.random.randint(0, 4, [batch_size])
            
            # Define loss function
            def loss_fn(params):
                logits, _ = model(input_data, return_embeddings=False)
                return mx.mean(nn.losses.cross_entropy(logits, labels))
                
            # Warmup
            for _ in range(3):
                loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
                mx.eval(loss, grads)
                
            # Measure time
            start_time = time.time()
            num_iterations = 10
            
            for _ in range(num_iterations):
                loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                
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
        embedding_dim = 64
        
        for batch_size in batch_sizes:
            # Create triplet embeddings
            anchor = mx.random.normal((batch_size, embedding_dim))
            positive = mx.random.normal((batch_size, embedding_dim))
            negative = mx.random.normal((batch_size, embedding_dim))
            
            # Create loss function
            triplet_loss = TripletLoss(margin=0.3)
            
            # Warmup
            for _ in range(5):
                loss = triplet_loss(anchor, positive, negative)
                mx.eval(loss)
                
            # Measure time
            start_time = time.time()
            num_iterations = 50
            
            for _ in range(num_iterations):
                loss = triplet_loss(anchor, positive, negative)
                mx.eval(loss)
                
            elapsed_time = (time.time() - start_time) / num_iterations
            
            result = {
                'batch_size': batch_size,
                'time_ms': elapsed_time * 1000,
                'triplets_per_second': batch_size / elapsed_time
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
            {'name': 'Small', 'preset': 'small'},
            {'name': 'Base', 'preset': 'base'},
            {'name': 'Large', 'preset': 'large'}
        ]
        
        vocab_size = 5000
        batch_size = 16
        seq_length = 128
        
        for config in configurations:
            # Create model
            model = create_model(
                preset=config['preset'],
                vocab_size=vocab_size,
                num_classes=4
            )
            
            # Count parameters
            total_params = sum(param.size if hasattr(param, 'size') else len(param) for param in model.parameters().values())
            param_memory = total_params * 4 / 1024 / 1024  # MB (float32)
            
            # Create dummy data
            input_data = mx.random.randint(0, vocab_size, [batch_size, seq_length])
            
            # Measure forward pass time
            start_time = time.time()
            num_iterations = 20
            
            for _ in range(num_iterations):
                logits, _ = model(input_data, return_embeddings=False)
                mx.eval(logits)
                
            elapsed_time = (time.time() - start_time) / num_iterations
            
            result = {
                'name': config['name'],
                'preset': config['preset'],
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
            
            # Group by vocab size
            data_by_vocab = defaultdict(list)
            for result in self.results['forward_pass']:
                if result['seq_length'] == 128:  # Fix sequence length
                    data_by_vocab[result['vocab_size']].append(result)
                    
            for vocab_size, results in sorted(data_by_vocab.items()):
                batch_sizes = [r['batch_size'] for r in results]
                throughputs = [r['samples_per_second'] for r in results]
                plt.plot(batch_sizes, throughputs, marker='o', 
                        label=f'Vocab {vocab_size}')
                
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
            large_model = next((r for r in self.results['model_sizes'] if r['name'] == 'Large'), None)
            if large_model:
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
    seq_lengths = [64, 128, 256]
    vocab_sizes = [1000, 5000, 10000]
    
    benchmark.benchmark_forward_pass(batch_sizes, seq_lengths, vocab_sizes)
    
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