"""
Training utilities for DT-GCNN
Monitoring, visualization, and helper functions
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime
import pandas as pd


class TrainingMonitor:
    """Real-time training monitoring and visualization"""
    
    def __init__(self, output_dir: str, plot_interval: int = 10):
        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        
        self.plot_interval = plot_interval
        self.metrics_history = []
        
        # Setup style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update metrics and create plots"""
        # Add epoch to metrics
        metrics['epoch'] = epoch
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Save metrics to file
        metrics_file = self.output_dir / "metrics_history.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Plot if interval reached
        if epoch % self.plot_interval == 0:
            self.plot_metrics()
    
    def plot_metrics(self):
        """Create training plots"""
        if len(self.metrics_history) < 2:
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.metrics_history)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DT-GCNN Training Progress', fontsize=16)
        
        # Plot 1: Loss curves
        ax = axes[0, 0]
        if 'train_loss' in df.columns:
            ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in df.columns:
            ax.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax = axes[0, 1]
        if 'train_acc' in df.columns:
            ax.plot(df['epoch'], df['train_acc'], label='Train Acc', linewidth=2)
        if 'val_acc' in df.columns:
            ax.plot(df['epoch'], df['val_acc'], label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Loss components
        ax = axes[1, 0]
        if 'classification_loss' in df.columns:
            ax.plot(df['epoch'], df['classification_loss'], 
                   label='Classification', linewidth=2)
        if 'triplet_loss' in df.columns:
            ax.plot(df['epoch'], df['triplet_loss'], 
                   label='Triplet', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Learning rate
        ax = axes[1, 1]
        if 'learning_rate' in df.columns:
            ax.plot(df['epoch'], df['learning_rate'], linewidth=2, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = self.plot_dir / f"training_curves_epoch_{df['epoch'].iloc[-1]:04d}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create loss distribution plot
        self._plot_loss_distribution(df)
    
    def _plot_loss_distribution(self, df: pd.DataFrame):
        """Plot loss distribution over time"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create boxplot for loss distribution
        if len(df) > 20:
            # Group by epochs for cleaner visualization
            epoch_groups = df.groupby(df['epoch'] // 10)
            
            loss_data = []
            positions = []
            
            for group_idx, group_df in epoch_groups:
                if 'train_loss' in group_df.columns:
                    loss_data.append(group_df['train_loss'].values)
                    positions.append(group_idx * 10)
            
            if loss_data:
                bp = ax.boxplot(loss_data, positions=positions, widths=5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Training Loss')
                ax.set_title('Training Loss Distribution')
                ax.grid(True, alpha=0.3)
        
        plot_path = self.plot_dir / "loss_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self):
        """Create final training summary report"""
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        # Calculate summary statistics
        summary = {
            'total_epochs': len(df),
            'best_val_loss': df['val_loss'].min() if 'val_loss' in df else None,
            'best_val_acc': df['val_acc'].max() if 'val_acc' in df else None,
            'best_epoch': df.loc[df['val_loss'].idxmin(), 'epoch'] if 'val_loss' in df else None,
            'final_train_loss': df['train_loss'].iloc[-1] if 'train_loss' in df else None,
            'final_val_loss': df['val_loss'].iloc[-1] if 'val_loss' in df else None,
            'total_training_time': None  # Calculated from timestamps
        }
        
        # Calculate training time
        if len(df) > 1:
            start_time = pd.to_datetime(df['timestamp'].iloc[0])
            end_time = pd.to_datetime(df['timestamp'].iloc[-1])
            summary['total_training_time'] = str(end_time - start_time)
        
        # Save summary
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create final plots
        self.plot_metrics()
        self._create_comparison_plot(df)
    
    def _create_comparison_plot(self, df: pd.DataFrame):
        """Create train vs validation comparison plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'train_loss' in df and 'val_loss' in df:
            # Calculate ratio
            ratio = df['val_loss'] / df['train_loss']
            
            ax.plot(df['epoch'], ratio, linewidth=2)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Val Loss / Train Loss')
            ax.set_title('Overfitting Monitor')
            ax.grid(True, alpha=0.3)
            
            # Add text annotation
            if ratio.iloc[-1] > 1.2:
                ax.text(0.95, 0.95, 'Potential Overfitting!', 
                       transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                       horizontalalignment='right',
                       verticalalignment='top')
        
        plot_path = self.plot_dir / "overfitting_monitor.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


class ModelProfiler:
    """Profile model performance and memory usage"""
    
    @staticmethod
    def profile_model(model: mx.Module, batch_size: int = 32, num_vertices: int = 1723):
        """Profile model performance"""
        # Create dummy input
        dummy_coords = mx.random.normal((batch_size, num_vertices, 3))
        dummy_features = mx.random.normal((batch_size, num_vertices, 6))
        
        # Warmup
        for _ in range(5):
            _ = model(dummy_coords, dummy_features)
        
        # Time forward pass
        import time
        num_runs = 20
        
        mx.eval(model.parameters())
        
        start_time = time.time()
        for _ in range(num_runs):
            embeddings, logits = model(dummy_coords, dummy_features)
            mx.eval(embeddings)
            mx.eval(logits)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_runs
        
        # Calculate throughput
        throughput = batch_size / avg_time
        
        # Memory usage
        if mx.metal.is_available():
            memory_mb = mx.metal.get_active_memory() / 1e6
        else:
            memory_mb = 0
        
        # Model statistics
        total_params = sum(p.size for p in model.parameters().values())
        param_memory_mb = sum(p.nbytes for p in model.parameters().values()) / 1e6
        
        profile_results = {
            'batch_size': batch_size,
            'num_vertices': num_vertices,
            'avg_forward_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'memory_usage_mb': memory_mb,
            'total_parameters': total_params,
            'parameter_memory_mb': param_memory_mb,
            'device': 'Metal' if mx.metal.is_available() else 'CPU'
        }
        
        return profile_results
    
    @staticmethod
    def create_profile_report(profile_results: Dict, output_path: str):
        """Create profiling report"""
        output_path = Path(output_path)
        
        # Create report
        report = []
        report.append("=" * 50)
        report.append("DT-GCNN Model Profiling Report")
        report.append("=" * 50)
        report.append(f"Device: {profile_results['device']}")
        report.append(f"Batch Size: {profile_results['batch_size']}")
        report.append(f"Number of Vertices: {profile_results['num_vertices']}")
        report.append("")
        report.append("Performance Metrics:")
        report.append(f"  - Average Forward Pass: {profile_results['avg_forward_time_ms']:.2f} ms")
        report.append(f"  - Throughput: {profile_results['throughput_samples_per_sec']:.1f} samples/sec")
        report.append("")
        report.append("Memory Usage:")
        report.append(f"  - Active Memory: {profile_results['memory_usage_mb']:.2f} MB")
        report.append(f"  - Parameter Memory: {profile_results['parameter_memory_mb']:.2f} MB")
        report.append("")
        report.append("Model Statistics:")
        report.append(f"  - Total Parameters: {profile_results['total_parameters']:,}")
        report.append("=" * 50)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write("\n".join(report))
        
        # Also save as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(profile_results, f, indent=2)


def visualize_embeddings(
    embeddings: mx.array,
    labels: mx.array,
    output_path: str,
    method: str = 'tsne',
    num_samples: int = 1000
):
    """Visualize learned embeddings using dimensionality reduction"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    
    # Convert to numpy
    embeddings_np = np.array(embeddings)
    labels_np = np.array(labels)
    
    # Sample if too many points
    if len(embeddings_np) > num_samples:
        indices = np.random.choice(len(embeddings_np), num_samples, replace=False)
        embeddings_np = embeddings_np[indices]
        labels_np = labels_np[indices]
    
    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings_np)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels_np,
        cmap='tab10',
        alpha=0.6,
        s=30
    )
    plt.colorbar(scatter)
    plt.title(f'DT-GCNN Embeddings Visualization ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test monitoring
    monitor = TrainingMonitor("./test_output")
    
    # Simulate training metrics
    for epoch in range(50):
        metrics = {
            'train_loss': 1.0 - epoch * 0.015 + np.random.normal(0, 0.05),
            'val_loss': 1.0 - epoch * 0.012 + np.random.normal(0, 0.08),
            'train_acc': min(0.95, epoch * 0.015 + np.random.normal(0, 0.02)),
            'val_acc': min(0.92, epoch * 0.013 + np.random.normal(0, 0.03)),
            'learning_rate': 0.001 * (0.95 ** epoch)
        }
        monitor.update(epoch, metrics)
    
    monitor.create_summary_report()