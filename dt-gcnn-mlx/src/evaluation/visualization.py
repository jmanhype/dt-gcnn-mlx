"""Visualization utilities for model analysis."""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List, Optional, Dict, Tuple, Union
import warnings

# Optional imports
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("UMAP not installed. Use 'pip install umap-learn' for UMAP visualization.")


def plot_confusion_matrix(cm: Union[mx.array, np.ndarray], 
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False,
                         cmap: str = 'Blues',
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Names for each class
        normalize: Whether to normalize the matrix
        cmap: Colormap to use
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Convert to numpy if needed
    if hasattr(cm, 'numpy'):
        cm = np.array(cm)
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
        
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, square=True,
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def plot_embeddings_tsne(embeddings: Union[mx.array, np.ndarray],
                        labels: Union[mx.array, np.ndarray],
                        class_names: Optional[List[str]] = None,
                        perplexity: int = 30,
                        n_iter: int = 1000,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None,
                        title: str = "t-SNE Embedding Visualization") -> None:
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels for each embedding
        class_names: Names for each class
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        figsize: Figure size
        save_path: Path to save the plot
        title: Plot title
    """
    # Convert to numpy
    if hasattr(embeddings, 'numpy'):
        embeddings = np.array(embeddings)
    if hasattr(labels, 'numpy'):
        labels = np.array(labels)
        
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=figsize)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
        
    for label, color, name in zip(unique_labels, colors, class_names):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color], label=name, alpha=0.6, s=50)
        
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()


def plot_embeddings_umap(embeddings: Union[mx.array, np.ndarray],
                        labels: Union[mx.array, np.ndarray],
                        class_names: Optional[List[str]] = None,
                        n_neighbors: int = 15,
                        min_dist: float = 0.1,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None,
                        title: str = "UMAP Embedding Visualization") -> None:
    """
    Visualize embeddings using UMAP.
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels for each embedding
        class_names: Names for each class
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        figsize: Figure size
        save_path: Path to save the plot
        title: Plot title
    """
    if not HAS_UMAP:
        warnings.warn("UMAP not available. Using t-SNE instead.")
        plot_embeddings_tsne(embeddings, labels, class_names, 
                           figsize=figsize, save_path=save_path, 
                           title=title.replace("UMAP", "t-SNE"))
        return
        
    # Convert to numpy
    if hasattr(embeddings, 'numpy'):
        embeddings = np.array(embeddings)
    if hasattr(labels, 'numpy'):
        labels = np.array(labels)
        
    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=figsize)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
        
    for label, color, name in zip(unique_labels, colors, class_names):
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[color], label=name, alpha=0.6, s=50)
        
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"UMAP plot saved to {save_path}")
    else:
        plt.show()


def plot_training_history(history: Dict[str, List[float]],
                         metrics: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (12, 4),
                         save_path: Optional[str] = None) -> None:
    """
    Plot training history with multiple metrics.
    
    Args:
        history: Dictionary with metric names as keys and lists of values
        metrics: Specific metrics to plot (default: all)
        figsize: Figure size
        save_path: Path to save the plot
    """
    if metrics is None:
        metrics = list(history.keys())
        
    # Separate loss and accuracy metrics
    loss_metrics = [m for m in metrics if 'loss' in m.lower()]
    acc_metrics = [m for m in metrics if 'acc' in m.lower()]
    other_metrics = [m for m in metrics if m not in loss_metrics + acc_metrics]
    
    n_plots = len([loss_metrics, acc_metrics, other_metrics]) - [loss_metrics, acc_metrics, other_metrics].count([])
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
        
    plot_idx = 0
    
    # Plot loss metrics
    if loss_metrics:
        ax = axes[plot_idx]
        for metric in loss_metrics:
            ax.plot(history[metric], label=metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
        
    # Plot accuracy metrics
    if acc_metrics:
        ax = axes[plot_idx]
        for metric in acc_metrics:
            ax.plot(history[metric], label=metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
        
    # Plot other metrics
    if other_metrics:
        ax = axes[plot_idx]
        for metric in other_metrics:
            ax.plot(history[metric], label=metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Other Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def plot_attention_weights(attention_weights: Union[mx.array, np.ndarray],
                          node_labels: Optional[List[str]] = None,
                          time_steps: Optional[List[int]] = None,
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> None:
    """
    Visualize attention weights as heatmap.
    
    Args:
        attention_weights: Attention weight matrix
        node_labels: Labels for nodes
        time_steps: Labels for time steps
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Convert to numpy
    if hasattr(attention_weights, 'numpy'):
        attention_weights = np.array(attention_weights)
        
    # Handle different dimensions
    if len(attention_weights.shape) == 2:
        # Simple 2D attention
        weights = attention_weights
    elif len(attention_weights.shape) == 3:
        # Multi-head attention - average over heads
        weights = np.mean(attention_weights, axis=0)
    elif len(attention_weights.shape) == 4:
        # Batch dimension - take first sample
        weights = np.mean(attention_weights[0], axis=0)
    else:
        raise ValueError(f"Unsupported attention weight shape: {attention_weights.shape}")
        
    n_nodes, n_time = weights.shape
    
    if node_labels is None:
        node_labels = [f'Node {i}' for i in range(n_nodes)]
    if time_steps is None:
        time_steps = list(range(n_time))
        
    plt.figure(figsize=figsize)
    sns.heatmap(weights, 
                xticklabels=time_steps,
                yticklabels=node_labels,
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Weight'})
    
    plt.xlabel('Time Step')
    plt.ylabel('Node')
    plt.title('Attention Weight Visualization')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention plot saved to {save_path}")
    else:
        plt.show()


def plot_graph_importance(importance_scores: Union[mx.array, np.ndarray],
                         adjacency_matrix: Union[mx.array, np.ndarray],
                         node_labels: Optional[List[str]] = None,
                         threshold: float = 0.1,
                         figsize: Tuple[int, int] = (10, 10),
                         save_path: Optional[str] = None) -> None:
    """
    Visualize node importance on graph structure.
    
    Args:
        importance_scores: Importance score for each node
        adjacency_matrix: Graph adjacency matrix
        node_labels: Labels for nodes
        threshold: Edge weight threshold for visualization
        figsize: Figure size
        save_path: Path to save the plot
    """
    try:
        import networkx as nx
    except ImportError:
        warnings.warn("NetworkX not installed. Cannot create graph visualization.")
        return
        
    # Convert to numpy
    if hasattr(importance_scores, 'numpy'):
        importance_scores = np.array(importance_scores)
    if hasattr(adjacency_matrix, 'numpy'):
        adjacency_matrix = np.array(adjacency_matrix)
        
    # Create graph
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Remove weak edges
    edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') 
                       if w < threshold]
    G.remove_edges_from(edges_to_remove)
    
    # Set node labels
    if node_labels:
        labels = {i: label for i, label in enumerate(node_labels)}
    else:
        labels = {i: f'{i}' for i in range(len(importance_scores))}
        
    # Normalize importance scores for color mapping
    norm_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min())
    
    plt.figure(figsize=figsize)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color=norm_scores,
                          node_size=1000 * (1 + norm_scores),
                          cmap='Reds',
                          vmin=0, vmax=1)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, label='Importance Score')
    
    plt.title('Node Importance Visualization')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph importance plot saved to {save_path}")
    else:
        plt.show()