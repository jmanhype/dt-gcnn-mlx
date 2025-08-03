# DT-GCNN API Reference

## Table of Contents

1. [Core Classes](#core-classes)
2. [Model Components](#model-components)
3. [Data Pipeline](#data-pipeline)
4. [Training Utilities](#training-utilities)
5. [Loss Functions](#loss-functions)
6. [Graph Operations](#graph-operations)
7. [Utilities](#utilities)

## Core Classes

### DTGCNN

The main model class implementing Dynamic Triplet Graph Convolutional Neural Network.

```python
class DTGCNN(nn.Module):
    """
    Dynamic Triplet Graph Convolutional Neural Network.
    
    Args:
        input_dim (int): Input feature dimension (default: 768)
        hidden_dim (int): Hidden layer dimension (default: 256)
        output_dim (int): Output embedding dimension (default: 128)
        num_gcn_layers (int): Number of GCN layers (default: 3)
        dropout (float): Dropout rate (default: 0.1)
        encoder_name (str): Pre-trained encoder name (default: 'bert-base-uncased')
        freeze_encoder (bool): Freeze encoder weights (default: False)
        graph_k (int): Number of neighbors in graph (default: 10)
        graph_threshold (float): Similarity threshold for edges (default: 0.7)
    
    Example:
        >>> model = DTGCNN(
        ...     input_dim=768,
        ...     hidden_dim=256,
        ...     output_dim=128,
        ...     num_gcn_layers=3
        ... )
        >>> embeddings = model(input_ids, attention_mask)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_gcn_layers: int = 3,
        dropout: float = 0.1,
        encoder_name: str = 'bert-base-uncased',
        freeze_encoder: bool = False,
        graph_k: int = 10,
        graph_threshold: float = 0.7
    ):
        ...
    
    def forward(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        adjacency_matrix: Optional[mx.array] = None
    ) -> mx.array:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            adjacency_matrix: Pre-computed adjacency matrix (optional)
        
        Returns:
            embeddings: Output embeddings of shape (batch_size, output_dim)
        """
        ...
    
    def compute_graph(self, embeddings: mx.array) -> mx.array:
        """
        Dynamically compute graph adjacency matrix.
        
        Args:
            embeddings: Node embeddings of shape (batch_size, hidden_dim)
        
        Returns:
            adjacency: Normalized adjacency matrix
        """
        ...
```

### DTGCNNConfig

Configuration class for model initialization.

```python
@dataclass
class DTGCNNConfig:
    """
    Configuration for DT-GCNN model.
    
    Attributes:
        # Model architecture
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output embedding dimension
        num_gcn_layers: Number of GCN layers
        gcn_activation: Activation function for GCN layers
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout probability
        
        # Training parameters
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        num_epochs: Number of training epochs
        warmup_epochs: Number of warmup epochs
        
        # Loss configuration
        triplet_margin: Margin for triplet loss
        mining_strategy: Strategy for mining hard negatives
        loss_reduction: Reduction method for loss
        
        # Optimization
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
        scheduler: Learning rate scheduler type
        gradient_clip_val: Gradient clipping value
        gradient_accumulation_steps: Steps to accumulate gradients
        
        # Graph construction
        graph_k_neighbors: Number of neighbors in k-NN graph
        graph_similarity_threshold: Threshold for edge creation
        dynamic_graph_update_freq: Epochs between graph updates
        
        # Hardware
        device: Device to run on ('cpu', 'gpu')
        mixed_precision: Use mixed precision training
        num_workers: Number of data loading workers
    """
    
    # Model architecture
    input_dim: int = 768
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    output_dim: int = 128
    num_gcn_layers: int = 3
    gcn_activation: str = 'relu'
    use_batch_norm: bool = True
    dropout_rate: float = 0.1
    
    # Training parameters
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 20
    warmup_epochs: int = 2
    
    # ... (additional fields)
```

## Model Components

### EmbeddingLayer

Text embedding layer using pre-trained transformers.

```python
class EmbeddingLayer(nn.Module):
    """
    Embedding layer using pre-trained transformer models.
    
    Args:
        model_name: Name of pre-trained model
        pooling_strategy: How to pool token embeddings ('cls', 'mean', 'max')
        fine_tune: Whether to fine-tune the encoder
    
    Methods:
        forward(input_ids, attention_mask): Compute embeddings
        freeze(): Freeze encoder parameters
        unfreeze(): Unfreeze encoder parameters
    """
    
    def forward(
        self,
        input_ids: mx.array,
        attention_mask: mx.array
    ) -> mx.array:
        """
        Args:
            input_ids: Token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
        
        Returns:
            embeddings: Pooled embeddings (batch_size, hidden_dim)
        """
        ...
```

### GCNLayer

Single graph convolutional layer.

```python
class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to use bias
        activation: Activation function name
        use_batch_norm: Whether to apply batch normalization
    
    Forward args:
        features: Node features (batch_size, in_features)
        adjacency: Adjacency matrix (batch_size, batch_size)
    
    Returns:
        Updated features (batch_size, out_features)
    """
    
    def forward(
        self,
        features: mx.array,
        adjacency: mx.array
    ) -> mx.array:
        ...
```

### GraphConstructor

Dynamic graph construction module.

```python
class GraphConstructor:
    """
    Constructs graphs from embeddings.
    
    Args:
        k_neighbors: Number of neighbors for k-NN
        similarity_metric: Similarity metric ('cosine', 'euclidean')
        threshold: Similarity threshold for edge creation
        normalize: Whether to normalize adjacency matrix
    
    Methods:
        construct_knn_graph: Build k-NN graph
        construct_threshold_graph: Build threshold-based graph
        construct_hybrid_graph: Combine k-NN and threshold
    """
    
    def construct_knn_graph(
        self,
        embeddings: mx.array,
        exclude_self: bool = True
    ) -> mx.array:
        """
        Construct k-nearest neighbors graph.
        
        Args:
            embeddings: Node embeddings (batch_size, embedding_dim)
            exclude_self: Whether to exclude self-loops
        
        Returns:
            adjacency: Sparse adjacency matrix
        """
        ...
```

## Data Pipeline

### TripletDataset

Dataset class for triplet data.

```python
class TripletDataset(Dataset):
    """
    Dataset for triplet training.
    
    Args:
        data_path: Path to data file (JSON or CSV)
        tokenizer: Tokenizer instance or name
        max_length: Maximum sequence length
        cache_dir: Directory for caching processed data
    
    Methods:
        __len__: Return dataset size
        __getitem__: Get triplet by index
        create_dataloaders: Create train/val dataloaders
        get_hard_negatives: Mine hard negatives
    """
    
    def __getitem__(self, idx: int) -> Dict[str, mx.array]:
        """
        Get a triplet sample.
        
        Returns:
            dict with keys:
                - anchor_ids: Token IDs for anchor
                - anchor_mask: Attention mask for anchor
                - positive_ids: Token IDs for positive
                - positive_mask: Attention mask for positive
                - negative_ids: Token IDs for negative
                - negative_mask: Attention mask for negative
        """
        ...
```

### DataCollator

Custom collator for batch processing.

```python
class TripletCollator:
    """
    Collates triplet samples into batches.
    
    Args:
        padding: Padding strategy ('max_length', 'longest')
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
    
    Methods:
        __call__: Collate list of samples into batch
    """
    
    def __call__(
        self,
        samples: List[Dict[str, mx.array]]
    ) -> Dict[str, mx.array]:
        """
        Collate samples into batch.
        
        Args:
            samples: List of triplet dictionaries
        
        Returns:
            Batched tensors with same keys as input
        """
        ...
```

### TripletGenerator

Generate triplets from labeled data.

```python
class TripletGenerator:
    """
    Generate triplets from sentence-label pairs.
    
    Args:
        similarity_threshold: Threshold for positive pairs
        negative_sampling: Strategy for negative sampling
        random_seed: Random seed for reproducibility
    
    Methods:
        generate_triplets: Create triplets from data
        generate_hard_triplets: Mine hard triplets
        balance_triplets: Balance positive/negative ratios
    """
    
    def generate_triplets(
        self,
        sentences: List[str],
        labels: List[int],
        num_triplets_per_anchor: int = 5
    ) -> List[Tuple[str, str, str]]:
        """
        Generate triplets from labeled sentences.
        
        Args:
            sentences: List of input sentences
            labels: Corresponding labels
            num_triplets_per_anchor: Triplets to generate per anchor
        
        Returns:
            List of (anchor, positive, negative) tuples
        """
        ...
```

## Training Utilities

### Trainer

Main training class.

```python
class Trainer:
    """
    Training manager for DT-GCNN.
    
    Args:
        model: DTGCNN model instance
        config: Training configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        callbacks: List of callback instances
    
    Methods:
        train: Run training loop
        validate: Run validation
        save_checkpoint: Save model state
        load_checkpoint: Load model state
    """
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            num_epochs: Override config epochs
            resume_from: Path to checkpoint to resume from
        
        Returns:
            Dictionary of metric histories
        """
        ...
```

### Callbacks

Training callbacks for monitoring and control.

```python
class EarlyStopping:
    """
    Early stopping callback.
    
    Args:
        patience: Epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        monitor: Metric to monitor
        mode: 'min' or 'max'
    """
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Dict[str, float]
    ) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop
        """
        ...

class ModelCheckpoint:
    """
    Save model checkpoints.
    
    Args:
        filepath: Path pattern for saving
        monitor: Metric to monitor
        save_best_only: Only save best model
        save_top_k: Save top k models
    """
    
    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        logs: Dict[str, float]
    ):
        ...
```

## Loss Functions

### TripletLoss

Base triplet loss implementation.

```python
class TripletLoss(nn.Module):
    """
    Triplet loss with configurable distance metrics.
    
    Args:
        margin: Margin for triplet loss
        distance_metric: Distance metric ('euclidean', 'cosine')
        reduction: Reduction method ('mean', 'sum', 'none')
        swap: Whether to use swap loss
    
    Forward args:
        anchor: Anchor embeddings (batch_size, embedding_dim)
        positive: Positive embeddings (batch_size, embedding_dim)
        negative: Negative embeddings (batch_size, embedding_dim)
    
    Returns:
        loss: Scalar loss value
    """
    
    def forward(
        self,
        anchor: mx.array,
        positive: mx.array,
        negative: mx.array
    ) -> mx.array:
        ...
```

### Mining Strategies

Hard negative mining implementations.

```python
class BatchHardMiner:
    """
    Batch hard mining strategy.
    
    Methods:
        mine: Mine hardest positive and negative for each anchor
    """
    
    def mine(
        self,
        embeddings: mx.array,
        labels: mx.array
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Mine hard triplets from batch.
        
        Args:
            embeddings: All embeddings in batch
            labels: Corresponding labels
        
        Returns:
            Tuple of (anchors, positives, negatives)
        """
        ...

class OnlineTripletMiner:
    """
    Online triplet mining during training.
    
    Args:
        margin: Triplet margin
        strategy: Mining strategy ('all', 'hard', 'semi-hard')
    """
    
    def mine(
        self,
        embeddings: mx.array,
        labels: mx.array
    ) -> List[Tuple[int, int, int]]:
        """
        Mine triplet indices.
        
        Returns:
            List of (anchor_idx, positive_idx, negative_idx)
        """
        ...
```

## Graph Operations

### Graph Utilities

Helper functions for graph operations.

```python
def normalize_adjacency(
    adjacency: mx.array,
    add_self_loops: bool = True,
    symmetric: bool = True
) -> mx.array:
    """
    Normalize adjacency matrix for GCN.
    
    Args:
        adjacency: Input adjacency matrix
        add_self_loops: Whether to add self-connections
        symmetric: Use symmetric normalization
    
    Returns:
        Normalized adjacency matrix
    """
    ...

def compute_laplacian(
    adjacency: mx.array,
    normalized: bool = True
) -> mx.array:
    """
    Compute graph Laplacian.
    
    Args:
        adjacency: Adjacency matrix
        normalized: Use normalized Laplacian
    
    Returns:
        Laplacian matrix
    """
    ...

def graph_statistics(adjacency: mx.array) -> Dict[str, float]:
    """
    Compute graph statistics.
    
    Returns:
        Dictionary with:
            - num_edges: Total number of edges
            - density: Graph density
            - avg_degree: Average node degree
            - clustering_coef: Clustering coefficient
    """
    ...
```

## Utilities

### Metrics

Evaluation metrics for embeddings.

```python
class EmbeddingMetrics:
    """
    Metrics for evaluating embedding quality.
    
    Methods:
        triplet_accuracy: Accuracy of triplet ordering
        mean_average_precision: MAP for retrieval
        embedding_diversity: Measure embedding spread
        silhouette_score: Cluster quality
    """
    
    @staticmethod
    def triplet_accuracy(
        anchor: mx.array,
        positive: mx.array,
        negative: mx.array
    ) -> float:
        """
        Compute triplet accuracy.
        
        Returns:
            Fraction of correctly ordered triplets
        """
        ...
    
    @staticmethod
    def retrieval_metrics(
        query_embeddings: mx.array,
        gallery_embeddings: mx.array,
        query_labels: mx.array,
        gallery_labels: mx.array,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics.
        
        Returns:
            Dictionary with precision@k, recall@k, mAP
        """
        ...
```

### Visualization

Tools for visualizing embeddings and graphs.

```python
class EmbeddingVisualizer:
    """
    Visualize embeddings in 2D/3D.
    
    Methods:
        plot_2d: Create 2D scatter plot
        plot_3d: Create 3D scatter plot
        plot_tsne: t-SNE visualization
        plot_umap: UMAP visualization
    """
    
    @staticmethod
    def plot_tsne(
        embeddings: mx.array,
        labels: Optional[mx.array] = None,
        perplexity: int = 30,
        save_path: Optional[str] = None
    ):
        """
        Create t-SNE visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Optional labels for coloring
            perplexity: t-SNE perplexity parameter
            save_path: Path to save figure
        """
        ...

class GraphVisualizer:
    """
    Visualize graph structure.
    
    Methods:
        plot_adjacency: Visualize adjacency matrix
        plot_graph_layout: Create graph layout visualization
        plot_degree_distribution: Plot node degree distribution
    """
    
    @staticmethod
    def plot_graph_layout(
        adjacency: mx.array,
        node_features: Optional[mx.array] = None,
        layout: str = 'spring',
        save_path: Optional[str] = None
    ):
        """
        Visualize graph structure.
        
        Args:
            adjacency: Adjacency matrix
            node_features: Optional features for coloring
            layout: Layout algorithm
            save_path: Path to save figure
        """
        ...
```

### Model Export

Export trained models for deployment.

```python
class ModelExporter:
    """
    Export models for inference.
    
    Methods:
        export_mlx: Export for MLX inference
        export_coreml: Export to CoreML
        export_onnx: Export to ONNX
        quantize: Quantize model for efficiency
    """
    
    @staticmethod
    def export_mlx(
        model: DTGCNN,
        save_path: str,
        example_input: Optional[Dict[str, mx.array]] = None
    ):
        """
        Export model for MLX inference.
        
        Args:
            model: Trained model
            save_path: Path to save exported model
            example_input: Example input for tracing
        """
        ...
    
    @staticmethod
    def export_coreml(
        model: DTGCNN,
        save_path: str,
        input_names: List[str] = ['input_ids', 'attention_mask'],
        output_names: List[str] = ['embeddings']
    ):
        """
        Export to CoreML for iOS/macOS deployment.
        """
        ...
```

## Example Usage

### Complete Training Pipeline

```python
from dt_gcnn_mlx import (
    DTGCNN, DTGCNNConfig, TripletDataset,
    Trainer, EarlyStopping, ModelCheckpoint,
    EmbeddingMetrics, EmbeddingVisualizer
)

# Configuration
config = DTGCNNConfig(
    input_dim=768,
    hidden_dims=[512, 256, 128],
    output_dim=128,
    num_gcn_layers=3,
    batch_size=512,
    learning_rate=1e-4,
    num_epochs=50
)

# Create model
model = DTGCNN(
    input_dim=config.input_dim,
    hidden_dim=config.hidden_dims[0],
    output_dim=config.output_dim,
    num_gcn_layers=config.num_gcn_layers
)

# Load data
dataset = TripletDataset("data/triplets.json")
train_loader, val_loader = dataset.create_dataloaders(
    batch_size=config.batch_size,
    train_split=0.9
)

# Setup callbacks
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss'),
    ModelCheckpoint(
        filepath='checkpoints/model_{epoch}_{val_loss:.4f}.pt',
        save_best_only=True
    )
]

# Train
trainer = Trainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=callbacks
)

history = trainer.train()

# Evaluate
metrics = EmbeddingMetrics()
val_embeddings = trainer.get_embeddings(val_loader)
val_metrics = metrics.compute_all_metrics(val_embeddings)

# Visualize
visualizer = EmbeddingVisualizer()
visualizer.plot_tsne(
    val_embeddings['embeddings'],
    val_embeddings['labels'],
    save_path='embeddings_tsne.png'
)

# Export model
from dt_gcnn_mlx import ModelExporter
exporter = ModelExporter()
exporter.export_mlx(model, 'model.mlx')
exporter.export_coreml(model, 'model.mlmodel')
```

### Custom Extensions

```python
# Custom GCN layer
class CustomGCNLayer(GCNLayer):
    def forward(self, features, adjacency):
        # Custom graph convolution logic
        h = super().forward(features, adjacency)
        # Additional processing
        return h

# Custom loss function
class ContrastiveTripletLoss(TripletLoss):
    def __init__(self, margin=0.2, temperature=0.1):
        super().__init__(margin)
        self.temperature = temperature
    
    def forward(self, anchor, positive, negative):
        # Standard triplet loss
        triplet_loss = super().forward(anchor, positive, negative)
        
        # Additional contrastive term
        similarity = mx.matmul(anchor, positive.T) / self.temperature
        contrastive_loss = -mx.log_softmax(similarity, axis=1).diagonal().mean()
        
        return triplet_loss + contrastive_loss

# Use custom components
model = DTGCNN(
    gcn_layer_class=CustomGCNLayer,
    loss_fn=ContrastiveTripletLoss()
)
```

## Type Annotations

All functions and classes use type hints for better IDE support:

```python
from typing import Optional, List, Dict, Tuple, Union
import mlx.core as mx

def example_function(
    embeddings: mx.array,
    labels: Optional[mx.array] = None,
    config: Optional[DTGCNNConfig] = None
) -> Dict[str, Union[float, mx.array]]:
    """Example with full type annotations."""
    ...
```