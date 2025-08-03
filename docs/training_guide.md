# DT-GCNN Training Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Training Configuration](#training-configuration)
4. [Training Strategies](#training-strategies)
5. [Monitoring and Debugging](#monitoring-and-debugging)
6. [Performance Optimization](#performance-optimization)
7. [Common Issues and Solutions](#common-issues-and-solutions)

## Getting Started

### Quick Start Training

```python
from dt_gcnn_mlx import DTGCNN, Trainer, TripletDataset

# Load your data
dataset = TripletDataset("data/triplets.json")
train_loader, val_loader = dataset.create_dataloaders(
    batch_size=512,
    train_split=0.9
)

# Initialize model
model = DTGCNN(
    input_dim=768,
    hidden_dim=256,
    output_dim=128,
    num_gcn_layers=3
)

# Create trainer and train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20
)

trainer.train()
```

### Advanced Training Setup

```python
from dt_gcnn_mlx import DTGCNNConfig, create_model, AdvancedTrainer

# Detailed configuration
config = DTGCNNConfig(
    # Model architecture
    input_dim=768,
    hidden_dims=[512, 256, 128],
    num_gcn_layers=3,
    gcn_activation='gelu',
    use_batch_norm=True,
    dropout_rate=0.1,
    
    # Training parameters
    batch_size=512,
    learning_rate=1e-4,
    weight_decay=1e-5,
    num_epochs=50,
    warmup_epochs=5,
    
    # Loss configuration
    triplet_margin=0.2,
    mining_strategy='batch_hard',
    loss_reduction='mean',
    
    # Optimization
    optimizer='adamw',
    scheduler='cosine',
    gradient_clip_val=1.0,
    gradient_accumulation_steps=4,
    
    # Graph construction
    graph_k_neighbors=15,
    graph_similarity_threshold=0.7,
    dynamic_graph_update_freq=10,
    
    # Hardware optimization
    mixed_precision=True,
    num_workers=8,
    prefetch_factor=2,
    persistent_workers=True
)

# Create model and trainer
model = create_model(config)
trainer = AdvancedTrainer(model, config)

# Train with callbacks
trainer.train(
    train_loader,
    val_loader,
    callbacks=[
        EarlyStopping(patience=5),
        ModelCheckpoint(save_top_k=3),
        LearningRateMonitor(),
        TensorboardLogger()
    ]
)
```

## Data Preparation

### Dataset Format

DT-GCNN expects data in triplet format. Each triplet consists of:
- **Anchor**: Reference text
- **Positive**: Semantically similar text
- **Negative**: Semantically dissimilar text

#### JSON Format

```json
{
  "triplets": [
    {
      "anchor": "The cat sat on the mat",
      "positive": "A feline rested on the rug",
      "negative": "The stock market crashed today"
    },
    {
      "anchor": "Machine learning is fascinating",
      "positive": "AI and deep learning are exciting fields",
      "negative": "I enjoy cooking Italian food"
    }
  ]
}
```

#### CSV Format

```csv
anchor,positive,negative
"The cat sat on the mat","A feline rested on the rug","The stock market crashed today"
"Machine learning is fascinating","AI and deep learning are exciting fields","I enjoy cooking Italian food"
```

### Data Preprocessing

```python
from dt_gcnn_mlx.data import DataPreprocessor, TripletGenerator

# Automatic triplet generation from labeled data
preprocessor = DataPreprocessor(
    tokenizer='sentence-piece',
    max_length=128,
    vocab_size=32000
)

# Generate triplets from sentence pairs with labels
generator = TripletGenerator(
    similarity_threshold=0.8,
    negative_sampling_strategy='hard'
)

# Process raw text data
raw_sentences = ["sentence1", "sentence2", ...]
labels = [0, 0, 1, 1, ...]  # Class labels

triplets = generator.generate_triplets(
    sentences=raw_sentences,
    labels=labels,
    num_triplets_per_anchor=5
)

# Save processed data
preprocessor.save_triplets(triplets, "data/processed_triplets.json")
```

### Data Augmentation

```python
from dt_gcnn_mlx.data import TripletAugmenter

augmenter = TripletAugmenter(
    strategies=['paraphrase', 'back_translation', 'synonym_replacement'],
    augmentation_prob=0.3
)

# Augment existing triplets
augmented_dataset = augmenter.augment_dataset(
    original_dataset,
    num_augmentations=2
)
```

## Training Configuration

### Hyperparameter Selection

#### Model Architecture
- **Input dimension**: Match your encoder output (768 for BERT-base)
- **Hidden dimensions**: Gradually decrease (e.g., [512, 256, 128])
- **GCN layers**: 2-4 layers typically sufficient
- **Output dimension**: 64-256 depending on downstream task

#### Training Parameters
- **Batch size**: Larger is better for triplet mining (256-1024)
- **Learning rate**: Start with 1e-4, adjust based on convergence
- **Weight decay**: 1e-5 to 1e-4 for regularization
- **Warmup**: 5-10% of total epochs

#### Graph Construction
- **k-neighbors**: 10-20 for sparse graphs
- **Similarity threshold**: 0.6-0.8 depending on data
- **Update frequency**: Every 5-10 epochs for efficiency

### Configuration Templates

#### Small Dataset (<10k samples)
```python
small_config = DTGCNNConfig(
    batch_size=128,
    learning_rate=5e-5,
    num_epochs=100,
    dropout_rate=0.3,
    weight_decay=1e-4,
    graph_k_neighbors=5,
    early_stopping_patience=10
)
```

#### Medium Dataset (10k-100k samples)
```python
medium_config = DTGCNNConfig(
    batch_size=512,
    learning_rate=1e-4,
    num_epochs=50,
    dropout_rate=0.1,
    weight_decay=1e-5,
    graph_k_neighbors=10,
    gradient_accumulation_steps=2
)
```

#### Large Dataset (>100k samples)
```python
large_config = DTGCNNConfig(
    batch_size=1024,
    learning_rate=2e-4,
    num_epochs=20,
    dropout_rate=0.05,
    weight_decay=1e-6,
    graph_k_neighbors=20,
    mixed_precision=True,
    gradient_accumulation_steps=4
)
```

## Training Strategies

### 1. Progressive Training

Train the model in stages for better convergence:

```python
class ProgressiveTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def train_progressive(self, train_loader, val_loader):
        # Stage 1: Train embedding layer only
        self.model.freeze_gcn_layers()
        self.train_stage(
            train_loader, val_loader,
            epochs=5, lr=1e-3
        )
        
        # Stage 2: Train GCN layers with frozen embeddings
        self.model.unfreeze_gcn_layers()
        self.model.freeze_embedding_layer()
        self.train_stage(
            train_loader, val_loader,
            epochs=10, lr=5e-4
        )
        
        # Stage 3: Fine-tune entire model
        self.model.unfreeze_all()
        self.train_stage(
            train_loader, val_loader,
            epochs=self.config.num_epochs,
            lr=self.config.learning_rate
        )
```

### 2. Curriculum Learning

Start with easy triplets and gradually increase difficulty:

```python
class CurriculumScheduler:
    def __init__(self, dataset, num_stages=4):
        self.dataset = dataset
        self.num_stages = num_stages
        self.current_stage = 0
    
    def get_stage_data(self):
        # Sort triplets by difficulty (similarity difference)
        difficulties = self.compute_triplet_difficulties()
        
        # Return subset based on current stage
        stage_size = len(self.dataset) // self.num_stages
        start_idx = self.current_stage * stage_size
        end_idx = min(
            (self.current_stage + 1) * stage_size,
            len(self.dataset)
        )
        
        return self.dataset[difficulties[start_idx:end_idx]]
    
    def advance_stage(self):
        self.current_stage = min(
            self.current_stage + 1,
            self.num_stages - 1
        )
```

### 3. Multi-Task Learning

Train on multiple objectives simultaneously:

```python
class MultiTaskDTGCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dtgcnn = DTGCNN(config)
        
        # Additional task heads
        self.classification_head = nn.Linear(
            config.output_dim, config.num_classes
        )
        self.regression_head = nn.Linear(
            config.output_dim, 1
        )
    
    def forward(self, x, task='triplet'):
        embeddings = self.dtgcnn(x)
        
        if task == 'triplet':
            return embeddings
        elif task == 'classification':
            return self.classification_head(embeddings)
        elif task == 'regression':
            return self.regression_head(embeddings)
```

## Monitoring and Debugging

### Training Metrics

Monitor these key metrics during training:

```python
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'loss': [],
            'triplet_accuracy': [],
            'mean_positive_distance': [],
            'mean_negative_distance': [],
            'hard_negative_ratio': [],
            'gradient_norm': []
        }
    
    def update(self, batch_metrics):
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)
    
    def plot_metrics(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for (key, values), ax in zip(self.metrics.items(), axes.flat):
            ax.plot(values)
            ax.set_title(key)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
        
        plt.tight_layout()
        plt.show()
```

### Embedding Visualization

Visualize learned embeddings using t-SNE or UMAP:

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(model, data_loader, num_samples=1000):
    embeddings = []
    labels = []
    
    with mx.no_grad():
        for batch in data_loader:
            if len(embeddings) >= num_samples:
                break
            
            emb = model(batch['anchor'])
            embeddings.append(emb)
            labels.extend(batch['labels'])
    
    # Reduce dimensionality
    embeddings = mx.concatenate(embeddings)[:num_samples]
    embeddings_2d = TSNE(n_components=2).fit_transform(
        embeddings.numpy()
    )
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels[:num_samples],
        cmap='tab10',
        alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Embeddings')
    plt.show()
```

### Debugging Tools

```python
class DebugCallback:
    def on_batch_end(self, trainer, outputs, batch, batch_idx):
        # Check for NaN values
        if mx.any(mx.isnan(outputs['loss'])):
            print(f"NaN detected in batch {batch_idx}")
            self.save_debug_info(trainer, batch)
        
        # Monitor gradient magnitudes
        grad_norms = [
            mx.linalg.norm(p.grad).item()
            for p in trainer.model.parameters()
            if p.grad is not None
        ]
        
        if max(grad_norms) > 100:
            print(f"Large gradient detected: {max(grad_norms)}")
        
        # Check embedding collapse
        embeddings = outputs['embeddings']
        emb_std = mx.std(embeddings, axis=0).mean()
        if emb_std < 0.01:
            print("Warning: Embedding collapse detected")
```

## Performance Optimization

### 1. Hardware Optimization

#### Apple Silicon Specific
```python
# Optimize for M1/M2/M3
import mlx.core as mx

# Set optimal thread count
mx.set_default_device(mx.gpu)
mx.core.set_num_threads(8)  # Adjust based on your chip

# Memory allocation strategy
class OptimizedDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Pre-allocate buffers
        self.buffer_size = batch_size * 2
        self.anchor_buffer = mx.zeros(
            (self.buffer_size, 768),
            dtype=mx.float32
        )
```

#### Memory Management
```python
# Gradient checkpointing for large models
@mx.checkpoint
def forward_with_checkpoint(model, inputs):
    return model(inputs)

# Clear cache periodically
class MemoryManager:
    def __init__(self, clear_every=10):
        self.clear_every = clear_every
        self.step = 0
    
    def step(self):
        self.step += 1
        if self.step % self.clear_every == 0:
            mx.clear_cache()
```

### 2. Training Speed Optimization

#### Mixed Precision Training
```python
class MixedPrecisionTrainer:
    def __init__(self, model, use_amp=True):
        self.model = model
        self.use_amp = use_amp
        self.scaler = mx.amp.GradScaler()
    
    def training_step(self, batch):
        with mx.amp.autocast(enabled=self.use_amp):
            outputs = self.model(batch)
            loss = self.compute_loss(outputs)
        
        # Scale loss and compute gradients
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        
        # Unscale and clip gradients
        self.scaler.unscale_(self.optimizer)
        mx.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0
        )
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
```

#### Distributed Data Parallel
```python
# For Mac Studio with multiple GPUs
class DistributedTrainer:
    def __init__(self, model, devices=['gpu:0', 'gpu:1']):
        self.devices = devices
        self.model = mx.nn.DataParallel(
            model, device_ids=devices
        )
    
    def train_epoch(self, data_loader):
        for batch in data_loader:
            # Automatically distributed across devices
            outputs = self.model(batch)
            loss = outputs.mean()  # Aggregate from devices
            loss.backward()
            self.optimizer.step()
```

### 3. Graph Construction Optimization

```python
class EfficientGraphConstructor:
    def __init__(self, k=10, batch_process=True):
        self.k = k
        self.batch_process = batch_process
    
    def construct_graph_batch(self, embeddings):
        batch_size = embeddings.shape[0]
        
        if self.batch_process:
            # Efficient batch similarity computation
            # Using matrix multiplication
            norm = mx.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / norm
            similarity = mx.matmul(normalized, normalized.T)
            
            # Efficient k-NN selection
            # Use top-k instead of full sort
            _, indices = mx.topk(similarity, k=self.k+1, axis=1)
            
            # Build sparse adjacency matrix
            row_indices = mx.repeat(
                mx.arange(batch_size), self.k
            )
            col_indices = indices[:, 1:].flatten()  # Exclude self
            
            adjacency = mx.sparse.coo_matrix(
                (mx.ones(len(row_indices)), 
                 (row_indices, col_indices)),
                shape=(batch_size, batch_size)
            )
            
            return adjacency
```

## Common Issues and Solutions

### 1. Training Instability

**Problem**: Loss explodes or becomes NaN

**Solutions**:
```python
# 1. Gradient clipping
mx.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Lower learning rate
optimizer = mx.optim.AdamW(model.parameters(), lr=1e-5)

# 3. Batch normalization
class StableDTGCNN(DTGCNN):
    def __init__(self, config):
        super().__init__(config)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in config.hidden_dims
        ])

# 4. Loss scaling
loss = triplet_loss * 0.1  # Scale down if needed
```

### 2. Poor Convergence

**Problem**: Model doesn't improve after initial epochs

**Solutions**:
```python
# 1. Learning rate scheduling
scheduler = mx.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=num_epochs,
    steps_per_epoch=len(train_loader)
)

# 2. Better initialization
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

model.apply(init_weights)

# 3. Warmup training
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps=1000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr_scale = self.step_count / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_scale
```

### 3. Overfitting

**Problem**: Training loss decreases but validation loss increases

**Solutions**:
```python
# 1. Regularization
config = DTGCNNConfig(
    dropout_rate=0.3,
    weight_decay=1e-4,
    use_batch_norm=True
)

# 2. Data augmentation
augmenter = TripletAugmenter(
    strategies=['paraphrase', 'noise_injection'],
    augmentation_prob=0.5
)

# 3. Early stopping
early_stopper = EarlyStopping(
    patience=5,
    min_delta=0.001,
    monitor='val_loss'
)

# 4. Ensemble training
models = [DTGCNN(config) for _ in range(3)]
ensemble_predictions = mx.mean(
    [model(x) for model in models], axis=0
)
```

### 4. Memory Issues

**Problem**: Out of memory errors on large datasets

**Solutions**:
```python
# 1. Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    outputs = model(batch)
    loss = criterion(outputs) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 2. Reduce batch size dynamically
class AdaptiveBatchSizeTrainer:
    def __init__(self, initial_batch_size=512):
        self.batch_size = initial_batch_size
    
    def adjust_batch_size(self, memory_usage):
        if memory_usage > 0.9:  # 90% memory used
            self.batch_size = self.batch_size // 2
            print(f"Reduced batch size to {self.batch_size}")

# 3. Offload to CPU
class CPUOffloadOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.cpu_state = None
    
    def step(self):
        # Move gradients to GPU only when needed
        self.move_to_gpu()
        self.optimizer.step()
        self.move_to_cpu()
```

## Best Practices Summary

1. **Start Simple**: Begin with basic configuration and add complexity
2. **Monitor Everything**: Track multiple metrics, not just loss
3. **Validate Often**: Use validation set to catch overfitting early
4. **Save Checkpoints**: Save model state regularly
5. **Experiment Tracking**: Use tools like Weights & Biases or TensorBoard
6. **Reproducibility**: Set random seeds and log configurations

## Advanced Tips

### Custom Loss Functions
```python
class FocalTripletLoss(nn.Module):
    """Focuses on hard examples"""
    def __init__(self, margin=0.2, gamma=2.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
    
    def forward(self, anchor, positive, negative):
        d_ap = mx.linalg.norm(anchor - positive, axis=1)
        d_an = mx.linalg.norm(anchor - negative, axis=1)
        
        losses = mx.maximum(0.0, d_ap - d_an + self.margin)
        
        # Apply focal weighting
        weights = (1 - mx.exp(-losses)) ** self.gamma
        weighted_losses = weights * losses
        
        return mx.mean(weighted_losses)
```

### Adaptive Margin
```python
class AdaptiveMarginScheduler:
    """Dynamically adjust triplet margin during training"""
    def __init__(self, initial_margin=0.2, target_margin=0.5):
        self.current_margin = initial_margin
        self.target_margin = target_margin
    
    def step(self, epoch, total_epochs):
        progress = epoch / total_epochs
        self.current_margin = (
            self.initial_margin + 
            (self.target_margin - self.initial_margin) * progress
        )
        return self.current_margin
```

Remember: Good training requires patience and systematic experimentation. Start with proven configurations and adjust based on your specific data and requirements.