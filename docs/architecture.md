# DT-GCNN Architecture Guide

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Dynamic Graph Construction](#dynamic-graph-construction)
4. [GCN Layers](#gcn-layers)
5. [Triplet Loss and Mining](#triplet-loss-and-mining)
6. [MLX Optimizations](#mlx-optimizations)
7. [Implementation Details](#implementation-details)

## Overview

The Dynamic Triplet Graph Convolutional Neural Network (DT-GCNN) is a sophisticated architecture designed to learn discriminative word embeddings by combining:

- **Dynamic graph construction** based on semantic similarity
- **Graph Convolutional Networks** for multi-hop feature aggregation
- **Triplet loss** with hard negative mining
- **MLX optimizations** for Apple Silicon acceleration

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Triplets                         │
│              (anchor, positive, negative)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────┐
│                  Embedding Layer                            │
│            BERT/RoBERTa (768-dim vectors)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────┐
│              Dynamic Graph Constructor                       │
│         k-NN + Semantic Threshold (MLX-optimized)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────┐
│                    GCN Stack                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  GCN Layer 1 │→ │  GCN Layer 2 │→ │  GCN Layer 3 │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────┐
│                  Projection Head                            │
│              MLP: 256 → 128 → output_dim                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────┐
│              Triplet Loss with Hard Mining                  │
│                  L = max(0, d_ap - d_an + margin)          │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Embedding Layer

The embedding layer transforms raw text into dense vector representations:

```python
class EmbeddingLayer(nn.Module):
    def __init__(self, model_name='bert-base-uncased', freeze=False):
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze:
            self.encoder.freeze()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        # Use [CLS] token or mean pooling
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
```

**Key Features:**
- Supports BERT, RoBERTa, and other transformer models
- Optional freezing for transfer learning
- MLX-optimized attention mechanisms

### 2. Dynamic Graph Constructor

The graph is constructed dynamically based on semantic similarity:

```python
class DynamicGraphConstructor:
    def __init__(self, k_neighbors=10, threshold=0.7):
        self.k = k_neighbors
        self.threshold = threshold
    
    def construct_graph(self, embeddings):
        # Compute similarity matrix
        similarity = mx.matmul(embeddings, embeddings.T)
        
        # Apply k-NN and threshold
        adjacency = self._apply_knn_threshold(similarity)
        
        # Normalize for GCN
        adjacency = self._normalize_adjacency(adjacency)
        
        return adjacency
```

**Graph Construction Process:**
1. Compute pairwise similarities using cosine similarity
2. Apply k-nearest neighbors selection
3. Filter by similarity threshold
4. Add self-loops and normalize

## GCN Layers

### Graph Convolutional Layer

Each GCN layer performs spectral graph convolution:

```python
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        self.weight = mx.random.normal(
            (in_features, out_features), 
            scale=math.sqrt(2.0 / in_features)
        )
        self.bias = mx.zeros((out_features,))
        self.activation = get_activation(activation)
    
    def forward(self, features, adjacency):
        # Graph convolution: A @ X @ W + b
        support = mx.matmul(features, self.weight)
        output = mx.matmul(adjacency, support) + self.bias
        return self.activation(output)
```

### Multi-Layer GCN Stack

The model stacks multiple GCN layers with residual connections:

```python
class GCNStack(nn.Module):
    def __init__(self, layer_dims, dropout=0.1):
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                GCNLayer(layer_dims[i], layer_dims[i+1])
            )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, adjacency):
        h = features
        for i, layer in enumerate(self.layers):
            h_new = layer(h, adjacency)
            
            # Residual connection (if dimensions match)
            if h.shape[-1] == h_new.shape[-1]:
                h_new = h_new + h
            
            h = self.dropout(h_new)
        
        return h
```

## Triplet Loss and Mining

### Triplet Loss Implementation

The triplet loss encourages the model to place semantically similar items closer:

```python
class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, distance='euclidean'):
        self.margin = margin
        self.distance_fn = get_distance_function(distance)
    
    def forward(self, anchor, positive, negative):
        d_ap = self.distance_fn(anchor, positive)
        d_an = self.distance_fn(anchor, negative)
        
        loss = mx.maximum(0.0, d_ap - d_an + self.margin)
        return mx.mean(loss)
```

### Hard Negative Mining

To improve training efficiency, we implement hard negative mining:

```python
class HardNegativeMiner:
    def __init__(self, mining_strategy='batch_hard'):
        self.strategy = mining_strategy
    
    def mine_triplets(self, embeddings, labels):
        if self.strategy == 'batch_hard':
            return self._batch_hard_mining(embeddings, labels)
        elif self.strategy == 'batch_all':
            return self._batch_all_mining(embeddings, labels)
    
    def _batch_hard_mining(self, embeddings, labels):
        # For each anchor, find hardest positive and negative
        dist_matrix = compute_pairwise_distances(embeddings)
        
        triplets = []
        for i in range(len(embeddings)):
            # Find hardest positive (same class, farthest)
            pos_mask = labels == labels[i]
            pos_mask[i] = False
            hardest_pos = mx.argmax(dist_matrix[i] * pos_mask)
            
            # Find hardest negative (different class, closest)
            neg_mask = labels != labels[i]
            hardest_neg = mx.argmin(
                dist_matrix[i] + (1 - neg_mask) * 1e9
            )
            
            triplets.append((i, hardest_pos, hardest_neg))
        
        return triplets
```

## MLX Optimizations

### 1. Metal Performance Shaders

MLX automatically leverages Metal Performance Shaders for:
- Matrix multiplication
- Convolution operations
- Activation functions
- Reduction operations

### 2. Memory Efficiency

```python
# Unified memory architecture
def optimize_memory_layout(tensor):
    # MLX handles memory layout automatically
    # Ensures data locality for GPU operations
    return mx.array(tensor, dtype=mx.float32)

# Gradient checkpointing for large models
@mx.checkpoint
def forward_with_checkpoint(model, inputs):
    return model(inputs)
```

### 3. Mixed Precision Training

```python
class MixedPrecisionTrainer:
    def __init__(self, model, use_amp=True):
        self.model = model
        self.use_amp = use_amp
        self.grad_scaler = GradScaler() if use_amp else None
    
    def training_step(self, batch):
        with mx.autocast(enabled=self.use_amp):
            outputs = self.model(batch)
            loss = self.compute_loss(outputs)
        
        if self.grad_scaler:
            scaled_loss = self.grad_scaler.scale(loss)
            mx.grad(scaled_loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            mx.grad(loss).backward()
            self.optimizer.step()
```

### 4. Distributed Training

For Mac Studio and M2 Ultra setups:

```python
class DistributedDTGCNN:
    def __init__(self, model, devices=['gpu:0', 'gpu:1']):
        self.devices = devices
        self.model_replicas = [
            model.to(device) for device in devices
        ]
    
    def parallel_forward(self, batch):
        # Split batch across devices
        batch_per_device = len(batch) // len(self.devices)
        
        futures = []
        for i, (device, model) in enumerate(
            zip(self.devices, self.model_replicas)
        ):
            start = i * batch_per_device
            end = start + batch_per_device
            device_batch = batch[start:end].to(device)
            
            futures.append(
                mx.async_eval(model(device_batch))
            )
        
        # Gather results
        outputs = mx.concatenate([f.result() for f in futures])
        return outputs
```

## Implementation Details

### 1. Initialization

Proper initialization is crucial for GCN training:

```python
def xavier_init(shape, gain=1.0):
    fan_in, fan_out = shape[0], shape[1]
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return mx.random.normal(shape, scale=std)

def kaiming_init(shape, mode='fan_in'):
    fan = shape[0] if mode == 'fan_in' else shape[1]
    std = math.sqrt(2.0 / fan)
    return mx.random.normal(shape, scale=std)
```

### 2. Normalization Techniques

Layer normalization for stable training:

```python
class GraphLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = mx.ones(normalized_shape)
        self.beta = mx.zeros(normalized_shape)
        self.eps = eps
    
    def forward(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / mx.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

### 3. Attention Mechanisms

Optional attention for dynamic graph weighting:

```python
class GraphAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, features, adjacency):
        B, N, D = features.shape
        
        # Multi-head attention
        Q = self.W_q(features).reshape(B, N, self.num_heads, self.head_dim)
        K = self.W_k(features).reshape(B, N, self.num_heads, self.head_dim)
        V = self.W_v(features).reshape(B, N, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # Apply adjacency mask
        scores = scores * adjacency.unsqueeze(1)
        
        # Softmax and apply to values
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, V)
        
        return out.reshape(B, N, D)
```

### 4. Training Loop

Complete training implementation:

```python
class DTGCNNTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = mx.optimizers.AdamW(
            learning_rate=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )
    
    def train_epoch(self, dataloader):
        epoch_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            # Forward pass
            anchor_emb = self.model(batch['anchor'])
            positive_emb = self.model(batch['positive'])
            negative_emb = self.model(batch['negative'])
            
            # Compute loss
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            mx.clip_grad_norm(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)
```

## Best Practices

1. **Graph Sparsity**: Keep adjacency matrix sparse (10-20 neighbors)
2. **Batch Size**: Use large batches (512+) for better negative sampling
3. **Learning Rate**: Start with 1e-4 and use warmup
4. **Regularization**: Apply dropout (0.1-0.3) between GCN layers
5. **Monitoring**: Track triplet accuracy and embedding quality metrics

## Advanced Topics

### Custom Distance Functions

```python
def hyperbolic_distance(x, y, c=1.0):
    """Compute distance in hyperbolic space."""
    sqrt_c = math.sqrt(c)
    norm_x = mx.linalg.norm(x, axis=-1, keepdims=True)
    norm_y = mx.linalg.norm(y, axis=-1, keepdims=True)
    
    # Compute Poincaré distance
    diff = x - y
    norm_diff = mx.linalg.norm(diff, axis=-1)
    
    num = 2 * c * norm_diff ** 2
    denom = (1 - c * norm_x ** 2) * (1 - c * norm_y ** 2)
    
    return mx.arcsinh(sqrt_c * mx.sqrt(num / denom)) / sqrt_c
```

### Graph Pooling Strategies

```python
class GraphPooling(nn.Module):
    def __init__(self, pooling_type='mean'):
        self.pooling_type = pooling_type
    
    def forward(self, node_features, graph_indicator=None):
        if self.pooling_type == 'mean':
            return mx.mean(node_features, axis=1)
        elif self.pooling_type == 'max':
            return mx.max(node_features, axis=1)
        elif self.pooling_type == 'attention':
            attn_weights = self.attention_layer(node_features)
            return mx.sum(node_features * attn_weights, axis=1)
```

## References

1. Original DT-GCNN paper
2. MLX documentation: https://github.com/ml-explore/mlx
3. GCN fundamentals: Kipf & Welling (2017)
4. Triplet loss: Schroff et al. (2015)