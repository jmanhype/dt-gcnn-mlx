# DT-GCNN Troubleshooting Guide

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Performance Problems](#performance-problems)
4. [Training Issues](#training-issues)
5. [Memory Problems](#memory-problems)
6. [Apple Silicon Specific](#apple-silicon-specific)
7. [Common Error Messages](#common-error-messages)
8. [FAQ](#faq)

## Installation Issues

### MLX Installation Fails

**Problem**: `pip install mlx` fails with compilation errors

**Solution**:
```bash
# Ensure you have latest Xcode Command Line Tools
xcode-select --install

# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install MLX with verbose output
pip install mlx --verbose

# If still failing, try building from source
git clone https://github.com/ml-explore/mlx
cd mlx
pip install -e .
```

### Metal Performance Shaders Not Found

**Problem**: Error about missing Metal Performance Shaders

**Solution**:
```bash
# Check macOS version (needs 13.0+)
sw_vers

# Update macOS if needed
# Then verify Metal support
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

### Dependency Conflicts

**Problem**: Conflicting package versions

**Solution**:
```bash
# Create clean virtual environment
python -m venv dt-gcnn-env
source dt-gcnn-env/bin/activate

# Install with exact versions
pip install -r requirements.txt --no-cache-dir

# Or use conda for better dependency resolution
conda create -n dt-gcnn python=3.9
conda activate dt-gcnn
conda install -c conda-forge mlx numpy scipy
pip install dt-gcnn-mlx
```

## Runtime Errors

### CUDA/GPU Errors on Apple Silicon

**Problem**: `RuntimeError: CUDA is not available`

**Solution**:
```python
# MLX uses Metal, not CUDA. Update device references:

# Wrong
device = torch.device('cuda')

# Correct
import mlx.core as mx
mx.set_default_device(mx.gpu)  # Uses Metal GPU

# Or explicitly specify
model = DTGCNN().to(mx.gpu)
```

### Model Loading Errors

**Problem**: `KeyError` when loading saved model

**Solution**:
```python
# Check model architecture matches saved state
def safe_load_model(model, checkpoint_path):
    checkpoint = mx.load(checkpoint_path)
    
    # Filter out mismatched keys
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    # Update model state
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Report missing keys
    missing = set(model_dict.keys()) - set(pretrained_dict.keys())
    if missing:
        print(f"Missing keys: {missing}")
    
    return model
```

### Tokenizer Errors

**Problem**: `ValueError: Token indices sequence length is longer than the specified maximum`

**Solution**:
```python
# Set appropriate max length
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Check actual lengths in your data
def check_sequence_lengths(texts):
    lengths = [len(tokenizer.encode(text)) for text in texts]
    print(f"Max length: {max(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95)}")
    return lengths

# Use dynamic truncation
def safe_tokenize(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='np'  # Convert to MLX later
    )
```

## Performance Problems

### Slow Training Speed

**Problem**: Training is slower than expected

**Solutions**:

1. **Enable Mixed Precision**:
```python
# Use float16 for faster computation
model = DTGCNN(dtype=mx.float16)

# Or use automatic mixed precision
with mx.amp.autocast():
    outputs = model(inputs)
```

2. **Optimize Batch Size**:
```python
# Find optimal batch size
def find_optimal_batch_size(model, starting_size=32):
    batch_size = starting_size
    
    while True:
        try:
            # Test forward pass
            dummy_input = {
                'input_ids': mx.random.randint(0, 1000, (batch_size, 128)),
                'attention_mask': mx.ones((batch_size, 128))
            }
            
            with mx.no_grad():
                _ = model(**dummy_input)
            
            print(f"Batch size {batch_size} works")
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimal = batch_size // 2
                print(f"Optimal batch size: {optimal}")
                return optimal
            else:
                raise
```

3. **Profile Performance**:
```python
import time
import mlx.core as mx

class PerformanceProfiler:
    def __init__(self):
        self.timings = {}
    
    def profile_model(self, model, dataloader, num_batches=10):
        mx.eval(model)  # Ensure model is compiled
        
        # Warmup
        for i, batch in enumerate(dataloader):
            if i >= 3:
                break
            _ = model(batch)
        
        # Profile
        times = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            start = time.time()
            _ = model(batch)
            mx.eval()  # Ensure computation completes
            end = time.time()
            
            times.append(end - start)
        
        avg_time = np.mean(times[1:])  # Skip first
        throughput = batch_size / avg_time
        
        print(f"Average batch time: {avg_time:.3f}s")
        print(f"Throughput: {throughput:.1f} samples/sec")
        
        return {
            'avg_batch_time': avg_time,
            'throughput': throughput
        }
```

### High Memory Usage

**Problem**: Running out of memory during training

**Solutions**:

1. **Gradient Accumulation**:
```python
def train_with_accumulation(model, dataloader, accumulation_steps=4):
    optimizer = mx.optim.AdamW(model.parameters())
    
    for i, batch in enumerate(dataloader):
        # Forward pass
        loss = model(batch) / accumulation_steps
        loss.backward()
        
        # Update weights every N steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

2. **Memory-Efficient Graph Construction**:
```python
class MemoryEfficientGraphConstructor:
    def __init__(self, chunk_size=1000):
        self.chunk_size = chunk_size
    
    def construct_graph_chunked(self, embeddings):
        n = embeddings.shape[0]
        adjacency = mx.zeros((n, n))
        
        # Process in chunks to reduce memory
        for i in range(0, n, self.chunk_size):
            end_i = min(i + self.chunk_size, n)
            chunk_i = embeddings[i:end_i]
            
            for j in range(0, n, self.chunk_size):
                end_j = min(j + self.chunk_size, n)
                chunk_j = embeddings[j:end_j]
                
                # Compute similarities for chunk
                sim_chunk = mx.matmul(chunk_i, chunk_j.T)
                adjacency[i:end_i, j:end_j] = sim_chunk
        
        return adjacency
```

## Training Issues

### Loss Not Decreasing

**Problem**: Training loss plateaus or doesn't decrease

**Diagnosis and Solutions**:

```python
class TrainingDiagnostics:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
    
    def diagnose(self):
        """Run comprehensive diagnostics"""
        
        # 1. Check gradient flow
        self.check_gradients()
        
        # 2. Verify loss computation
        self.verify_loss()
        
        # 3. Check data quality
        self.check_data_quality()
        
        # 4. Test different learning rates
        self.lr_range_test()
    
    def check_gradients(self):
        """Check if gradients are flowing properly"""
        batch = next(iter(self.dataloader))
        loss = self.model(batch)
        loss.backward()
        
        grad_norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = mx.linalg.norm(param.grad).item()
                grad_norms[name] = grad_norm
                
                if grad_norm == 0:
                    print(f"WARNING: Zero gradient in {name}")
                elif grad_norm > 10:
                    print(f"WARNING: Large gradient in {name}: {grad_norm}")
        
        return grad_norms
    
    def verify_loss(self):
        """Verify loss is computed correctly"""
        batch = next(iter(self.dataloader))
        
        # Manual triplet loss computation
        anchor = self.model.encode(batch['anchor'])
        positive = self.model.encode(batch['positive'])
        negative = self.model.encode(batch['negative'])
        
        d_ap = mx.linalg.norm(anchor - positive, axis=1)
        d_an = mx.linalg.norm(anchor - negative, axis=1)
        
        manual_loss = mx.mean(mx.maximum(0, d_ap - d_an + 0.2))
        model_loss = self.model.compute_loss(batch)
        
        print(f"Manual loss: {manual_loss.item():.4f}")
        print(f"Model loss: {model_loss.item():.4f}")
        
        if abs(manual_loss - model_loss) > 0.01:
            print("WARNING: Loss computation mismatch!")
```

### Overfitting

**Problem**: Validation loss increases while training loss decreases

**Solutions**:

```python
# 1. Add regularization
class RegularizedDTGCNN(DTGCNN):
    def __init__(self, config, l2_reg=0.01):
        super().__init__(config)
        self.l2_reg = l2_reg
    
    def compute_loss(self, batch):
        # Standard loss
        loss = super().compute_loss(batch)
        
        # Add L2 regularization
        l2_loss = 0
        for param in self.parameters():
            l2_loss += mx.sum(param ** 2)
        
        return loss + self.l2_reg * l2_loss

# 2. Use dropout strategically
def add_dropout_layers(model, dropout_rate=0.5):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Add dropout after linear layers
            setattr(model, f"{name}_dropout", nn.Dropout(dropout_rate))

# 3. Data augmentation for text
class TextAugmenter:
    def __init__(self, aug_prob=0.3):
        self.aug_prob = aug_prob
    
    def augment(self, text):
        if mx.random.uniform() > self.aug_prob:
            return text
        
        # Random augmentation strategies
        strategies = [
            self.synonym_replacement,
            self.random_deletion,
            self.random_swap,
            self.back_translation
        ]
        
        aug_fn = mx.random.choice(strategies)
        return aug_fn(text)
```

## Memory Problems

### Memory Leak During Training

**Problem**: Memory usage grows continuously

**Solution**:
```python
import gc
import mlx.core as mx

class MemoryMonitor:
    def __init__(self, log_every=10):
        self.log_every = log_every
        self.step = 0
    
    def on_batch_end(self):
        self.step += 1
        
        if self.step % self.log_every == 0:
            # Force garbage collection
            gc.collect()
            
            # Clear MLX cache
            mx.clear_cache()
            
            # Log memory usage
            memory_info = mx.metal.get_memory_info()
            print(f"Step {self.step}: Memory used: {memory_info['used'] / 1e9:.2f} GB")

# Use in training loop
monitor = MemoryMonitor()

for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Monitor memory
        monitor.on_batch_end()
        
        # Explicitly delete large tensors
        del loss
```

### Batch Size Memory Optimization

**Problem**: Can't fit desired batch size in memory

**Solution**:
```python
class DynamicBatchSizer:
    def __init__(self, initial_batch_size=512, min_batch_size=32):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
    
    def get_dataloader(self, dataset):
        while self.batch_size >= self.min_batch_size:
            try:
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )
                
                # Test one batch
                batch = next(iter(dataloader))
                _ = model(batch)
                
                print(f"Using batch size: {self.batch_size}")
                return dataloader
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.batch_size = self.batch_size // 2
                    mx.clear_cache()
                else:
                    raise
        
        raise ValueError(f"Cannot fit batch size >= {self.min_batch_size}")
```

## Apple Silicon Specific

### Metal Performance Issues

**Problem**: Poor performance on M1/M2/M3

**Solutions**:

1. **Verify Metal is being used**:
```python
import mlx.core as mx

# Check Metal availability
print(f"Metal available: {mx.metal.is_available()}")

# Set device explicitly
mx.set_default_device(mx.gpu)

# Verify tensors are on GPU
tensor = mx.array([1, 2, 3])
print(f"Tensor device: {tensor.device}")
```

2. **Optimize for Apple Silicon**:
```python
# Use optimal data types
# float16 is faster on Apple Silicon
model = model.to(dtype=mx.float16)

# Use fused operations
class OptimizedGCNLayer(nn.Module):
    def forward(self, x, adj):
        # Fused multiply-add is faster
        return mx.addmm(self.bias, adj, mx.matmul(x, self.weight))
```

3. **Thread optimization**:
```python
import os

# Optimize thread count for your chip
# M1: 8 cores, M1 Pro/Max: 10 cores, M2: 8-12 cores
num_cores = os.cpu_count()
optimal_threads = min(num_cores - 2, 8)  # Leave some for system
mx.core.set_num_threads(optimal_threads)
```

### Unified Memory Issues

**Problem**: Confusion about memory allocation on Apple Silicon

**Solution**:
```python
# Apple Silicon has unified memory - CPU and GPU share same memory
# This means no explicit memory transfers needed

# MLX handles this automatically
data = mx.array(numpy_array)  # Automatically accessible by GPU

# But be mindful of memory limits
import subprocess

def get_system_memory():
    """Get total system memory on macOS"""
    cmd = ['sysctl', 'hw.memsize']
    result = subprocess.run(cmd, capture_output=True, text=True)
    memsize = int(result.stdout.split()[-1])
    return memsize / (1024**3)  # Convert to GB

total_memory = get_system_memory()
safe_memory_limit = total_memory * 0.7  # Use 70% max
```

## Common Error Messages

### "RuntimeError: MPS backend out of memory"

**Solution**:
```python
# 1. Reduce batch size
batch_size = batch_size // 2

# 2. Enable gradient checkpointing
@mx.checkpoint
def checkpointed_forward(model, x):
    return model(x)

# 3. Clear cache between epochs
for epoch in range(num_epochs):
    train_epoch()
    mx.clear_cache()
```

### "ValueError: shapes not aligned"

**Solution**:
```python
# Debug shape mismatches
def debug_shapes(model, batch):
    print("Input shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
    
    # Trace through model
    x = batch['input_ids']
    
    with mx.no_grad():
        # Check each layer
        for name, layer in model.named_children():
            if hasattr(layer, 'forward'):
                x = layer(x)
                print(f"{name} output: {x.shape}")
```

### "ImportError: No module named 'mlx'"

**Solution**:
```bash
# Ensure you're in the right environment
which python

# Reinstall MLX
pip uninstall mlx
pip install mlx --no-cache-dir

# Verify installation
python -c "import mlx; print(mlx.__version__)"
```

## FAQ

### Q: How do I convert PyTorch models to MLX?

**A**: Use the conversion utilities:
```python
from dt_gcnn_mlx.utils import convert_pytorch_to_mlx

# Load PyTorch model
pytorch_model = torch.load('model.pt')

# Convert to MLX
mlx_model = convert_pytorch_to_mlx(
    pytorch_model,
    sample_input=sample_batch
)

# Save MLX model
mx.save('model.mlx', mlx_model.state_dict())
```

### Q: Can I use multiple GPUs on Mac Studio?

**A**: MLX currently supports single GPU. For multi-GPU, use data parallelism:
```python
# Split data across multiple processes
from multiprocessing import Pool

def train_on_subset(data_subset):
    model = DTGCNN()
    # Train on subset
    return model.state_dict()

# Parallel training
with Pool(processes=2) as pool:
    model_states = pool.map(train_on_subset, data_splits)

# Average model weights
averaged_state = average_state_dicts(model_states)
```

### Q: How do I debug slow training?

**A**: Use the built-in profiler:
```python
from dt_gcnn_mlx.utils import profile_training

# Profile one epoch
profile_report = profile_training(
    model, train_loader,
    num_batches=100
)

print(profile_report)
# Shows time spent in each component
```

### Q: Memory usage keeps growing, even with small batches?

**A**: Check for common memory leak sources:
```python
# 1. Storing computation graphs
losses = []  # Don't do this with tensors!

# Instead, convert to Python float
losses = []
for batch in dataloader:
    loss = model(batch)
    losses.append(loss.item())  # Convert to float
    
# 2. Not clearing gradients
optimizer.zero_grad()  # Always call after optimizer.step()

# 3. Keeping references to old tensors
def train_step(batch):
    loss = model(batch)
    # Don't return loss tensor, return float
    return loss.item()
```

### Q: How do I know if my model is using the Neural Engine?

**A**: MLX automatically uses the Neural Engine when beneficial:
```python
# Check if operation will use Neural Engine
import mlx.core as mx

# Certain ops automatically use ANE
# - Convolutions with specific sizes
# - Matrix multiplications meeting criteria
# - Optimized attention operations

# Monitor performance
with mx.profiler.profile() as p:
    output = model(input)

print(p.report())  # Shows which accelerator was used
```

## Getting Help

If you're still experiencing issues:

1. **Check the logs**: Enable verbose logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Minimal reproducible example**: Create a simple script that reproduces the issue

3. **System information**: Include this when reporting issues:
```python
from dt_gcnn_mlx.utils import get_system_info
print(get_system_info())
```

4. **Community resources**:
   - GitHub Issues: https://github.com/yourusername/dt-gcnn/issues
   - MLX Discord: https://discord.gg/mlx
   - Stack Overflow: Tag with `mlx` and `dt-gcnn`

Remember: Most issues have been encountered before. Check existing issues and discussions first!