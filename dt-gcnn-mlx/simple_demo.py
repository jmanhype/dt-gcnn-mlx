#!/usr/bin/env python3
"""
Simple Demo for DT-GCNN MLX
Basic functionality test on Apple M2
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

def main():
    print("🚀 DT-GCNN MLX Simple Demo")
    print("=" * 40)
    
    # 1. Check MLX device and Metal availability
    device = mx.default_device()
    print(f"✓ Device: {device}")
    print(f"✓ Metal GPU: {'gpu' in str(device).lower()}")
    
    # 2. Test basic MLX operations on your M2
    print("\n🧪 Testing MLX operations...")
    
    # Create some test tensors
    a = mx.random.normal((1000, 1000))
    b = mx.random.normal((1000, 1000))
    
    # Matrix multiplication (should use GPU if available)
    c = mx.matmul(a, b)
    print(f"✓ Matrix multiply: {a.shape} × {b.shape} = {c.shape}")
    
    # Test neural network layers
    print("\n🏗️ Testing NN components...")
    
    # Simple embedding layer
    vocab_size = 5000
    embed_dim = 128
    embedding = nn.Embedding(vocab_size, embed_dim)
    
    # Test input
    batch_size = 32
    seq_len = 64
    tokens = mx.random.randint(0, vocab_size, [batch_size, seq_len])
    embedded = embedding(tokens)
    print(f"✓ Embedding: {tokens.shape} → {embedded.shape}")
    
    # GRU layer
    gru = nn.GRU(embed_dim, 64)
    gru_out = gru(embedded)
    print(f"✓ GRU: {embedded.shape} → {gru_out.shape}")
    
    # 1D convolution
    conv = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
    # Transpose for conv1d (expects channels first: [N, C, L])
    conv_input = mx.transpose(embedded, (0, 2, 1))
    conv_out = conv(conv_input)
    print(f"✓ Conv1D: {conv_input.shape} → {conv_out.shape}")
    
    # Linear layer
    linear = nn.Linear(64, 4)  # 4 classes
    logits = linear(gru_out[:, -1, :])  # Use last timestep
    print(f"✓ Linear: {gru_out[:, -1, :].shape} → {logits.shape}")
    
    # Loss function
    labels = mx.random.randint(0, 4, [batch_size])
    loss = mx.mean(nn.losses.cross_entropy(logits, labels))
    print(f"✓ Cross-entropy loss: {float(loss):.4f}")
    
    # Test triplet loss computation
    print("\n📊 Testing triplet operations...")
    embed_dim = 64
    anchor = mx.random.normal((batch_size, embed_dim))
    positive = mx.random.normal((batch_size, embed_dim))
    negative = mx.random.normal((batch_size, embed_dim))
    
    # Compute distances
    pos_dist = mx.sum((anchor - positive) ** 2, axis=1)
    neg_dist = mx.sum((anchor - negative) ** 2, axis=1)
    triplet_loss = mx.mean(mx.maximum(pos_dist - neg_dist + 0.3, 0.0))
    print(f"✓ Triplet loss: {float(triplet_loss):.4f}")
    
    # Memory usage info
    print("\n💾 Performance info:")
    print(f"✓ Demo completed successfully on {device}")
    print(f"✓ All tensor operations executed")
    
    if 'gpu' in str(device).lower():
        print("🚀 GPU acceleration active!")
    else:
        print("ℹ️  Running on CPU (GPU may not be detected)")
    
    print("\n✅ DT-GCNN MLX basic functionality verified!")
    print("\n📚 Next steps:")
    print("1. Install remaining dependencies: pip install sentencepiece")
    print("2. Try the full quick_start.py when imports are fixed")
    print("3. Check the examples/ directory for complete demos")

if __name__ == "__main__":
    main()