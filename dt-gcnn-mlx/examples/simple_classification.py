"""Simple classification example using DT-GCNN on synthetic data."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.dt_gcnn import DTGCNN
from src.data.preprocessing import normalize_adjacency_matrix
from src.training.trainer import DTGCNNTrainer, TrainingConfig


def generate_synthetic_graph_data(num_samples=1000, num_nodes=20, num_features=8, 
                                  seq_length=30, num_classes=3):
    """Generate synthetic graph time series data for classification."""
    
    # Create a simple graph structure (community graph)
    adj = np.zeros((num_nodes, num_nodes))
    
    # Create communities
    nodes_per_community = num_nodes // num_classes
    for i in range(num_classes):
        start = i * nodes_per_community
        end = (i + 1) * nodes_per_community if i < num_classes - 1 else num_nodes
        
        # Dense connections within community
        for j in range(start, end):
            for k in range(start, end):
                if j != k:
                    adj[j, k] = np.random.rand() * 0.8 + 0.2
                    
    # Sparse connections between communities
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj[i, j] == 0 and i != j:
                if np.random.rand() < 0.1:
                    adj[i, j] = np.random.rand() * 0.3
                    
    # Make symmetric and add self-loops
    adj = (adj + adj.T) / 2
    adj = adj + np.eye(num_nodes)
    
    # Normalize adjacency matrix
    adj = adj / adj.sum(axis=1, keepdims=True)
    
    # Generate features and labels
    features = []
    labels = []
    
    for i in range(num_samples):
        # Class determines the pattern
        class_id = i % num_classes
        
        # Base signal for the class
        t = np.linspace(0, 4 * np.pi, seq_length)
        
        if class_id == 0:
            # Sinusoidal pattern
            base_signal = np.sin(t) + 0.5 * np.sin(3 * t)
        elif class_id == 1:
            # Exponential decay pattern
            base_signal = np.exp(-t / (2 * np.pi)) * np.cos(2 * t)
        else:
            # Square wave pattern
            base_signal = np.sign(np.sin(t)) + 0.3 * np.sign(np.sin(3 * t))
            
        # Create node features based on community structure
        node_features = np.zeros((seq_length, num_nodes, num_features))
        
        for node_idx in range(num_nodes):
            # Which community does this node belong to?
            community = min(node_idx // nodes_per_community, num_classes - 1)
            
            # Add community-specific patterns
            for feat_idx in range(num_features):
                if community == class_id:
                    # Strong signal for nodes in the class community
                    node_features[:, node_idx, feat_idx] = (
                        base_signal * (1 + 0.1 * feat_idx) + 
                        np.random.randn(seq_length) * 0.1
                    )
                else:
                    # Weak/random signal for other nodes
                    node_features[:, node_idx, feat_idx] = (
                        base_signal * 0.1 + 
                        np.random.randn(seq_length) * 0.3
                    )
                    
        features.append(node_features)
        labels.append(class_id)
        
    return np.array(features), np.array(labels), adj


def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_accuracy'], label='Train Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to training_history.png")


def main():
    """Main function to run the example."""
    print("DT-GCNN Simple Classification Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    mx.random.seed(42)
    
    # Generate synthetic data
    print("\n1. Generating synthetic graph time series data...")
    num_samples = 600
    num_nodes = 20
    num_features = 8
    seq_length = 30
    num_classes = 3
    
    features, labels, adj = generate_synthetic_graph_data(
        num_samples=num_samples,
        num_nodes=num_nodes,
        num_features=num_features,
        seq_length=seq_length,
        num_classes=num_classes
    )
    
    print(f"   Generated {num_samples} samples")
    print(f"   Graph: {num_nodes} nodes, {num_features} features per node")
    print(f"   Sequence length: {seq_length}")
    print(f"   Number of classes: {num_classes}")
    
    # Convert to MLX arrays
    features = mx.array(features)
    labels = mx.array(labels)
    adj_matrix = mx.array(adj)
    
    # Split into train/validation sets
    train_size = int(0.8 * num_samples)
    train_features = features[:train_size]
    train_labels = labels[:train_size]
    val_features = features[train_size:]
    val_labels = labels[train_size:]
    
    print(f"\n2. Data split:")
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {num_samples - train_size}")
    
    # Create model
    print("\n3. Creating DT-GCNN model...")
    model = DTGCNN(
        num_nodes=num_nodes,
        input_dim=num_features,
        hidden_dims=[32, 64, 128],
        temporal_kernel_size=3,
        dilations=[1, 2, 4, 8],
        num_classes=num_classes,
        dropout=0.3
    )
    
    # Count parameters
    total_params = sum(p.size for p in model.parameters().values())
    print(f"   Total parameters: {total_params:,}")
    
    # Setup training
    print("\n4. Setting up training...")
    optimizer = optim.Adam(learning_rate=0.001)
    
    # Training configuration
    batch_size = 32
    num_epochs = 50
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    print(f"\n5. Training for {num_epochs} epochs...")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_batches = 0
        
        # Mini-batch training
        for i in range(0, len(train_features), batch_size):
            end = min(i + batch_size, len(train_features))
            batch_features = train_features[i:end]
            batch_labels = train_labels[i:end]
            
            # Forward pass
            def loss_fn(model, features, adj, labels):
                logits = model(features, adj, training=True)
                loss = mx.mean(nn.losses.cross_entropy(logits, labels))
                return loss, logits
            
            # Compute loss and gradients
            (loss, logits), grads = mx.value_and_grad(loss_fn, has_aux=True)(
                model, batch_features, adj_matrix, batch_labels
            )
            
            # Update model
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            # Track metrics
            train_loss += loss.item()
            predictions = mx.argmax(logits, axis=1)
            train_correct += mx.sum(predictions == batch_labels).item()
            train_batches += 1
            
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_batches = 0
        
        for i in range(0, len(val_features), batch_size):
            end = min(i + batch_size, len(val_features))
            batch_features = val_features[i:end]
            batch_labels = val_labels[i:end]
            
            # Forward pass (no dropout in eval mode)
            logits = model(batch_features, adj_matrix, training=False)
            loss = mx.mean(nn.losses.cross_entropy(logits, batch_labels))
            
            # Track metrics
            val_loss += loss.item()
            predictions = mx.argmax(logits, axis=1)
            val_correct += mx.sum(predictions == batch_labels).item()
            val_batches += 1
            
        # Calculate epoch metrics
        train_loss_avg = train_loss / train_batches
        train_acc = train_correct / len(train_features)
        val_loss_avg = val_loss / val_batches
        val_acc = val_correct / len(val_features)
        
        # Store history
        history['train_loss'].append(train_loss_avg)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_accuracy'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")
            
    print("-" * 50)
    print(f"\nTraining completed!")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Test on some examples
    print("\n6. Testing on sample predictions...")
    model.eval()
    
    # Get a few validation samples
    test_samples = 5
    test_features = val_features[:test_samples]
    test_labels = val_labels[:test_samples]
    
    # Make predictions
    logits = model(test_features, adj_matrix, training=False)
    predictions = mx.argmax(logits, axis=1)
    
    print("\nSample predictions:")
    print("-" * 30)
    for i in range(test_samples):
        print(f"Sample {i+1}: True label = {test_labels[i].item()}, "
              f"Predicted = {predictions[i].item()}")
        
    # Confusion matrix
    print("\n7. Computing confusion matrix...")
    all_predictions = []
    all_labels = []
    
    for i in range(0, len(val_features), batch_size):
        end = min(i + batch_size, len(val_features))
        batch_features = val_features[i:end]
        batch_labels = val_labels[i:end]
        
        logits = model(batch_features, adj_matrix, training=False)
        predictions = mx.argmax(logits, axis=1)
        
        all_predictions.extend(predictions.tolist())
        all_labels.extend(batch_labels.tolist())
        
    # Simple confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_predictions):
        confusion[true, pred] += 1
        
    print("\nConfusion Matrix:")
    print("-" * 20)
    print("True\\Pred", end="")
    for i in range(num_classes):
        print(f"\t{i}", end="")
    print()
    for i in range(num_classes):
        print(f"{i}", end="")
        for j in range(num_classes):
            print(f"\t{confusion[i, j]}", end="")
        print()
        
    # Save model
    print("\n8. Saving trained model...")
    model_path = "simple_classification_model.npz"
    mx.save(model_path, model.parameters())
    print(f"Model saved to {model_path}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()