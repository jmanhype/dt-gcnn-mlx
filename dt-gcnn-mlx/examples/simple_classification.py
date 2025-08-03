"""Simple classification example using DT-GCNN on synthetic text data."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DTGCNN, create_model
from src.data import create_sample_data, TripletDataset, create_data_loader


def generate_classification_data(num_samples=1000, vocab_size=5000, seq_length=128, num_classes=4):
    """Generate synthetic text classification data."""
    
    data = []
    labels = []
    
    for i in range(num_samples):
        class_id = i % num_classes
        
        # Generate class-specific patterns
        if class_id == 0:
            # Short sequences with high frequency words
            sequence = np.random.randint(1, vocab_size // 4, seq_length // 2).tolist()
            sequence.extend([0] * (seq_length - len(sequence)))  # Pad with zeros
        elif class_id == 1:
            # Medium sequences with medium frequency words  
            sequence = np.random.randint(vocab_size // 4, vocab_size // 2, int(seq_length * 0.7)).tolist()
            sequence.extend([0] * (seq_length - len(sequence)))
        elif class_id == 2:
            # Long sequences with low frequency words
            sequence = np.random.randint(vocab_size // 2, vocab_size, int(seq_length * 0.9)).tolist()
            sequence.extend([0] * (seq_length - len(sequence)))
        else:
            # Mixed sequences
            sequence = np.random.randint(1, vocab_size, seq_length).tolist()
            
        data.append(sequence)
        labels.append(class_id)
        
    return np.array(data), np.array(labels)


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
    
    # Parameters
    num_samples = 1000
    vocab_size = 5000
    seq_length = 128
    num_classes = 4
    batch_size = 32
    num_epochs = 20
    
    # Generate synthetic data
    print("\n1. Generating synthetic text classification data...")
    data, labels = generate_classification_data(
        num_samples=num_samples,
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_classes=num_classes
    )
    
    print(f"   Generated {num_samples} samples")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Number of classes: {num_classes}")
    
    # Convert to MLX arrays
    data = mx.array(data)
    labels = mx.array(labels)
    
    # Split into train/validation sets
    train_size = int(0.8 * num_samples)
    train_data = data[:train_size]
    train_labels = labels[:train_size]
    val_data = data[train_size:]
    val_labels = labels[train_size:]
    
    print(f"\n2. Data split:")
    print(f"   Training samples: {train_size}")
    print(f"   Validation samples: {num_samples - train_size}")
    
    # Create model
    print("\n3. Creating DT-GCNN model...")
    model = create_model(
        preset="small",
        vocab_size=vocab_size,
        num_classes=num_classes
    )
    
    # Count parameters
    total_params = sum(param.size if hasattr(param, 'size') else len(param) for param in model.parameters().values())
    print(f"   Total parameters: {total_params:,}")
    
    # Setup training
    print("\n4. Setting up training...")
    optimizer = optim.Adam(learning_rate=0.001)
    
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
        train_loss = 0
        train_correct = 0
        train_batches = 0
        
        # Mini-batch training
        for i in range(0, len(train_data), batch_size):
            end = min(i + batch_size, len(train_data))
            batch_data = train_data[i:end]
            batch_labels = train_labels[i:end]
            
            # Define loss function for this batch
            def loss_fn(model, x, y):
                logits, _ = model(x, return_embeddings=False)
                return mx.mean(nn.losses.cross_entropy(logits, y))
            
            # Get loss and gradients using nn.value_and_grad
            loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
            loss, grads = loss_and_grad_fn(model, batch_data, batch_labels)
            
            # Update model
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            # Track metrics
            train_loss += loss.item()
            logits, _ = model(batch_data, return_embeddings=False)
            predictions = mx.argmax(logits, axis=1)
            train_correct += mx.sum(predictions == batch_labels).item()
            train_batches += 1
            
        # Validation
        val_loss = 0
        val_correct = 0
        val_batches = 0
        
        for i in range(0, len(val_data), batch_size):
            end = min(i + batch_size, len(val_data))
            batch_data = val_data[i:end]
            batch_labels = val_labels[i:end]
            
            # Forward pass
            logits, _ = model(batch_data, return_embeddings=False)
            loss = mx.mean(nn.losses.cross_entropy(logits, batch_labels))
            
            # Track metrics
            val_loss += loss.item()
            predictions = mx.argmax(logits, axis=1)
            val_correct += mx.sum(predictions == batch_labels).item()
            val_batches += 1
            
        # Calculate epoch metrics
        train_loss_avg = train_loss / train_batches
        train_acc = train_correct / len(train_data)
        val_loss_avg = val_loss / val_batches
        val_acc = val_correct / len(val_data)
        
        # Store history
        history['train_loss'].append(train_loss_avg)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss_avg)
        history['val_accuracy'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
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
    
    # Get a few validation samples
    test_samples = 5
    test_data = val_data[:test_samples]
    test_labels = val_labels[:test_samples]
    
    # Make predictions
    logits, _ = model(test_data, return_embeddings=False)
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
    
    for i in range(0, len(val_data), batch_size):
        end = min(i + batch_size, len(val_data))
        batch_data = val_data[i:end]
        batch_labels = val_labels[i:end]
        
        logits, _ = model(batch_data, return_embeddings=False)
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
    mx.savez(model_path, **model.parameters())
    print(f"Model saved to {model_path}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()