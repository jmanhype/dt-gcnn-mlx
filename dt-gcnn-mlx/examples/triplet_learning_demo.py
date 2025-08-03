"""Triplet learning demonstration using DT-GCNN for text embeddings."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import DTGCNN, create_model
from src.losses.triplet_loss import TripletLoss
from src.losses.mining_strategies import BatchHardMiner
from src.data import create_sample_data, TripletDataset, create_data_loader


def generate_text_patterns(num_samples_per_class=50, vocab_size=5000, seq_length=128, num_classes=4):
    """Generate distinct text patterns for each class."""
    
    data = []
    labels = []
    
    for class_id in range(num_classes):
        for _ in range(num_samples_per_class):
            if class_id == 0:
                # Short sequences with high frequency words (1-1000)
                seq_len = np.random.randint(seq_length//4, seq_length//2)
                sequence = np.random.randint(1, vocab_size//5, seq_len).tolist()
                sequence.extend([0] * (seq_length - len(sequence)))
                
            elif class_id == 1:
                # Medium sequences with medium frequency words (1000-2500)
                seq_len = np.random.randint(seq_length//2, 3*seq_length//4)
                sequence = np.random.randint(vocab_size//5, vocab_size//2, seq_len).tolist()
                sequence.extend([0] * (seq_length - len(sequence)))
                
            elif class_id == 2:
                # Long sequences with mixed frequency words
                seq_len = np.random.randint(3*seq_length//4, seq_length)
                sequence = np.random.randint(vocab_size//2, vocab_size, seq_len).tolist()
                sequence.extend([0] * (seq_length - len(sequence)))
                
            else:
                # Random patterns with repeated tokens
                base_tokens = np.random.randint(1, vocab_size//10, 10)
                sequence = []
                for _ in range(seq_length):
                    if np.random.random() < 0.7:
                        sequence.append(np.random.choice(base_tokens))
                    else:
                        sequence.append(np.random.randint(1, vocab_size))
                        
            data.append(sequence)
            labels.append(class_id)
            
    return np.array(data), np.array(labels)


def visualize_embeddings(embeddings, labels, title="Embeddings Visualization"):
    """Visualize embeddings using t-SNE."""
    # Convert to numpy if needed
    if hasattr(embeddings, 'tolist'):
        embeddings = np.array(embeddings.tolist())
    if hasattr(labels, 'tolist'):
        labels = np.array(labels.tolist())
        
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    class_names = ['Short-High', 'Medium-Med', 'Long-Low', 'Repeated']
    
    for class_id in np.unique(labels):
        mask = labels == class_id
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=colors[class_id], label=class_names[class_id],
                   alpha=0.6, s=50)
        
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    print(f"Saved visualization to {title.lower().replace(' ', '_')}.png")


def compute_embedding_statistics(embeddings, labels):
    """Compute statistics about embedding quality."""
    if hasattr(embeddings, 'tolist'):
        embeddings_np = np.array(embeddings.tolist())
    else:
        embeddings_np = np.array(embeddings)
        
    if hasattr(labels, 'tolist'):
        labels_np = np.array(labels.tolist())
    else:
        labels_np = np.array(labels)
    
    # Intra-class distances
    intra_distances = []
    inter_distances = []
    
    unique_labels = np.unique(labels_np)
    
    for label in unique_labels:
        class_embeddings = embeddings_np[labels_np == label]
        
        # Intra-class distances
        for i in range(len(class_embeddings)):
            for j in range(i + 1, len(class_embeddings)):
                dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                intra_distances.append(dist)
                
    # Inter-class distances
    for label1 in unique_labels:
        for label2 in unique_labels:
            if label1 < label2:
                class1_embeddings = embeddings_np[labels_np == label1]
                class2_embeddings = embeddings_np[labels_np == label2]
                
                for emb1 in class1_embeddings[:10]:  # Sample for efficiency
                    for emb2 in class2_embeddings[:10]:
                        dist = np.linalg.norm(emb1 - emb2)
                        inter_distances.append(dist)
                        
    if intra_distances and inter_distances:
        print("\nEmbedding Statistics:")
        print(f"Average intra-class distance: {np.mean(intra_distances):.4f} (±{np.std(intra_distances):.4f})")
        print(f"Average inter-class distance: {np.mean(inter_distances):.4f} (±{np.std(inter_distances):.4f})")
        print(f"Separation ratio: {np.mean(inter_distances) / np.mean(intra_distances):.4f}")


def main():
    """Main function to run triplet learning demo."""
    print("DT-GCNN Triplet Learning Demo")
    print("=" * 50)
    
    # Set random seed
    np.random.seed(42)
    mx.random.seed(42)
    
    # Generate data
    print("\n1. Generating text pattern data...")
    num_samples_per_class = 50
    num_classes = 4
    vocab_size = 5000
    seq_length = 128
    embedding_dim = 64
    
    data, labels = generate_text_patterns(
        num_samples_per_class=num_samples_per_class,
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_classes=num_classes
    )
    
    print(f"   Total samples: {len(data)}")
    print(f"   Classes: {num_classes}")
    print(f"   Samples per class: {num_samples_per_class}")
    
    # Convert to MLX
    data = mx.array(data)
    labels = mx.array(labels)
    
    # Split data
    train_size = int(0.8 * len(data))
    indices = mx.array(np.random.permutation(len(data)))
    
    train_data = data[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    val_data = data[indices[train_size:]]
    val_labels = labels[indices[train_size:]]
    
    print(f"\n2. Data split:")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    # Create model
    print("\n3. Creating DT-GCNN model for embeddings...")
    model = create_model(
        preset="small",
        vocab_size=vocab_size,
        num_classes=num_classes
    )
    
    # Setup training
    print("\n4. Setting up triplet loss training...")
    triplet_loss = TripletLoss(margin=0.3)
    miner = BatchHardMiner()
    optimizer = optim.Adam(learning_rate=0.001)
    
    # Get initial embeddings
    print("\n5. Computing initial embeddings...")
    initial_embeddings = model.get_embeddings(val_data)
    visualize_embeddings(initial_embeddings, val_labels, "Initial Embeddings (Untrained)")
    compute_embedding_statistics(initial_embeddings, val_labels)
    
    # Training
    batch_size = 32
    num_epochs = 50
    
    print(f"\n6. Training with triplet loss for {num_epochs} epochs...")
    print("-" * 50)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for start_idx in range(0, len(train_data) - batch_size + 1, batch_size):
            batch_data = train_data[start_idx:start_idx + batch_size]
            batch_labels = train_labels[start_idx:start_idx + batch_size]
            
            # Forward pass
            def loss_fn(params):
                embeddings = model.get_embeddings(batch_data)
                
                # Mine triplets
                triplets, stats = miner.mine(embeddings, batch_labels)
                
                if len(triplets) == 0:
                    return mx.array(0.0)
                
                # Extract triplet indices
                anchor_idx = mx.array([t[0] for t in triplets])
                positive_idx = mx.array([t[1] for t in triplets])
                negative_idx = mx.array([t[2] for t in triplets])
                
                # Get triplet embeddings
                anchor_emb = embeddings[anchor_idx]
                positive_emb = embeddings[positive_idx]
                negative_emb = embeddings[negative_idx]
                
                # Compute triplet loss
                loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
                return loss
                
            # Compute loss and gradients
            loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
            
            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            epoch_loss += loss.item()
            num_batches += 1
            
        # Validation
        val_embeddings = model.get_embeddings(val_data)
        val_triplets, _ = miner.mine(val_embeddings, val_labels)
        
        if len(val_triplets) > 0:
            anchor_idx = mx.array([t[0] for t in val_triplets])
            positive_idx = mx.array([t[1] for t in val_triplets])
            negative_idx = mx.array([t[2] for t in val_triplets])
            
            anchor_emb = val_embeddings[anchor_idx]
            positive_emb = val_embeddings[positive_idx]
            negative_emb = val_embeddings[negative_idx]
            
            val_loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
        else:
            val_loss = mx.array(0.0)
        
        # Record losses
        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
            
    print("-" * 50)
    
    # Get final embeddings
    print("\n7. Computing final embeddings...")
    final_embeddings = model.get_embeddings(val_data)
    visualize_embeddings(final_embeddings, val_labels, "Final Embeddings (Trained)")
    compute_embedding_statistics(final_embeddings, val_labels)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Triplet Loss')
    plt.title('Triplet Loss Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('triplet_loss_history.png')
    print("\nSaved training history to triplet_loss_history.png")
    
    # Demonstrate retrieval
    print("\n8. Demonstrating similarity search...")
    
    # Get all embeddings
    all_embeddings = model.get_embeddings(data)
    
    # Pick a query sample
    query_idx = 0
    query_embedding = all_embeddings[query_idx:query_idx+1]
    query_label = labels[query_idx].item()
    
    # Compute distances to all samples
    distances = mx.sqrt(mx.sum((all_embeddings - query_embedding) ** 2, axis=1))
    
    # Get top-5 nearest neighbors (excluding self)
    sorted_indices = mx.argsort(distances)
    
    print(f"\nQuery sample: Index {query_idx}, Class {query_label}")
    print("Top-5 nearest neighbors:")
    print("-" * 30)
    
    for i in range(1, 6):  # Skip index 0 (self)
        neighbor_idx = sorted_indices[i].item()
        neighbor_label = labels[neighbor_idx].item()
        distance = distances[neighbor_idx].item()
        print(f"Rank {i}: Index {neighbor_idx}, Class {neighbor_label}, "
              f"Distance {distance:.4f}")
        
    # Save model
    print("\n9. Saving trained model...")
    model_path = "triplet_model.npz"
    mx.savez(model_path, **model.parameters())
    print(f"Model saved to {model_path}")
    
    print("\nTriplet learning demo completed successfully!")


if __name__ == "__main__":
    main()