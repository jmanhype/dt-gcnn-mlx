"""Triplet learning demonstration using DT-GCNN for graph embeddings."""

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

from src.models.dt_gcnn import DTGCNN
from src.losses.triplet_loss import BatchHardTripletLoss, BatchAllTripletLoss
from src.data.preprocessing import normalize_adjacency_matrix


def generate_graph_patterns(num_samples_per_class=50, num_nodes=15, num_features=6,
                           seq_length=20, num_classes=4):
    """Generate distinct graph temporal patterns for each class."""
    
    # Create a simple ring graph with shortcuts
    adj = np.eye(num_nodes)
    
    # Ring connections
    for i in range(num_nodes):
        adj[i, (i + 1) % num_nodes] = 1
        adj[(i + 1) % num_nodes, i] = 1
        
    # Add some shortcuts
    for i in range(0, num_nodes, 3):
        j = (i + num_nodes // 2) % num_nodes
        adj[i, j] = adj[j, i] = 0.5
        
    # Normalize
    adj = adj / adj.sum(axis=1, keepdims=True)
    
    # Generate distinct patterns for each class
    features = []
    labels = []
    
    for class_id in range(num_classes):
        for _ in range(num_samples_per_class):
            # Time array
            t = np.linspace(0, 2 * np.pi, seq_length)
            
            # Create class-specific patterns
            node_features = np.zeros((seq_length, num_nodes, num_features))
            
            if class_id == 0:
                # Propagating wave pattern
                for node in range(num_nodes):
                    phase = 2 * np.pi * node / num_nodes
                    for feat in range(num_features):
                        node_features[:, node, feat] = (
                            np.sin(t + phase + feat * 0.5) + 
                            np.random.randn(seq_length) * 0.1
                        )
                        
            elif class_id == 1:
                # Synchronized oscillation pattern
                base_signal = np.cos(2 * t) + 0.5 * np.cos(4 * t)
                for node in range(num_nodes):
                    for feat in range(num_features):
                        node_features[:, node, feat] = (
                            base_signal * (1 + node * 0.05) + 
                            np.random.randn(seq_length) * 0.1
                        )
                        
            elif class_id == 2:
                # Localized activity pattern
                active_nodes = np.random.choice(num_nodes, size=num_nodes//3, replace=False)
                for node in active_nodes:
                    for feat in range(num_features):
                        node_features[:, node, feat] = (
                            np.exp(-((t - np.pi) ** 2) / 0.5) * (1 + feat * 0.2) +
                            np.random.randn(seq_length) * 0.1
                        )
                        
            else:
                # Chaotic pattern
                for node in range(num_nodes):
                    x = np.random.randn()
                    for i in range(1, seq_length):
                        # Logistic map
                        x = 3.9 * x * (1 - x)
                        for feat in range(num_features):
                            node_features[i, node, feat] = (
                                x + np.random.randn() * 0.1
                            )
                            
            features.append(node_features)
            labels.append(class_id)
            
    return np.array(features), np.array(labels), adj


def visualize_embeddings(embeddings, labels, title="Embeddings Visualization"):
    """Visualize embeddings using t-SNE."""
    # Convert to numpy if needed
    if hasattr(embeddings, 'numpy'):
        embeddings = np.array(embeddings)
    if hasattr(labels, 'numpy'):
        labels = np.array(labels)
        
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    class_names = ['Wave', 'Synchronized', 'Localized', 'Chaotic']
    
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
    embeddings_np = np.array(embeddings)
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
    print("\n1. Generating graph pattern data...")
    num_samples_per_class = 40
    num_classes = 4
    num_nodes = 15
    num_features = 6
    seq_length = 20
    embedding_dim = 128
    
    features, labels, adj = generate_graph_patterns(
        num_samples_per_class=num_samples_per_class,
        num_nodes=num_nodes,
        num_features=num_features,
        seq_length=seq_length,
        num_classes=num_classes
    )
    
    print(f"   Total samples: {len(features)}")
    print(f"   Classes: {num_classes}")
    print(f"   Samples per class: {num_samples_per_class}")
    
    # Convert to MLX
    features = mx.array(features)
    labels = mx.array(labels)
    adj_matrix = mx.array(adj)
    
    # Split data
    train_size = int(0.8 * len(features))
    indices = np.random.permutation(len(features))
    
    train_features = features[indices[:train_size]]
    train_labels = labels[indices[:train_size]]
    val_features = features[indices[train_size:]]
    val_labels = labels[indices[train_size:]]
    
    print(f"\n2. Data split:")
    print(f"   Training samples: {len(train_features)}")
    print(f"   Validation samples: {len(val_features)}")
    
    # Create model
    print("\n3. Creating DT-GCNN model for embeddings...")
    model = DTGCNN(
        num_nodes=num_nodes,
        input_dim=num_features,
        hidden_dims=[32, 64, 128],
        temporal_kernel_size=3,
        dilations=[1, 2, 4],
        num_classes=embedding_dim,  # Output embedding dimension
        dropout=0.3,
        use_embeddings=True  # No final classification layer
    )
    
    # Setup training
    print("\n4. Setting up triplet loss training...")
    triplet_loss = BatchHardTripletLoss(margin=0.3)
    optimizer = optim.Adam(learning_rate=0.001)
    
    # Get initial embeddings
    print("\n5. Computing initial embeddings...")
    model.eval()
    initial_embeddings = model.get_embeddings(val_features, adj_matrix)
    visualize_embeddings(initial_embeddings, val_labels, "Initial Embeddings (Untrained)")
    compute_embedding_statistics(initial_embeddings, val_labels)
    
    # Training
    batch_size = 32  # Should have multiple samples per class
    num_epochs = 100
    
    print(f"\n6. Training with triplet loss for {num_epochs} epochs...")
    print("-" * 50)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle data while keeping class balance
        for start_idx in range(0, len(train_features) - batch_size + 1, batch_size):
            # Get balanced batch (ensure multiple samples per class)
            batch_indices = []
            samples_per_class = batch_size // num_classes
            
            for class_id in range(num_classes):
                class_indices = np.where(np.array(train_labels) == class_id)[0]
                selected = np.random.choice(class_indices, 
                                          size=min(samples_per_class, len(class_indices)),
                                          replace=False)
                batch_indices.extend(selected)
                
            if len(batch_indices) < 4:  # Need at least 4 samples
                continue
                
            batch_features = train_features[batch_indices]
            batch_labels = train_labels[batch_indices]
            
            # Forward pass
            def loss_fn(model, features, adj, labels):
                embeddings = model.get_embeddings(features, adj)
                loss = triplet_loss(embeddings, labels)
                return loss
                
            # Compute loss and gradients
            loss, grads = mx.value_and_grad(loss_fn)(
                model, batch_features, adj_matrix, batch_labels
            )
            
            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            epoch_loss += loss.item()
            num_batches += 1
            
        # Validation
        model.eval()
        val_embeddings = model.get_embeddings(val_features, adj_matrix)
        val_loss = triplet_loss(val_embeddings, val_labels)
        
        # Record losses
        avg_train_loss = epoch_loss / max(num_batches, 1)
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
            
    print("-" * 50)
    
    # Get final embeddings
    print("\n7. Computing final embeddings...")
    model.eval()
    final_embeddings = model.get_embeddings(val_features, adj_matrix)
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
    all_embeddings = model.get_embeddings(features, adj_matrix)
    
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
    mx.save(model_path, model.parameters())
    print(f"Model saved to {model_path}")
    
    print("\nTriplet learning demo completed successfully!")


if __name__ == "__main__":
    main()