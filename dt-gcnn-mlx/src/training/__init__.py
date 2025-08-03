"""
Training utilities for DT-GCNN
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class DT_GCNN_Trainer:
    """Simple trainer for demo purposes"""
    
    def __init__(self, model, triplet_loss, miner, learning_rate=1e-3, 
                 weight_decay=0.01, num_classes=4, classification_weight=1.0, 
                 triplet_weight=1.0):
        self.model = model
        self.triplet_loss = triplet_loss
        self.miner = miner
        self.num_classes = num_classes
        self.classification_weight = classification_weight
        self.triplet_weight = triplet_weight
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    
    def train_step(self, anchor, positive, negative, labels):
        """Perform one training step"""
        
        def loss_fn(params):
            # Forward passes with embeddings
            logits_a, z_anchor = self.model(anchor, return_embeddings=True)
            _, z_pos = self.model(positive, return_embeddings=True)
            _, z_neg = self.model(negative, return_embeddings=True)
            
            # Classification loss
            ce_loss = mx.mean(nn.losses.cross_entropy(logits_a, labels))
            
            # Triplet loss (simplified)
            triplet_distances = mx.sum((z_anchor - z_pos) ** 2, axis=1) - mx.sum((z_anchor - z_neg) ** 2, axis=1) + 0.3
            trip_loss = mx.mean(mx.maximum(triplet_distances, 0.0))
            
            # Combined loss
            total_loss = self.classification_weight * ce_loss + self.triplet_weight * trip_loss
            
            return total_loss, {"classification_loss": ce_loss, "triplet_loss": trip_loss}
        
        # Compute loss and gradients  
        loss_and_grad = mx.value_and_grad(loss_fn)
        (loss, metrics), grads = loss_and_grad(self.model.parameters())
        
        # Update parameters
        self.optimizer.update(self.model, grads)
        
        # Add total loss to metrics
        metrics["loss"] = loss
        
        # Compute accuracy
        logits_a, _ = self.model(anchor, return_embeddings=True)
        predictions = mx.argmax(logits_a, axis=1)
        accuracy = mx.mean(predictions == labels)
        metrics["accuracy"] = accuracy
        
        return {k: float(v) for k, v in metrics.items()}

__all__ = ["DT_GCNN_Trainer"]