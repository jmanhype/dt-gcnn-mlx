"""
DT-GCNN Trainer for MLX
Training loop with joint losses, optimization, and monitoring
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Callable
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import logging

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model parameters
    num_vertices: int = 1723
    embedding_dim: int = 256
    triplet_margin: float = 0.5
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    gradient_clip: float = 1.0
    
    # Loss weights
    classification_weight: float = 1.0
    triplet_weight: float = 1.0
    
    # Scheduler parameters
    scheduler_type: str = "cosine"  # "cosine", "step", "exponential"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 5
    keep_best_n: int = 3
    
    # Device and optimization
    use_metal: bool = True
    jit_compile: bool = True
    mixed_precision: bool = False
    
    # Logging
    log_interval: int = 10
    verbose: bool = True


class LearningRateScheduler:
    """Learning rate scheduler implementations"""
    
    def __init__(self, config: TrainingConfig, optimizer: optim.Optimizer):
        self.config = config
        self.optimizer = optimizer
        self.base_lr = config.learning_rate
        self.current_epoch = 0
        
    def step(self, epoch: int, metrics: Optional[Dict[str, float]] = None):
        """Update learning rate"""
        self.current_epoch = epoch
        
        if epoch < self.config.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.config.warmup_epochs
        else:
            # Apply scheduler after warmup
            if self.config.scheduler_type == "cosine":
                lr = self._cosine_annealing(epoch - self.config.warmup_epochs)
            elif self.config.scheduler_type == "step":
                lr = self._step_decay(epoch - self.config.warmup_epochs)
            elif self.config.scheduler_type == "exponential":
                lr = self._exponential_decay(epoch - self.config.warmup_epochs)
            else:
                lr = self.base_lr
        
        # Apply minimum learning rate
        lr = max(lr, self.config.min_lr)
        
        # Update optimizer
        self.optimizer.learning_rate = lr
        return lr
    
    def _cosine_annealing(self, epoch: int) -> float:
        """Cosine annealing schedule"""
        max_epochs = self.config.num_epochs - self.config.warmup_epochs
        return self.config.min_lr + (self.base_lr - self.config.min_lr) * \
               (1 + np.cos(np.pi * epoch / max_epochs)) / 2
    
    def _step_decay(self, epoch: int, step_size: int = 30, gamma: float = 0.1) -> float:
        """Step decay schedule"""
        return self.base_lr * (gamma ** (epoch // step_size))
    
    def _exponential_decay(self, epoch: int, gamma: float = 0.95) -> float:
        """Exponential decay schedule"""
        return self.base_lr * (gamma ** epoch)


class EarlyStopping:
    """Early stopping mechanism"""
    
    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, value: float) -> bool:
        """Check if should stop training"""
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class DT_GCNN_Trainer:
    """Trainer for DT-GCNN with joint losses"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # Setup device
        if config.use_metal and mx.metal.is_available():
            mx.metal.set_memory_limit(8 * 1024 * 1024 * 1024)  # 8GB limit
            logging.info("Using Metal GPU acceleration")
        else:
            logging.info("Using CPU")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler and early stopping
        self.scheduler = LearningRateScheduler(config, self.optimizer)
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO if config.verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def compute_loss(
        self,
        embeddings: mx.array,
        logits: mx.array,
        labels: mx.array,
        triplet_indices: Optional[Tuple[mx.array, mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Compute joint loss"""
        losses = {}
        
        # Classification loss
        classification_loss = nn.losses.cross_entropy(logits, labels, reduction='mean')
        losses['classification'] = classification_loss
        
        # Triplet loss (if indices provided)
        if triplet_indices is not None:
            anchor_idx, positive_idx, negative_idx = triplet_indices
            
            # Get triplet embeddings
            anchor_emb = embeddings[anchor_idx]
            positive_emb = embeddings[positive_idx]
            negative_emb = embeddings[negative_idx]
            
            # Compute distances
            pos_dist = mx.sum((anchor_emb - positive_emb) ** 2, axis=1)
            neg_dist = mx.sum((anchor_emb - negative_emb) ** 2, axis=1)
            
            # Triplet loss
            triplet_loss = mx.maximum(
                pos_dist - neg_dist + self.config.triplet_margin,
                0.0
            ).mean()
            losses['triplet'] = triplet_loss
            
            # Total loss
            total_loss = (self.config.classification_weight * classification_loss +
                         self.config.triplet_weight * triplet_loss)
        else:
            total_loss = classification_loss
        
        losses['total'] = total_loss
        return total_loss, losses
    
    def train_step(
        self,
        batch: Dict[str, mx.array]
    ) -> Tuple[mx.array, Dict[str, float]]:
        """Single training step"""
        def loss_fn(params):
            # Forward pass
            embeddings, logits = self.model(
                batch['coordinates'],
                batch['features']
            )
            
            # Compute loss
            loss, loss_dict = self.compute_loss(
                embeddings,
                logits,
                batch['labels'],
                batch.get('triplet_indices', None)
            )
            
            return loss, (loss_dict, embeddings, logits)
        
        # Compute gradients
        grad_fn = mx.grad(loss_fn, has_aux=True)
        grads, (loss_dict, embeddings, logits) = grad_fn(self.model.parameters())
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            grads = tree_map(
                lambda g: mx.clip(g, -self.config.gradient_clip, self.config.gradient_clip),
                grads
            )
        
        # Update parameters
        self.optimizer.update(self.model, grads)
        
        # Compute accuracy
        predictions = mx.argmax(logits, axis=1)
        accuracy = mx.mean(predictions == batch['labels'])
        
        # Convert to Python scalars for logging
        metrics = {
            'loss': float(loss_dict['total']),
            'classification_loss': float(loss_dict['classification']),
            'accuracy': float(accuracy)
        }
        
        if 'triplet' in loss_dict:
            metrics['triplet_loss'] = float(loss_dict['triplet'])
        
        return loss_dict['total'], metrics
    
    @mx.compile
    def validation_step(
        self,
        batch: Dict[str, mx.array]
    ) -> Tuple[mx.array, Dict[str, float]]:
        """Single validation step"""
        # Forward pass
        embeddings, logits = self.model(
            batch['coordinates'],
            batch['features']
        )
        
        # Compute loss
        loss, loss_dict = self.compute_loss(
            embeddings,
            logits,
            batch['labels'],
            batch.get('triplet_indices', None)
        )
        
        # Compute accuracy
        predictions = mx.argmax(logits, axis=1)
        accuracy = mx.mean(predictions == batch['labels'])
        
        # Convert to Python scalars
        metrics = {
            'loss': float(loss_dict['total']),
            'classification_loss': float(loss_dict['classification']),
            'accuracy': float(accuracy)
        }
        
        if 'triplet' in loss_dict:
            metrics['triplet_loss'] = float(loss_dict['triplet'])
        
        return loss, metrics
    
    def train_epoch(
        self,
        train_loader: Any,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_metrics = {
            'loss': 0.0,
            'classification_loss': 0.0,
            'triplet_loss': 0.0,
            'accuracy': 0.0
        }
        
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss, metrics = self.train_step(batch)
            
            # Update epoch metrics
            for key, value in metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
            
            num_batches += 1
            
            # Logging
            if self.config.verbose and batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"Acc: {metrics['accuracy']:.4f}"
                )
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        epoch_metrics['epoch_time'] = time.time() - start_time
        
        return epoch_metrics
    
    def validate(
        self,
        val_loader: Any
    ) -> Dict[str, float]:
        """Validate model"""
        val_metrics = {
            'loss': 0.0,
            'classification_loss': 0.0,
            'triplet_loss': 0.0,
            'accuracy': 0.0
        }
        
        num_batches = 0
        
        for batch in val_loader:
            _, metrics = self.validation_step(batch)
            
            for key, value in metrics.items():
                if key in val_metrics:
                    val_metrics[key] += value
            
            num_batches += 1
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.parameters(),
            'optimizer_state': self.optimizer.state,
            'metrics': metrics,
            'config': asdict(self.config),
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.npz"
        mx.save(checkpoint_path, checkpoint)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.npz"
            mx.save(best_path, checkpoint)
            self.logger.info(f"Saved best model with val_loss: {metrics['val_loss']:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = mx.load(checkpoint_path)
        
        # Restore model state
        self.model.update(checkpoint['model_state'])
        
        # Restore optimizer state
        self.optimizer.state = checkpoint['optimizer_state']
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint['metrics']
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the best N"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.npz"))
        
        if len(checkpoints) > self.config.keep_best_n:
            # Sort by modification time
            checkpoints.sort(key=lambda p: p.stat().st_mtime)
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-self.config.keep_best_n]:
                checkpoint.unlink()
    
    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """Full training loop"""
        num_epochs = num_epochs or self.config.num_epochs
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model parameters: {sum(p.size for p in tree_flatten(self.model.parameters())[0]):,}")
        
        if self.config.use_metal:
            self.logger.info(f"Metal memory usage: {mx.metal.get_active_memory() / 1e9:.2f} GB")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Update learning rate
            lr = self.scheduler.step(epoch)
            
            # Training
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs} (LR: {lr:.6f})")
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['learning_rates'].append(lr)
            
            # Log epoch results
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Time: {train_metrics['epoch_time']:.1f}s"
            )
            
            # Check for best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_frequency == 0 or is_best:
                self.save_checkpoint(
                    epoch,
                    {'train': train_metrics, 'val': val_metrics},
                    is_best
                )
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Memory management
            if self.config.use_metal and epoch % 10 == 0:
                mx.metal.clear_cache()
                self.logger.info(f"Metal memory: {mx.metal.get_active_memory() / 1e9:.2f} GB")
        
        self.logger.info("Training completed!")
        return self.training_history
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {
            'model_params': sum(p.size for p in tree_flatten(self.model.parameters())[0]),
            'model_mb': sum(p.nbytes for p in tree_flatten(self.model.parameters())[0]) / 1e6
        }
        
        if self.config.use_metal and mx.metal.is_available():
            stats['metal_active_gb'] = mx.metal.get_active_memory() / 1e9
            stats['metal_peak_gb'] = mx.metal.get_peak_memory() / 1e9
        
        return stats