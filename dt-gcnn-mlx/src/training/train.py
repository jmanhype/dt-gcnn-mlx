"""
Training script for DT-GCNN
Main entry point for model training with configuration management
"""

import mlx.core as mx
import argparse
import json
import logging
from pathlib import Path
import sys
import os
from typing import Dict, Any, Optional
import yaml
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ..models.dt_gcnn import DT_GCNN
from .trainer import DT_GCNN_Trainer, TrainingConfig
from .data_loader import MeshDataLoader, create_data_loaders


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def setup_logging(output_dir: Path, verbose: bool = True) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        output_dir: Directory to write log files
        verbose: Enable verbose logging (INFO level)

    Returns:
        logging.Logger: Configured logger instance
    """
    log_level = logging.INFO if verbose else logging.WARNING
    
    # Create logs directory
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Setup file handler
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def validate_gpu_setup() -> bool:
    """
    Validate GPU/Metal setup.

    Returns:
        bool: True if Metal GPU is available, False otherwise
    """
    try:
        if hasattr(mx, 'metal') and mx.metal.is_available():
            # Get Metal device info
            try:
                memory_gb = mx.metal.get_active_memory() / 1e9
                logging.info(f"Metal GPU available - Current memory: {memory_gb:.2f} GB")
            except (AttributeError, RuntimeError) as e:
                logging.warning(f"Could not get Metal memory info: {e}")
                logging.info("Metal GPU available")

            # Set memory limit to prevent crashes
            try:
                mx.metal.set_memory_limit(8 * 1024 * 1024 * 1024)  # 8GB
            except (AttributeError, RuntimeError) as e:
                logging.warning(f"Could not set Metal memory limit: {e}")

            return True
        else:
            logging.warning("Metal GPU not available - using CPU")
            return False
    except Exception as e:
        logging.error(f"Error validating GPU setup: {e}")
        logging.warning("Falling back to CPU")
        return False


def create_model(config: TrainingConfig) -> DT_GCNN:
    """Create and initialize model"""
    model = DT_GCNN(
        num_vertices=config.num_vertices,
        embedding_dim=config.embedding_dim,
        num_classes=10,  # Update based on your dataset
        dropout_rate=0.2,
        use_batch_norm=True
    )
    
    # Initialize parameters
    dummy_coords = mx.zeros((1, config.num_vertices, 3))
    dummy_features = mx.zeros((1, config.num_vertices, 6))
    _ = model(dummy_coords, dummy_features)
    
    # Log model info
    total_params = sum(p.size for p in model.parameters().values())
    logging.info(f"Model created with {total_params:,} parameters")
    
    return model


def train_model(args: argparse.Namespace) -> None:
    """
    Main training function.

    Args:
        args: Command-line arguments parsed by argparse
    """
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.verbose)
    logger.info("Starting DT-GCNN training")
    
    # Load configuration
    if args.config:
        config_dict = load_config(args.config)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and hasattr(TrainingConfig, key):
                config_dict[key] = value
        config = TrainingConfig(**config_dict)
    else:
        # Use command line arguments
        config = TrainingConfig(
            num_vertices=args.num_vertices,
            embedding_dim=args.embedding_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            checkpoint_dir=str(output_dir / "checkpoints"),
            use_metal=not args.cpu,
            verbose=args.verbose
        )
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    # Validate GPU setup
    gpu_available = validate_gpu_setup()
    if args.cpu:
        config.use_metal = False
        logger.info("Forcing CPU usage as requested")
    elif not gpu_available:
        config.use_metal = False
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        cache_data=args.cache_data
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Load checkpoint if provided
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = mx.load(args.resume)
        model.update(checkpoint['model_state'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
    
    # Create trainer
    trainer = DT_GCNN_Trainer(model, config)
    
    # Load checkpoint into trainer if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Training
    logger.info("Starting training...")
    start_time = time.time()
    
    try:
        history = trainer.train(train_loader, val_loader)
        
        # Save final model
        final_checkpoint = output_dir / "final_model.npz"
        mx.save(final_checkpoint, {
            'model_state': model.parameters(),
            'config': config.__dict__,
            'history': history
        })
        logger.info(f"Final model saved to {final_checkpoint}")
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Print final results
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time / 3600:.2f} hours")
        logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
        logger.info(f"Best validation accuracy: {max(history['val_acc']):.4f}")
        
        # Memory stats
        if config.use_metal:
            memory_stats = trainer.get_memory_stats()
            logger.info(f"Peak Metal memory: {memory_stats['metal_peak_gb']:.2f} GB")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save interrupted checkpoint
        interrupt_checkpoint = output_dir / "interrupted_model.npz"
        mx.save(interrupt_checkpoint, {
            'model_state': model.parameters(),
            'epoch': trainer.current_epoch,
            'config': config.__dict__
        })
        logger.info(f"Interrupted model saved to {interrupt_checkpoint}")
    
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


def main() -> None:
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(description="Train DT-GCNN model")
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for checkpoints and logs')
    
    # Model arguments
    parser.add_argument('--num-vertices', type=int, default=1723,
                        help='Number of mesh vertices')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='Embedding dimension')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                        help='Initial learning rate')
    
    # Configuration
    parser.add_argument('--config', type=str,
                        help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint to resume from')
    
    # Hardware arguments
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if Metal is available')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Data processing
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--cache-data', action='store_true',
                        help='Cache dataset in memory')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()