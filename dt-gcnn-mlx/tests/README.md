# DT-GCNN Test Suite

Comprehensive test suite for the DT-GCNN MLX implementation, covering model architecture, loss functions, data pipeline, and training procedures.

## Test Modules

### 1. Model Tests (`test_model.py`)
Tests for the core DT-GCNN architecture:
- **TestGraphConvLayer**: Graph convolutional layer functionality
  - Layer initialization and weight shapes
  - Forward pass computation
  - Gradient flow verification
- **TestDilatedTemporalConvLayer**: Temporal convolution with dilation
  - Dilated convolution mechanics
  - Receptive field calculations
  - Temporal pattern processing
- **TestDTGCNN**: Complete model integration
  - Model initialization with various configurations
  - Forward pass with different input sizes
  - Embedding extraction
  - Training vs evaluation mode behavior

### 2. Triplet Loss Tests (`test_triplet_loss.py`)
Tests for triplet loss implementations:
- **TestTripletLoss**: Basic triplet loss functionality
  - Loss computation with anchor/positive/negative triplets
  - Margin enforcement
  - Easy vs hard negative handling
- **TestBatchHardTripletLoss**: Batch hard mining strategy
  - Hard triplet mining within batches
  - Handling of no valid triplets
  - Pairwise distance computation
- **TestBatchAllTripletLoss**: Batch all strategy
  - All valid triplets computation
  - Triplet counting validation
- **TestLossFunctionIntegration**: Integration with models
  - Loss with model outputs
  - Gradient computation through loss

### 3. Data Pipeline Tests (`test_data_pipeline.py`)
Tests for data loading and preprocessing:
- **TestGraphDataset**: Dataset loading and management
  - Loading from files
  - Indexing and batching
  - Data augmentation pipeline
- **TestGraphBatchSampler**: Balanced batch sampling
  - Class-balanced sampling
  - Sampler exhaustion handling
- **TestPreprocessing**: Preprocessing utilities
  - Adjacency matrix normalization
  - Graph Laplacian computation
  - Temporal signal augmentation
  - Sliding window creation
- **TestDataPipeline**: End-to-end pipeline integration

### 4. Training Tests (`test_training.py`)
Tests for training procedures:
- **TestTrainingConfig**: Configuration validation
- **TestDTGCNNTrainer**: Training loop functionality
  - Single training step
  - Validation step
  - Triplet loss training
  - Gradient clipping
- **TestEarlyStopping**: Early stopping callback
  - Patience mechanism
  - Improvement detection
- **TestModelCheckpoint**: Model checkpointing
  - Checkpoint saving/loading
  - Best model tracking
- **TestLearningRateScheduler**: LR scheduling
  - Step scheduler
  - Cosine annealing
  - Warmup scheduling
- **TestTrainingIntegration**: Full training loop

## Running Tests

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Module
```bash
python run_tests.py -m model
python run_tests.py -m triplet_loss
python run_tests.py -m data_pipeline
python run_tests.py -m training
```

### Run Specific Test Class
```bash
python run_tests.py model.TestDTGCNN
python run_tests.py triplet_loss.TestBatchHardTripletLoss
```

### Run Specific Test Method
```bash
python run_tests.py model.TestDTGCNN.test_forward_pass
python run_tests.py data_pipeline.TestPreprocessing.test_adjacency_normalization
```

### List Available Tests
```bash
python run_tests.py -l
```

### Verbosity Levels
```bash
python run_tests.py -v 0  # Quiet
python run_tests.py -v 1  # Normal
python run_tests.py -v 2  # Verbose (default)
```

## Test Coverage

The test suite covers:
- ✅ Model architecture components
- ✅ Loss function implementations
- ✅ Data loading and preprocessing
- ✅ Training procedures and callbacks
- ✅ Gradient flow and backpropagation
- ✅ Edge cases and error handling
- ✅ Performance characteristics
- ✅ Integration between components

## Writing New Tests

To add new tests:

1. Create a test file following the naming convention `test_*.py`
2. Import unittest and the modules to test
3. Create test classes inheriting from `unittest.TestCase`
4. Write test methods starting with `test_`

Example:
```python
import unittest
import mlx.core as mx
from src.models.new_module import NewFeature

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.feature = NewFeature()
        
    def test_functionality(self):
        """Test basic functionality."""
        result = self.feature.process(mx.array([1, 2, 3]))
        self.assertEqual(result.shape, (3,))
```

## Test Data

Tests use synthetic data generated on-the-fly. For tests requiring persistent data, use the datasets in `../datasets/synthetic/`.

## Continuous Integration

The test suite is designed to be run in CI/CD pipelines:
```bash
# Run tests and exit with appropriate code
python run_tests.py
# Exit code 0 = all tests passed
# Exit code 1 = some tests failed
```

## Performance Testing

For performance benchmarks, see `../examples/performance_benchmark.py` which provides detailed performance analysis on Apple Silicon.