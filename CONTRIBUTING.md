# Contributing to DT-GCNN

First off, thank you for considering contributing to DT-GCNN! It's people like you that make DT-GCNN such a great tool.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How Can I Contribute?](#how-can-i-contribute)
4. [Development Setup](#development-setup)
5. [Style Guidelines](#style-guidelines)
6. [Commit Messages](#commit-messages)
7. [Pull Request Process](#pull-request-process)
8. [Testing](#testing)
9. [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [your.email@example.com].

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your feature or bugfix
5. Make your changes
6. Submit a pull request

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include system information** (OS, Python version, MLX version)
- **Include logs and error messages**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the proposed enhancement**
- **Explain why this enhancement would be useful**
- **List any alternative solutions you've considered**

### Code Contributions

#### Before Starting

1. Check if there's an existing issue for your contribution
2. If not, create an issue to discuss your proposed changes
3. Wait for feedback before starting major work

#### Categories of Contributions

- **Bug fixes**: Fix reported issues
- **Features**: Add new functionality
- **Performance**: Improve speed or memory usage
- **Documentation**: Improve or add documentation
- **Tests**: Add missing tests or improve existing ones

## Development Setup

### Prerequisites

- macOS 13.0+ with Apple Silicon (M1/M2/M3)
- Python 3.8+
- Git

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/dt-gcnn.git
cd dt-gcnn

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

### Development Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing
- **pre-commit**: Git hooks

## Style Guidelines

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use Black for formatting
- Use type hints for all new code

```python
# Good
def compute_similarity(
    embeddings: mx.array,
    metric: str = "cosine"
) -> mx.array:
    """Compute pairwise similarity matrix.
    
    Args:
        embeddings: Input embeddings of shape (n, d)
        metric: Similarity metric to use
        
    Returns:
        Similarity matrix of shape (n, n)
    """
    ...

# Bad
def compute_similarity(embeddings, metric="cosine"):
    # No docstring, no type hints
    ...
```

### Documentation Style

- Use Google-style docstrings
- Include type information in docstrings
- Add examples for complex functions

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description of function.
    
    Longer description if needed, explaining the purpose
    and behavior of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> result = example_function(42, "test")
        >>> print(result)
        True
    """
    ...
```

### File Organization

```
dt-gcnn/
├── dt-gcnn-mlx/
│   ├── src/
│   │   ├── dt_gcnn_mlx/
│   │   │   ├── __init__.py
│   │   │   ├── models/      # Model implementations
│   │   │   ├── data/        # Data loading and processing
│   │   │   ├── training/    # Training utilities
│   │   │   ├── losses/      # Loss functions
│   │   │   └── utils/       # Helper functions
│   └── tests/
│       ├── unit/            # Unit tests
│       ├── integration/     # Integration tests
│       └── benchmarks/      # Performance benchmarks
├── docs/                    # Documentation
├── examples/               # Example scripts
└── configs/               # Configuration files
```

## Commit Messages

We follow the Conventional Commits specification:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(models): add attention mechanism to GCN layers

Implement multi-head attention for dynamic graph weighting.
This improves model performance on sparse graphs.

Closes #123
```

```
fix(training): resolve memory leak in gradient accumulation

Clear computation graph after each accumulation step to
prevent memory buildup during long training runs.

Fixes #456
```

## Pull Request Process

1. **Update your fork**
   ```bash
   git checkout main
   git pull upstream main
   git push origin main
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation
   - Run tests locally

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

### PR Requirements

- [ ] Code passes all tests
- [ ] Code follows style guidelines
- [ ] Tests are included for new functionality
- [ ] Documentation is updated
- [ ] Commit messages follow conventions
- [ ] PR description clearly explains changes

### PR Review Process

1. Automated checks run (tests, linting, etc.)
2. Code review by maintainers
3. Address feedback
4. Approval and merge

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=dt_gcnn_mlx --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run tests in parallel
pytest -n auto
```

### Writing Tests

```python
import pytest
import mlx.core as mx
from dt_gcnn_mlx import DTGCNN

class TestDTGCNN:
    @pytest.fixture
    def model(self):
        """Create model instance for testing."""
        return DTGCNN(input_dim=768, output_dim=128)
    
    def test_forward_pass(self, model):
        """Test model forward pass."""
        batch_size = 4
        seq_length = 128
        
        input_ids = mx.random.randint(0, 1000, (batch_size, seq_length))
        attention_mask = mx.ones((batch_size, seq_length))
        
        output = model(input_ids, attention_mask)
        
        assert output.shape == (batch_size, 128)
        assert not mx.any(mx.isnan(output))
```

### Test Categories

- **Unit tests**: Test individual components
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark speed and memory
- **Regression tests**: Prevent previously fixed bugs

## Documentation

### Adding Documentation

1. **API Documentation**: Add docstrings to all public functions
2. **User Guides**: Update guides in `docs/`
3. **Examples**: Add runnable examples in `examples/`
4. **README**: Update if adding major features

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View locally
open _build/html/index.html
```

### Documentation Standards

- Write for clarity and completeness
- Include code examples
- Explain not just "what" but "why"
- Keep examples up-to-date
- Test all code in documentation

## Advanced Contributing

### Performance Optimization

When optimizing performance:

1. **Profile first**: Identify bottlenecks
2. **Benchmark**: Measure improvements
3. **Document**: Explain optimizations
4. **Test**: Ensure correctness

```python
# Example benchmark
import time
import mlx.core as mx

def benchmark_operation(func, inputs, num_runs=100):
    # Warmup
    for _ in range(10):
        func(*inputs)
    
    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        func(*inputs)
    mx.eval()  # Ensure completion
    
    elapsed = time.time() - start
    return elapsed / num_runs
```

### Adding New Models

When adding new model variants:

1. Inherit from base classes
2. Follow naming conventions
3. Add comprehensive tests
4. Document differences
5. Add to model registry

### Debugging Tips

1. **Use debug mode**: `export PYTHONFAULTHANDLER=1`
2. **Enable MLX debugging**: `export MLX_DEBUG=1`
3. **Profile memory**: Use memory profilers
4. **Check shapes**: Print tensor shapes frequently

## Questions?

Feel free to:

- Open an issue for questions
- Join our Discord community
- Email maintainers directly

Thank you for contributing to DT-GCNN! 🎉