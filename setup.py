"""Setup script for DT-GCNN MLX package."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Package metadata
setup(
    name="dt-gcnn-mlx",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Dynamic Triplet Graph Convolutional Neural Network optimized for Apple Silicon using MLX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dt-gcnn",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/dt-gcnn/issues",
        "Documentation": "https://dt-gcnn.readthedocs.io",
        "Source Code": "https://github.com/yourusername/dt-gcnn",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(where="dt-gcnn-mlx/src"),
    package_dir={"": "dt-gcnn-mlx/src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.18.0",
            "myst-parser>=0.18.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "umap-learn>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dt-gcnn-train=dt_gcnn_mlx.cli.train:main",
            "dt-gcnn-evaluate=dt_gcnn_mlx.cli.evaluate:main",
            "dt-gcnn-export=dt_gcnn_mlx.cli.export:main",
        ],
    },
    include_package_data=True,
    package_data={
        "dt_gcnn_mlx": [
            "configs/*.yaml",
            "configs/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "deep learning",
        "graph neural networks",
        "GCN",
        "triplet loss",
        "MLX",
        "Apple Silicon",
        "machine learning",
        "neural networks",
        "NLP",
        "embeddings",
    ],
)