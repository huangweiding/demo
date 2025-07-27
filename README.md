# Sliding Window Attention Demo

This repository contains a PyTorch implementation of a sliding window attention mechanism with RMSNorm, demonstrating how to build a transformer model that uses different attention patterns for different layers.

## Features

- **RMSNorm**: Root Mean Square Layer Normalization implementation
- **SlidingWindowLayer**: Attention layer that can switch between full attention and sliding window attention
- **MyModel**: Complete transformer model with configurable attention patterns

## Requirements

- Python 3.10+
- PyTorch 2.2.2+
- NumPy < 2.0 (for compatibility with PyTorch)

## Installation

```bash
# Create a conda environment
conda create -n demo python=3.10
conda activate demo

# Install PyTorch
pip install torch

# Install NumPy (ensure version < 2.0 for PyTorch compatibility)
pip install "numpy<2"
```

## Usage

```python
import torch
from sliding_window_demo import MyModel

# Create model with sliding window attention
model = MyModel(
    vocab_size=1000,
    hidden_size=512,
    num_heads=8,
    num_layers=12,
    sliding_window=1024,
    max_num_layer=6  # First 6 layers use full attention, rest use sliding window
)

# Test the model
x = torch.randint(0, 1000, (1, 128))  # Batch size 1, sequence length 128
output = model(x)
```

## Model Architecture

The model implements a hybrid attention approach:
- **Full Attention Layers**: The first `max_num_layer` layers use standard full attention
- **Sliding Window Attention Layers**: Remaining layers use sliding window attention for efficiency

This approach balances computational efficiency with model performance, similar to models like Mistral-7B.

## Files

- `sliding_window_demo.py`: Main implementation file containing all classes and demo code
- `README.md`: This documentation file

## License

This project is open source and available under the MIT License. 