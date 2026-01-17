# Generative AI Chatbot (Transformer from Scratch)

A character-level Generative Pre-trained Transformer (GPT) language model engineered from the ground up using PyTorch.

This project implements the core architecture of the Transformer model manually, including Multi-Head Self-Attention, Feed-Forward Networks, and Residual Connections. It is designed to demonstrate deep learning fundamentals and efficient model training using CUDA acceleration.

## Key Features

* **Decoder-Only Architecture**: Implements a causal GPT-style transformer for text generation.
* **Custom Implementation**: Manual construction of attention mechanisms, embeddings, and normalization layers rather than using pre-built abstract classes.
* **CUDA Acceleration**: Optimized training pipeline leveraging NVIDIA GPUs for parallelized tensor operations.
* **Hyperparameter Tuning**: Tuned learning rates, context window sizes, and dropout to achieve a cross-entropy loss below 0.5.


## Usage

### 1. Installation
Install PyTorch with CUDA support (recommended) or CPU version:
```bash
pip install torch
