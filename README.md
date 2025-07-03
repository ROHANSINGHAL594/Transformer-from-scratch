# 🧠 Transformer from Scratch: "Attention Is All You Need" in PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

This repository provides a complete implementation of the Transformer model, as described in the groundbreaking paper "[Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)" (Vaswani et al., 2017), built from scratch using PyTorch. The goal is to offer a clear, modular, and educational codebase for understanding the inner workings of this revolutionary architecture.

## ✨ Introduction to Transformers

Transformers have revolutionized the field of Natural Language Processing (NLP) and beyond. Unlike traditional recurrent neural networks (RNNs), they leverage self-attention mechanisms to process entire sequences in parallel, enabling efficient learning of long-range dependencies. This parallelization significantly speeds up training and allows for handling much larger datasets, leading to breakthroughs in machine translation, text generation, and various other AI applications.

## 🚀 Features

This implementation includes all core components of the original Transformer architecture:

-   **Multi-Head Attention**: Efficiently captures diverse relationships within sequences.
-   **Positional Encoding**: Infuses sequence order information into the embeddings.
-   **Add & Norm Layer**: Residual connections followed by layer normalization for stable training.
-   **Position-wise Feed-Forward Networks**: Applies non-linear transformations to each position independently.
-   **Encoder Block**: Stacks multiple layers of self-attention and feed-forward networks.
-   **Decoder Block**: Incorporates masked self-attention and encoder-decoder attention for sequence generation.
-   **Complete Transformer Model**: Integrates all components into an end-to-end trainable architecture.

## 🛠️ Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ROHANSINGHAL594/Transformer-from-scratch.git
cd Transformer-from-scratch/Attention is all you need
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## 📊 Training

The model has been trained on the `eng-french.csv` dataset, a real-world translation dataset. Due to hardware limitations (lack of a powerful GPU), the training was conducted for a limited number of epochs.

To train the model, run the `train2.py` script:

```bash
python train2.py
```

### Training Results (Loss per Epoch)

| Epoch | Loss     |
| :---- | :------- |
| 01    | 3.7462   |
| 02    | 2.6549   |
| 03    | 2.2281   |
| 04    | 1.9559   |
| 05    | 1.7655   |
| 06    | 1.6198   |
| 07    | 1.5045   |
| 08    | 1.4103   |
| 09    | 1.3320   |
| 10    | 1.2645   |

The model is now trained and can be used for further experimentation or fine-tuning.

## 🚀 Usage

You can integrate and use the Transformer model in your own PyTorch projects. Here's a basic example:

```python
import torch
from transformer import Transformer

# Example parameters (adjust based on your vocabulary sizes and model configuration)
input_vocab_size = 10000  # Size of the source language vocabulary
target_vocab_size = 10000 # Size of the target language vocabulary
d_model = 512             # Dimension of the model's embeddings
num_heads = 8             # Number of attention heads
num_layers = 3            # Number of encoder/decoder layers
d_ff = 512                # Dimension of the feed-forward network
dropout = 0.1             # Dropout rate

# Instantiate the Transformer model
model = Transformer(
    d_model=d_model,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    dff=d_ff,
    heads=num_heads,
    number_layers=num_layers,
    dropout=dropout
)

# Example input tensors (batch_size, sequence_length)
# These should be numericalized and padded sequences
# For actual use, replace with your tokenized and indexed data
src_data = torch.randint(1, input_vocab_size, (64, 50)) # Example source sequence
tgt_data = torch.randint(1, target_vocab_size, (64, 50)) # Example target sequence (shifted right)

# Forward pass
output = model(src_data, tgt_data)
print(f"Output shape: {output.shape}") # Expected: (batch_size, sequence_length, target_vocab_size)
```

For more detailed usage and implementation specifics, refer to the `transformer.py` file.

## 📚 References

-   [Attention Is All You Need (2017) - Original Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
-   PyTorch Official Documentation

## 🚧 Future Improvements

-   Add BLEU score evaluation for quantitative assessment of translation quality.
-   Implement functionality to save and load model checkpoints.
-   Visualize attention weights to understand the model's focus.
-   Explore training on larger datasets with more computational resources.

## 🙋‍♂️ Author's Note

This project represents my first in-depth implementation of a complex research paper from scratch. It has been an invaluable learning experience, providing a deep understanding of how each component of the Transformer architecture functions. Despite encountering numerous bugs and challenges, the process of debugging and resolving them has significantly enhanced my problem-solving skills.

I welcome any feedback, suggestions, or collaboration opportunities. I hope this repository serves as a helpful resource for others embarking on their journey to understand and implement Transformers!