# Multi-Head Attention Implementation

This repository contains two comprehensive Jupyter notebooks demonstrating multi-head attention mechanisms used in transformer models.

## Notebooks

### 1. Multi_head_attention.ipynb
Visualizes attention heads from a pre-trained BERT model using `bertviz` for interactive neuron-level views and `matplotlib` to plot attention weights.

**Key Features:**
- Interactive attention visualization using BertViz
- Attention weight plotting across all layers and heads
- Uses pre-trained BERT model (bert-base-uncased)

### 2. Multi_head_attention_from_scratch.ipynb
A complete educational implementation of multi-head attention from scratch, covering the fundamentals of transformer architecture.

**Topics Covered:**

#### Text Tokenization
- Regular expression-based tokenization
- Byte Pair Encoding (BPE) using tiktoken
- Custom tokenizer implementation with special tokens
- Vocabulary creation and token-to-ID mapping

#### Data Preparation
- Creating input-target pairs for next-word prediction
- Implementing PyTorch DataLoader for efficient batching
- Sliding window approach for sequence generation

#### Embeddings
- Token embeddings using PyTorch nn.Embedding
- Positional embeddings for sequence order encoding
- Combining token and positional embeddings

#### Attention Mechanisms
- Self-attention implementation from scratch
- Scaled dot-product attention
- Causal attention with masking
- Multi-head attention architecture
- Query, Key, Value transformations

#### Implementation Details
- Attention score computation
- Softmax normalization and temperature scaling
- Context vector generation
- Efficient multi-head attention with weight splits

## Requirements

- Python 3.10+
- PyTorch 2.0+
- tiktoken (for BPE tokenization)
- transformers 4.30+ (for BERT models)
- bertviz 1.4.0+ (for attention visualization)
- matplotlib 3.7+ (for plotting)
- numpy (for numerical operations)

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Multi_head_attention.ipynb
Open the notebook in Jupyter/VS Code and run cells top-to-bottom. The notebook:
- Loads a pre-trained BERT model
- Visualizes attention patterns interactively
- Plots attention weights for specific tokens

**Key Parameters:**
- `model_type = 'bert'`
- `model_version = 'bert-base-uncased'`
- `sentence_a = "The artist painted the portrait of a woman with a brush"`
- Adjustable layer and head selection for focused visualization

### Multi_head_attention_from_scratch.ipynb
This notebook provides a step-by-step implementation guide:
1. Start with text tokenization basics
2. Progress through embedding creation
3. Build attention mechanisms incrementally
4. Culminate in a complete multi-head attention implementation

**Sample Text:**
The notebook uses "The Verdict" by Edith Wharton as sample text (the-verdict.txt file required).

## Key Concepts Explained

### Scaled Dot-Product Attention
The notebook explains why attention scores are divided by sqrt(d_k):
- Prevents softmax saturation with large dimensions
- Stabilizes gradients during training
- Maintains variance close to 1

### Multi-Head Attention Benefits
- Allows model to attend to different representation subspaces
- Captures various types of relationships simultaneously
- More expressive than single-head attention

### Causal Masking
- Prevents tokens from attending to future positions
- Essential for autoregressive language models
- Implemented using upper triangular mask

## Model Specifications

**GPT-2 Comparison:**
- Smallest GPT-2: 12 attention heads, 768-dim embeddings, 117M parameters
- Largest GPT-2: 25 attention heads, 1600-dim embeddings, 1.5B parameters

**BPE Tokenizer:**
- Vocabulary size: 50,257 tokens
- Handles out-of-vocabulary words via subword tokenization
- No need for unknown token handling

## Files

- `Multi_head_attention.ipynb` - BERT attention visualization
- `Multi_head_attention_from_scratch.ipynb` - Complete implementation from scratch
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Learning Path

1. **Start with Multi_head_attention.ipynb** to visualize how attention works in practice
2. **Then explore Multi_head_attention_from_scratch.ipynb** to understand the implementation details
3. Experiment with different texts, model sizes, and hyperparameters

## References

- "Attention Is All You Need" (Vaswani et al., 2017)
- GPT-2 and GPT-3 architectures
- BERT: Pre-training of Deep Bidirectional Transformers

## License

Educational resource for understanding transformer attention mechanisms.
