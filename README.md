# Word2Vec K-Words-to-Next Model

A PyTorch implementation of a Word2Vec variant that predicts the next word given k previous words. Features position-aware embeddings and an interactive autocomplete demo.

## Features

- **Position-aware embeddings**: Separate embeddings for each word position
- **Improved architecture**: Multi-layer network with dropout and layer normalization
- **MPS acceleration**: Automatic GPU acceleration on Apple Silicon
- **Interactive autocomplete**: Real-time next-word predictions
- **Efficient training**: Handles millions of training sequences

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd w2v

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Model

```bash
# Quick training (3 epochs, ~10 min on MPS)
python scripts/train.py --k 3 --epochs 3 --batch_size 512 --vocab_size 10000

# Better quality (10 epochs)
python scripts/train.py --k 4 --epochs 10 --batch_size 256 --vocab_size 15000
```

### 2. Run Autocomplete

```bash
# Interactive autocomplete console
python autocomplete.py
```

Type sentences and see predictions after entering k words (default k=3).

### 3. Explore Word2Vec Embeddings

```bash
# Interactive embedding explorer
python explore_embeddings.py

# Or find similar words directly
python explore_embeddings.py --word king
```

#### Embedding Explorer Commands:

**Word Similarity:**
```bash
>>> similar king
Most similar to 'king':
  prince: 0.812
  queen: 0.794
  emperor: 0.765
  monarch: 0.742
  throne: 0.728
```

**Compare Two Words:**
```bash
>>> similarity king queen
Similarity between 'king' and 'queen': 0.794

>>> similarity king computer
Similarity between 'king' and 'computer': 0.123
```

**Word Analogies** (king - man + woman = ?):
```bash
>>> analogy king man woman
king is to man as woman is to:
  queen: 0.752
  princess: 0.694
  daughter: 0.671
```

**Vector Arithmetic:**
```bash
>>> add paris germany
paris + germany ≈
  france: 0.821
  berlin: 0.795
  europe: 0.743

>>> subtract paris - france
paris - france ≈
  london: 0.687
  berlin: 0.654
  rome: 0.632
```

### 4. Inspect Training Data (Optional)

```bash
# Save first 1000 training sequences to file
python scripts/train.py --save_sequences --max_sequences 1000 --epochs 0

# This will create sequences.txt showing:
# [the quick brown] → fox
# [quick brown fox] → jumps
# etc.
```

## Model Architecture

- **Input**: k consecutive words (via position-aware embeddings)
- **Hidden layers**: 3 fully-connected layers with ReLU, LayerNorm, and Dropout
- **Output**: Probability distribution over vocabulary for next word

### Key Parameters

- `k`: Number of context words (default: 3)
- `vocab_size`: Maximum vocabulary size (default: 20000)
- `embedding_dim`: Embedding dimension (default: 300)
- `hidden_dim`: Hidden layer dimension (default: 512)
- `dropout`: Dropout rate for regularization (default: 0.2)

## Detailed Instructions

### How to Train the Model

1. **Basic Training** (recommended for first-time users):
```bash
# Train for 3 epochs with default settings (~15 minutes on MPS)
python scripts/train.py --k 3 --epochs 3

# The model will:
# - Download Text8 dataset automatically (first run only)
# - Create ~10 million training sequences
# - Save best model to checkpoints/improved_best.pt
# - Show progress bar and loss for each epoch
```

2. **Custom Training** (for better results):
```bash
# Higher quality model with more context and larger vocabulary
python scripts/train.py \
    --k 4 \                    # Use 4 context words (default: 3)
    --epochs 10 \              # Train for 10 epochs (default: 10)
    --batch_size 256 \         # Batch size (default: 256)
    --vocab_size 20000 \       # Vocabulary size (default: 20000)
    --embedding_dim 300 \      # Embedding dimension (default: 300)
    --hidden_dim 512 \         # Hidden layer size (default: 512)
    --learning_rate 0.001 \    # Learning rate (default: 0.001)
    --dropout 0.2              # Dropout rate (default: 0.2)
```

3. **Monitor Training**:
- Watch the loss decrease each epoch (lower is better)
- Model evaluates on test sentences every few epochs
- Best model is automatically saved based on lowest loss

### How to Use Autocomplete

1. **Basic Usage**:
```bash
# Run autocomplete with the best trained model
python autocomplete.py

# You'll see:
# ============================================================
# WORD2VEC AUTOCOMPLETE
# ============================================================
# Type text and see predictions (needs 3 words for context)
# Type 'quit' or Ctrl+C to exit
# ------------------------------------------------------------
# 
# >>> 
```

2. **Using Autocomplete**:
```bash
# Type at least k words (default k=3) to see predictions:
>>> the united states
Next: of(0.45) | america(0.12) | and(0.08) | is(0.06) | in(0.05)

>>> machine learning is
Next: a(0.32) | the(0.18) | used(0.11) | based(0.09) | an(0.07)

# Keep typing to see predictions update:
>>> machine learning is a
Next: powerful(0.22) | technique(0.15) | form(0.12) | type(0.10) | method(0.08)
```

3. **Use Specific Checkpoint**:
```bash
# Use a specific model checkpoint
python autocomplete.py --checkpoint checkpoints/improved_epoch_10.pt

# The autocomplete will load that specific model
```

### Training Tips

- **First Run**: Dataset download takes ~1 minute, then preprocessing ~30 seconds
- **Training Speed**: ~5-10 minutes per epoch on Apple M1/M2 with MPS
- **Memory Usage**: ~2GB RAM for default settings
- **Better Predictions**: Train for at least 5 epochs with vocab_size=15000+
- **Debugging**: Use `--save_sequences` to inspect what the model learns from

### Checkpoints

Models are saved to `checkpoints/`:
- `improved_best.pt`: Best model based on loss (used by default for autocomplete)
- `improved_epoch_N.pt`: Checkpoint every 5 epochs

## Project Structure

```
w2v/
├── models/
│   └── model.py              # Model architecture
├── scripts/
│   ├── train.py              # Training script with data inspection
│   └── data_processor.py     # Data loading utilities
├── autocomplete.py           # Next-word prediction demo
├── explore_embeddings.py     # Word2Vec embedding explorer
└── README.md
```

### File Descriptions

#### `models/model.py`
Defines the `ImprovedKWordsToNext` neural network architecture:
- Position-aware embeddings for each word position
- 3-layer fully connected network with ReLU activations
- Layer normalization and dropout for regularization
- Methods for training (`forward`) and inference (`predict_next_word`, `most_likely_next_words`)

#### `scripts/train.py`
Main training script that:
- Loads and preprocesses the Text8 dataset
- Creates training sequences of k consecutive words → next word
- Implements training loop with Adam optimizer and cosine learning rate scheduling
- Saves checkpoints and tracks best model
- Evaluates model performance during training
- **Includes `--save_sequences` option to export training data for inspection**

#### `scripts/data_processor.py`
Handles data loading and preprocessing:
- Downloads Text8 dataset (100MB of Wikipedia text) if not present
- Builds vocabulary from most frequent words
- Implements subsampling of frequent words
- Creates word-to-index and index-to-word mappings
- Provides negative sampling distribution for original Word2Vec (not used in current model)

#### `autocomplete.py`
Interactive console application:
- Loads trained model from checkpoint
- Takes user text input
- Predicts next word based on last k words
- Displays top 5 predictions with probabilities
- Simple interface: type text, see predictions, type 'quit' to exit

#### `explore_embeddings.py`
Word2Vec embedding exploration tool:
- Extracts and analyzes learned word embeddings
- Find most similar words to any given word
- Calculate similarity scores between word pairs
- Solve word analogies (king - man + woman = queen)
- Perform vector arithmetic with word embeddings
- Interactive interface for exploring semantic relationships

## Example Usage

```python
from models.model import ImprovedKWordsToNext
import torch

# Load model
model = ImprovedKWordsToNext(vocab_size=10000, k=3)
model.load_state_dict(torch.load('checkpoints/improved_best.pt')['model_state_dict'])
model.eval()

# Predict next word
context = [word_to_idx['the'], word_to_idx['quick'], word_to_idx['brown']]
predictions = model.most_likely_next_words(context, top_n=5)
```

## Performance

- **Training**: ~10.4 million sequences from Text8
- **Speed**: ~5-10 min/epoch on Apple M1/M2 with MPS
- **Memory**: ~500MB for model + data

## License

MIT