"""
Lightweight vocabulary loader for fast inference without recreating the dataset.
"""

import os
import pickle
import torch
from models.model import ImprovedKWordsToNext


class VocabularyLoader:
    """Fast vocabulary loader that avoids recreating the entire dataset."""
    
    def __init__(self, vocab_path='checkpoints/k_words_vocab.pkl'):
        """Load vocabulary from the preprocessed file."""
        self.vocab_path = vocab_path
        self._load_vocabulary()
    
    def _load_vocabulary(self):
        """Load vocabulary data from pickle file."""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(
                f"Vocabulary file not found: {self.vocab_path}\n"
                f"Please run: python tmp/extract_vocab.py"
            )
        
        with open(self.vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        # Set vocabulary attributes
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.vocab_size = vocab_data['vocab_size']
        self.k = vocab_data['k']
        self.embedding_dim = vocab_data['embedding_dim']
        self.hidden_dim = vocab_data['hidden_dim']
        self.dropout = vocab_data['dropout']
    
    def load_model(self, checkpoint_path='checkpoints/k_words.pt', device=None):
        """Load the trained model with vocabulary."""
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Create model
        model = ImprovedKWordsToNext(
            vocab_size=self.vocab_size,
            k=self.k,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, device
    
    def words_to_indices(self, words):
        """Convert list of words to indices."""
        indices = []
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx.get('<UNK>', 0))
        return indices
    
    def indices_to_words(self, indices):
        """Convert list of indices to words."""
        words = []
        for idx in indices:
            if idx < len(self.idx_to_word):
                words.append(self.idx_to_word[idx])
            else:
                words.append('<UNK>')
        return words
    
    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size