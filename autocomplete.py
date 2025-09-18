#!/usr/bin/env python3
"""
Simple autocomplete console using the improved model.
Shows predictions above the cursor as you type.
"""

import os
import sys
import torch
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import ImprovedKWordsToNext
from scripts.train import ImprovedKWordsDataset


class SimpleAutocomplete:
    def __init__(self, checkpoint_path=None, k=3):
        self.k = k
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path=None):
        """Load the trained model from checkpoint."""
        print("Loading model...")
        
        # Find latest checkpoint if not specified
        if checkpoint_path is None:
            # Look for improved model checkpoints first
            checkpoints = glob.glob('checkpoints/improved_*.pt')
            if not checkpoints:
                # Fallback to any checkpoint
                checkpoints = glob.glob('checkpoints/*.pt')
            
            if not checkpoints:
                print("No checkpoints found! Please train the model first:")
                print("  python scripts/train.py --k 3 --epochs 1")
                sys.exit(1)
            
            # Use best model if available, otherwise latest
            if 'checkpoints/improved_best.pt' in checkpoints:
                checkpoint_path = 'checkpoints/improved_best.pt'
            else:
                checkpoint_path = sorted(checkpoints)[-1]
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        args = checkpoint['args']
        self.k = args.k
        
        # Create dataset to get vocabulary
        print("Loading vocabulary...")
        self.dataset = ImprovedKWordsDataset(
            vocab_size=args.vocab_size,
            min_count=args.min_count,
            k=args.k
        )
        
        # Create and load model
        self.model = ImprovedKWordsToNext(
            vocab_size=self.dataset.vocab_size,
            k=args.k,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded (Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']:.4f})")
        print("-" * 60)
    
    def _get_predictions(self, text):
        """Get next word predictions based on the last k words."""
        words = text.lower().split()
        
        if len(words) < self.k:
            return None
        
        # Get last k words
        context_words = words[-self.k:]
        
        # Convert to indices
        context_indices = []
        for word in context_words:
            if word in self.dataset.word_to_idx:
                context_indices.append(self.dataset.word_to_idx[word])
            else:
                context_indices.append(self.dataset.word_to_idx.get('<UNK>', 0))
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model.most_likely_next_words(context_indices, top_n=10)
        
        # Format predictions
        word_predictions = []
        for idx, prob in predictions:
            if idx < len(self.dataset.idx_to_word):
                word = self.dataset.idx_to_word[idx]
                if word not in ['<PAD>', '<UNK>']:
                    word_predictions.append(f"{word}({prob:.2f})")
        
        return word_predictions[:5] if word_predictions else None
    
    def run(self):
        """Run the interactive console."""
        print("\n" + "="*60)
        print("WORD2VEC AUTOCOMPLETE")
        print("="*60)
        print(f"Type text and see predictions (needs {self.k} words for context)")
        print("Type 'quit' or Ctrl+C to exit")
        print("-"*60)
        print()
        
        try:
            while True:
                text = input(">>> ")
                
                if text.lower() == 'quit':
                    break
                    
                if not text.strip():
                    continue
                
                # Get and display predictions
                predictions = self._get_predictions(text)
                
                if predictions:
                    print(f"Next: {' | '.join(predictions)}")
                else:
                    print(f"Next: (need {self.k} words for context)")
                print()
                
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Word2Vec Autocomplete')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (uses best/latest if not specified)')
    
    args = parser.parse_args()
    
    console = SimpleAutocomplete(checkpoint_path=args.checkpoint)
    console.run()


if __name__ == '__main__':
    main()