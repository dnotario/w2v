#!/usr/bin/env python3
"""
Interactive autocomplete console using the improved model.
Shows predictions ahead of cursor, Tab to cycle, Enter to select, Esc to cancel.
"""

import os
import sys
import torch
import glob
import termios
import tty
import select

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import ImprovedKWordsToNext
from scripts.train import ImprovedKWordsDataset


class AutocompleteConsole:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self._load_model(checkpoint_path)
        
        # Console state
        self.text = ""
        self.cursor_pos = 0
        self.suggestions = []
        self.selected_index = 0
        self.showing_suggestions = False
        
    def _load_model(self, checkpoint_path=None):
        """Load the trained model from checkpoint."""
        print("Loading model...")
        
        # Find latest checkpoint if not specified
        if checkpoint_path is None:
            checkpoints = glob.glob('checkpoints/improved_*.pt')
            if not checkpoints:
                checkpoints = glob.glob('checkpoints/*.pt')
            
            if not checkpoints:
                print("No checkpoints found! Please train the model first:")
                print("  python scripts/train.py --k 3 --epochs 1")
                sys.exit(1)
            
            checkpoint_path = 'checkpoints/improved_best.pt' if 'checkpoints/improved_best.pt' in checkpoints else sorted(checkpoints)[-1]
        
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
    
    def _get_predictions(self):
        """Get next word predictions based on the last k words before cursor."""
        # Get text up to cursor
        text_before = self.text[:self.cursor_pos]
        words = text_before.lower().split()
        
        if len(words) < self.k:
            return []
        
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
        
        # Filter and return words only
        word_predictions = []
        for idx, prob in predictions:
            if idx < len(self.dataset.idx_to_word):
                word = self.dataset.idx_to_word[idx]
                if word not in ['<PAD>', '<UNK>']:
                    word_predictions.append(word)
        
        return word_predictions[:5]
    
    def _display(self):
        """Display the current text with suggestions ahead of cursor."""
        # Clear line
        sys.stdout.write('\r' + ' ' * 120 + '\r')
        
        # Display text with cursor
        before = self.text[:self.cursor_pos]
        after = self.text[self.cursor_pos:]
        
        sys.stdout.write(before)
        
        # Show suggestions if available
        if self.showing_suggestions and self.suggestions:
            suggestion_str = ' ['
            for i, word in enumerate(self.suggestions):
                if i == self.selected_index:
                    suggestion_str += f' ▶{word} '
                else:
                    suggestion_str += f' {word} '
            suggestion_str += ']'
            
            # Show in gray color
            sys.stdout.write('\033[90m' + suggestion_str + '\033[0m')
        
        sys.stdout.write(after)
        
        # Move cursor back to position
        if len(after) > 0:
            sys.stdout.write('\033[' + str(len(after)) + 'D')
        
        sys.stdout.flush()
    
    def _handle_key(self, key):
        """Handle keyboard input."""
        if key == '\t':  # Tab - show/cycle suggestions
            if not self.showing_suggestions:
                self.suggestions = self._get_predictions()
                if self.suggestions:
                    self.showing_suggestions = True
                    self.selected_index = 0
            else:
                # Cycle through suggestions
                self.selected_index = (self.selected_index + 1) % len(self.suggestions)
        
        elif key == '\r' or key == '\n':  # Enter - accept suggestion or newline
            if self.showing_suggestions and self.suggestions:
                # Insert selected suggestion
                selected_word = self.suggestions[self.selected_index]
                
                # Add space before if needed
                if self.cursor_pos > 0 and self.text[self.cursor_pos-1] != ' ':
                    selected_word = ' ' + selected_word
                
                # Insert at cursor
                self.text = self.text[:self.cursor_pos] + selected_word + ' ' + self.text[self.cursor_pos:]
                self.cursor_pos += len(selected_word) + 1
                
                # Hide suggestions
                self.showing_suggestions = False
                self.suggestions = []
                self.selected_index = 0
            else:
                # Print current line and start new one
                print(self.text)
                self.text = ""
                self.cursor_pos = 0
                self.showing_suggestions = False
        
        elif key == '\x1b':  # Escape - cancel suggestions
            self.showing_suggestions = False
            self.selected_index = 0
        
        elif key == '\x7f':  # Backspace
            if self.cursor_pos > 0:
                self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                self.cursor_pos -= 1
                self.showing_suggestions = False
        
        elif key == '\x1b[D':  # Left arrow
            if self.cursor_pos > 0:
                self.cursor_pos -= 1
                self.showing_suggestions = False
        
        elif key == '\x1b[C':  # Right arrow
            if self.cursor_pos < len(self.text):
                self.cursor_pos += 1
                self.showing_suggestions = False
        
        elif key == '\x03':  # Ctrl+C
            return False
        
        elif len(key) == 1 and key.isprintable():
            # Regular character
            self.text = self.text[:self.cursor_pos] + key + self.text[self.cursor_pos:]
            self.cursor_pos += 1
            
            # Auto-show suggestions after space
            if key == ' ':
                self.suggestions = self._get_predictions()
                if self.suggestions:
                    self.showing_suggestions = True
                    self.selected_index = 0
            else:
                self.showing_suggestions = False
        
        return True
    
    def run(self):
        """Run the interactive console."""
        print("\n" + "="*60)
        print("WORD2VEC AUTOCOMPLETE")
        print("="*60)
        print(f"Commands:")
        print(f"  Type {self.k}+ words then SPACE to see suggestions")
        print(f"  TAB     - Show/cycle through suggestions")
        print(f"  ENTER   - Accept selected suggestion")
        print(f"  ESC     - Cancel suggestions")
        print(f"  Ctrl+C  - Exit")
        print("-"*60)
        print("\nStart typing:\n")
        
        # Setup terminal
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setraw(sys.stdin.fileno())
            
            while True:
                self._display()
                
                # Read key
                key = sys.stdin.read(1)
                
                # Handle escape sequences
                if key == '\x1b':
                    # Check if more characters are available
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key += sys.stdin.read(2)
                
                if not self._handle_key(key):
                    break
        
        finally:
            # Restore terminal
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("\n\nGoodbye!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Word2Vec Autocomplete')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (uses best/latest if not specified)')
    
    args = parser.parse_args()
    
    console = AutocompleteConsole(checkpoint_path=args.checkpoint)
    console.run()


if __name__ == '__main__':
    main()