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
from vocabulary_loader import VocabularyLoader


class AutocompleteConsole:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self._load_model(checkpoint_path)
        
        # Console state
        self.text = ""
        self.cursor_pos = 0
        self.suggestions = []
        self.selected_index = 0
        self.focus_on_suggestions = False  # True = focus on suggestions, False = focus on text
        
    def _load_model(self, checkpoint_path=None):
        """Load the trained model from checkpoint."""
        print("Loading model...")
        
        # Find latest checkpoint if not specified
        if checkpoint_path is None:
            if os.path.exists('checkpoints/k_words.pt'):
                checkpoint_path = 'checkpoints/k_words.pt'
            else:
                checkpoints = glob.glob('checkpoints/*.pt')
                if not checkpoints:
                    print("No checkpoints found! Please train the model first:")
                    print("  python scripts/train.py --k 3 --epochs 1")
                    sys.exit(1)
                checkpoint_path = sorted(checkpoints)[-1]
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load vocabulary (fast!)
        print("Loading vocabulary...")
        self.vocab_loader = VocabularyLoader()
        self.k = self.vocab_loader.k
        
        # Load model
        self.model, self.device = self.vocab_loader.load_model(checkpoint_path, self.device)
        
        # Load checkpoint to get training info
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
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
        context_indices = self.vocab_loader.words_to_indices(context_words)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model.most_likely_next_words(context_indices, top_n=10)
        
        # Filter and return words with probabilities, sorted by probability
        word_predictions = []
        for idx, prob in predictions:
            if idx < len(self.vocab_loader.idx_to_word):
                word = self.vocab_loader.idx_to_word[idx]
                if word not in ['<PAD>', '<UNK>']:
                    word_predictions.append((word, prob))
        
        # Sort by probability (highest first) and return top 5
        word_predictions.sort(key=lambda x: x[1], reverse=True)
        return [word for word, prob in word_predictions[:5]]
    
    def _display(self):
        """Display the current text with suggestions ahead of cursor."""
        # Clear line
        sys.stdout.write('\r' + ' ' * 120 + '\r')
        
        # Display text with cursor
        before = self.text[:self.cursor_pos]
        after = self.text[self.cursor_pos:]
        
        sys.stdout.write(before)
        
        # Always show suggestions if available
        if self.suggestions:
            sys.stdout.write(' [')
            for i, word in enumerate(self.suggestions):
                if i > 0:
                    sys.stdout.write(' ')
                
                # Highlight focused suggestion with different color
                if i == self.selected_index and self.focus_on_suggestions:
                    # White/bright for focused suggestion
                    sys.stdout.write('\033[97m' + word + '\033[0m')
                else:
                    # Gray for unfocused suggestions
                    sys.stdout.write('\033[90m' + word + '\033[0m')
            sys.stdout.write(']')
        
        sys.stdout.write(after)
        
        # Move cursor back to position
        if len(after) > 0:
            sys.stdout.write('\033[' + str(len(after)) + 'D')
        
        sys.stdout.flush()
    
    def _handle_key(self, key):
        """Handle keyboard input."""
        if key == '\t':  # Tab - cycle through suggestions
            if self.suggestions:
                if not self.focus_on_suggestions:
                    # First Tab - focus on suggestions
                    self.focus_on_suggestions = True
                    self.selected_index = 0
                else:
                    # Subsequent Tabs - cycle through
                    self.selected_index = (self.selected_index + 1) % len(self.suggestions)
                    if self.selected_index == 0:
                        # Cycled back to start, return to text
                        self.focus_on_suggestions = False
        
        elif key == '\r' or key == '\n':  # Enter - accept suggestion or newline
            if self.focus_on_suggestions and self.suggestions:
                # Insert selected suggestion
                selected_word = self.suggestions[self.selected_index]
                
                # Add space before if needed
                if self.cursor_pos > 0 and self.text[self.cursor_pos-1] != ' ':
                    selected_word = ' ' + selected_word
                
                # Insert at cursor
                self.text = self.text[:self.cursor_pos] + selected_word + ' ' + self.text[self.cursor_pos:]
                self.cursor_pos += len(selected_word) + 1
                
                # Update suggestions and return focus to text
                self.focus_on_suggestions = False
                self.suggestions = self._get_predictions()
                self.selected_index = 0
            else:
                # Print current line and start new one
                print(self.text)
                self.text = ""
                self.cursor_pos = 0
                self.focus_on_suggestions = False
                self.suggestions = []
        
        elif key == '\x1b':  # Escape - return focus to text
            self.focus_on_suggestions = False
            self.selected_index = 0
        
        elif key == '\x7f':  # Backspace
            if self.focus_on_suggestions:
                # If focused on suggestions, return to text
                self.focus_on_suggestions = False
            elif self.cursor_pos > 0:
                self.text = self.text[:self.cursor_pos-1] + self.text[self.cursor_pos:]
                self.cursor_pos -= 1
                self.suggestions = self._get_predictions()
        
        elif key == '\x1b[D':  # Left arrow
            if self.focus_on_suggestions:
                # Cycle through suggestions backward
                if self.suggestions:
                    self.selected_index = (self.selected_index - 1) % len(self.suggestions)
            elif self.cursor_pos > 0:
                self.cursor_pos -= 1
                self.suggestions = self._get_predictions()
        
        elif key == '\x1b[C':  # Right arrow
            if self.focus_on_suggestions:
                # Cycle through suggestions forward
                if self.suggestions:
                    self.selected_index = (self.selected_index + 1) % len(self.suggestions)
            elif self.cursor_pos < len(self.text):
                self.cursor_pos += 1
                self.suggestions = self._get_predictions()
        
        elif key == '\x03':  # Ctrl+C
            return False
        
        elif len(key) == 1 and key.isprintable():
            if self.focus_on_suggestions:
                # If focused on suggestions, return to text input
                self.focus_on_suggestions = False
            
            # Regular character
            self.text = self.text[:self.cursor_pos] + key + self.text[self.cursor_pos:]
            self.cursor_pos += 1
            
            # Always update suggestions after typing
            self.suggestions = self._get_predictions()
            self.selected_index = 0
        
        return True
    
    def run(self):
        """Run the interactive console."""
        print("\n" + "="*60)
        print("WORD2VEC AUTOCOMPLETE")
        print("="*60)
        print(f"Commands:")
        print(f"  Type {self.k}+ words to see suggestions (always shown)")
        print(f"  TAB     - Cycle through suggestions")
        print(f"  ARROWS  - Navigate suggestions when focused")
        print(f"  ENTER   - Accept highlighted suggestion")
        print(f"  ESC     - Return focus to text")
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