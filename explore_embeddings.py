#!/usr/bin/env python3
"""
Interactive tool to explore Word2Vec embeddings.
Test word similarity, analogies, and vector arithmetic.
"""

import os
import sys
import torch
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import ImprovedKWordsToNext
from scripts.train import ImprovedKWordsDataset


class EmbeddingExplorer:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path=None):
        """Load the trained model and extract embeddings."""
        print("Loading model...")
        
        # Find checkpoint
        if checkpoint_path is None:
            checkpoints = glob.glob('checkpoints/improved_*.pt')
            if not checkpoints:
                print("No checkpoints found! Train a model first:")
                print("  python scripts/train.py --k 3 --epochs 3")
                sys.exit(1)
            
            checkpoint_path = 'checkpoints/improved_best.pt' if 'checkpoints/improved_best.pt' in checkpoints else sorted(checkpoints)[-1]
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        args = checkpoint['args']
        
        # Create dataset for vocabulary
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
        
        # Extract embeddings (average of all position embeddings)
        self._extract_embeddings()
        
        print(f"Model loaded: {self.dataset.vocab_size} words, {args.embedding_dim}-dim embeddings")
        print("-" * 60)
    
    def _extract_embeddings(self):
        """Extract word embeddings from the model."""
        # Average embeddings across all positions for each word
        embeddings_list = []
        for pos_embedding in self.model.embeddings:
            embeddings_list.append(pos_embedding.weight.data)
        
        # Average across positions
        self.embeddings = torch.stack(embeddings_list).mean(dim=0)  # (vocab_size, embedding_dim)
        
        # Normalize for cosine similarity
        self.embeddings_norm = self.embeddings / self.embeddings.norm(dim=1, keepdim=True)
    
    def word_to_vec(self, word):
        """Get embedding vector for a word."""
        if word not in self.dataset.word_to_idx:
            return None
        idx = self.dataset.word_to_idx[word]
        return self.embeddings[idx]
    
    def most_similar(self, word, top_n=10):
        """Find most similar words."""
        if word not in self.dataset.word_to_idx:
            return []
        
        idx = self.dataset.word_to_idx[word]
        word_vec = self.embeddings_norm[idx]
        
        # Compute similarities
        similarities = torch.matmul(self.embeddings_norm, word_vec)
        
        # Get top words (excluding the word itself)
        values, indices = torch.topk(similarities, top_n + 1)
        
        results = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            if idx != self.dataset.word_to_idx[word]:
                results.append((self.dataset.idx_to_word[idx], val))
        
        return results[:top_n]
    
    def similarity(self, word1, word2):
        """Compute similarity between two words."""
        if word1 not in self.dataset.word_to_idx or word2 not in self.dataset.word_to_idx:
            return None
        
        idx1 = self.dataset.word_to_idx[word1]
        idx2 = self.dataset.word_to_idx[word2]
        
        vec1 = self.embeddings_norm[idx1]
        vec2 = self.embeddings_norm[idx2]
        
        return torch.dot(vec1, vec2).item()
    
    def analogy(self, word1, word2, word3, top_n=5):
        """
        Solve analogy: word1 is to word2 as word3 is to ?
        Example: king - man + woman = queen
        """
        if any(w not in self.dataset.word_to_idx for w in [word1, word2, word3]):
            return []
        
        # Get vectors
        vec1 = self.embeddings[self.dataset.word_to_idx[word1]]
        vec2 = self.embeddings[self.dataset.word_to_idx[word2]]
        vec3 = self.embeddings[self.dataset.word_to_idx[word3]]
        
        # Compute target: vec2 - vec1 + vec3
        target = vec2 - vec1 + vec3
        target_norm = target / target.norm()
        
        # Find closest words
        similarities = torch.matmul(self.embeddings_norm, target_norm)
        values, indices = torch.topk(similarities, top_n + 3)
        
        # Filter out input words
        input_words = {word1, word2, word3}
        results = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            word = self.dataset.idx_to_word[idx]
            if word not in input_words:
                results.append((word, val))
        
        return results[:top_n]
    
    def vector_arithmetic(self, positive_words, negative_words, top_n=5):
        """
        Perform vector arithmetic: sum(positive) - sum(negative)
        Example: ['paris', 'germany'] - ['france'] ≈ berlin
        """
        # Get positive vectors
        result = torch.zeros_like(self.embeddings[0])
        
        for word in positive_words:
            if word in self.dataset.word_to_idx:
                idx = self.dataset.word_to_idx[word]
                result += self.embeddings[idx]
        
        # Subtract negative vectors
        for word in negative_words:
            if word in self.dataset.word_to_idx:
                idx = self.dataset.word_to_idx[word]
                result -= self.embeddings[idx]
        
        # Normalize and find similar
        result_norm = result / result.norm()
        similarities = torch.matmul(self.embeddings_norm, result_norm)
        values, indices = torch.topk(similarities, top_n + len(positive_words) + len(negative_words))
        
        # Filter out input words
        input_words = set(positive_words + negative_words)
        results = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            word = self.dataset.idx_to_word[idx]
            if word not in input_words:
                results.append((word, val))
        
        return results[:top_n]
    
    def run_interactive(self):
        """Run interactive exploration mode."""
        print("\n" + "="*60)
        print("WORD2VEC EMBEDDING EXPLORER")
        print("="*60)
        print("\nCommands:")
        print("  similar <word>           - Find similar words")
        print("  similarity <w1> <w2>     - Compute similarity between two words")
        print("  analogy <w1> <w2> <w3>   - Solve: w1 is to w2 as w3 is to ?")
        print("  add <w1> <w2> ...        - Add word vectors")
        print("  subtract <w1> - <w2>     - Subtract word vectors")
        print("  quit                     - Exit")
        print("-"*60)
        
        while True:
            try:
                cmd = input("\n>>> ").strip().lower()
                
                if cmd == 'quit':
                    break
                
                elif cmd.startswith('similar '):
                    word = cmd[8:].strip()
                    results = self.most_similar(word)
                    if results:
                        print(f"\nMost similar to '{word}':")
                        for w, score in results:
                            print(f"  {w}: {score:.3f}")
                    else:
                        print(f"Word '{word}' not in vocabulary")
                
                elif cmd.startswith('similarity '):
                    parts = cmd[11:].split()
                    if len(parts) == 2:
                        sim = self.similarity(parts[0], parts[1])
                        if sim is not None:
                            print(f"\nSimilarity between '{parts[0]}' and '{parts[1]}': {sim:.3f}")
                        else:
                            print("One or both words not in vocabulary")
                    else:
                        print("Usage: similarity <word1> <word2>")
                
                elif cmd.startswith('analogy '):
                    parts = cmd[8:].split()
                    if len(parts) == 3:
                        results = self.analogy(parts[0], parts[1], parts[2])
                        if results:
                            print(f"\n{parts[0]} is to {parts[1]} as {parts[2]} is to:")
                            for w, score in results:
                                print(f"  {w}: {score:.3f}")
                        else:
                            print("One or more words not in vocabulary")
                    else:
                        print("Usage: analogy <word1> <word2> <word3>")
                
                elif cmd.startswith('add '):
                    words = cmd[4:].split()
                    results = self.vector_arithmetic(words, [])
                    if results:
                        print(f"\n{' + '.join(words)} ≈")
                        for w, score in results:
                            print(f"  {w}: {score:.3f}")
                
                elif cmd.startswith('subtract '):
                    parts = cmd[9:].split(' - ')
                    if len(parts) == 2:
                        positive = parts[0].split()
                        negative = parts[1].split()
                        results = self.vector_arithmetic(positive, negative)
                        if results:
                            print(f"\n{' + '.join(positive)} - {' - '.join(negative)} ≈")
                            for w, score in results:
                                print(f"  {w}: {score:.3f}")
                    else:
                        print("Usage: subtract <words> - <words>")
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except (KeyboardInterrupt, EOFError):
                break
        
        print("\nGoodbye!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Explore Word2Vec Embeddings')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint to load')
    parser.add_argument('--word', type=str, default=None,
                        help='Find similar words to this word')
    
    args = parser.parse_args()
    
    explorer = EmbeddingExplorer(checkpoint_path=args.checkpoint)
    
    if args.word:
        # Non-interactive mode: just show similar words
        results = explorer.most_similar(args.word)
        if results:
            print(f"\nMost similar to '{args.word}':")
            for word, score in results:
                print(f"  {word}: {score:.3f}")
        else:
            print(f"Word '{args.word}' not in vocabulary")
    else:
        # Interactive mode
        explorer.run_interactive()


if __name__ == '__main__':
    main()