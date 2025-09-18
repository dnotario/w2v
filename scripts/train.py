#!/usr/bin/env python3
"""
Train the improved K-Words-to-Next model with better data handling.
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import ImprovedKWordsToNext
from scripts.data_processor import Text8Dataset


class ImprovedKWordsDataset(Dataset):
    """Improved dataset that creates many more training sequences."""
    
    def __init__(self, text_path='data/text8', vocab_size=10000, min_count=5, 
                 k=3, subsample_threshold=1e-3):
        # Use base dataset for preprocessing
        self.base_dataset = Text8Dataset(
            text_path=text_path,
            vocab_size=vocab_size,
            min_count=min_count,
            window_size=1,
            num_negative_samples=0,
            subsample_threshold=subsample_threshold
        )
        
        self.vocab_size = len(self.base_dataset.word_to_idx)
        self.word_to_idx = self.base_dataset.word_to_idx
        self.idx_to_word = self.base_dataset.idx_to_word
        self.k = k
        
        # Get the ACTUAL word indices (not positions)
        self.indices = []
        for pos in self.base_dataset.subsampled_indices:
            self.indices.append(self.base_dataset.word_indices[pos])
        self.num_sequences = len(self.indices) - k
        
        print(f"Dataset ready: {self.num_sequences:,} training sequences from {len(self.indices):,} words")
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        # Get k words and next word directly from indices
        k_words = torch.tensor(self.indices[idx:idx+self.k], dtype=torch.long)
        next_word = torch.tensor(self.indices[idx+self.k], dtype=torch.long)
        
        return k_words, next_word


def collate_fn(batch):
    """Simple collate function."""
    k_words = torch.stack([item[0] for item in batch])
    next_words = torch.stack([item[1] for item in batch])
    return k_words, next_words


def train_epoch(model, dataloader, optimizer, device, epoch, log_interval=100):
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (k_words, next_words) in enumerate(progress_bar):
        k_words = k_words.to(device)
        next_words = next_words.to(device)
        
        optimizer.zero_grad()
        loss = model(k_words, next_words)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches


def evaluate_model(model, dataset, device):
    model.eval()
    
    test_sequences = [
        ['the', 'quick', 'brown'],
        ['in', 'the', 'world'],
        ['one', 'of', 'the'],
        ['it', 'is', 'a'],
        ['this', 'is', 'the']
    ]
    
    print("\n" + "="*50)
    print("Next Word Predictions:")
    print("="*50)
    
    with torch.no_grad():
        for sequence in test_sequences:
            indices = []
            for word in sequence:
                if word in dataset.word_to_idx:
                    indices.append(dataset.word_to_idx[word])
                else:
                    indices.append(dataset.word_to_idx.get('<UNK>', 0))
            
            if len(indices) == model.k:
                predictions = model.most_likely_next_words(indices, top_n=5)
                
                print(f"\n{' '.join(sequence)} →")
                for idx, prob in predictions:
                    if idx < len(dataset.idx_to_word):
                        next_word = dataset.idx_to_word[idx]
                        print(f"  {next_word}: {prob:.3f}")


def save_sequences(dataset, output_path='sequences.txt', max_sequences=1000):
    """Save training sequences to a file for inspection."""
    print(f"\nSaving sequences to {output_path}...")
    
    sequences_to_save = min(max_sequences, len(dataset))
    
    with open(output_path, 'w') as f:
        f.write(f"# Training Sequences (k={dataset.k})\n")
        f.write(f"# Vocabulary size: {dataset.vocab_size}\n")
        f.write(f"# Total sequences: {len(dataset):,}\n")
        f.write(f"# Showing first {sequences_to_save:,} sequences\n")
        f.write("#" + "="*60 + "\n\n")
        
        for i in tqdm(range(sequences_to_save), desc="Writing sequences"):
            k_word_indices = dataset.indices[i:i+dataset.k]
            next_word_index = dataset.indices[i+dataset.k]
            
            k_words = [dataset.idx_to_word[idx] for idx in k_word_indices]
            next_word = dataset.idx_to_word[next_word_index]
            
            context_str = ' '.join(k_words)
            f.write(f"Seq {i+1:6d}: [{context_str:40s}] → {next_word}\n")
    
    print(f"Saved {sequences_to_save:,} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Improved K-Words Model')
    
    # Model parameters
    parser.add_argument('--k', type=int, default=3,
                        help='Number of context words')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Data parameters
    parser.add_argument('--vocab_size', type=int, default=20000,
                        help='Maximum vocabulary size')
    parser.add_argument('--min_count', type=int, default=5,
                        help='Minimum word frequency')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Debug options
    parser.add_argument('--save_sequences', action='store_true',
                        help='Save training sequences to file for inspection')
    parser.add_argument('--sequences_file', type=str, default='sequences.txt',
                        help='Output file for sequences')
    parser.add_argument('--max_sequences', type=int, default=1000,
                        help='Maximum sequences to save')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using device: CUDA")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
    
    # Dataset
    print("\nLoading dataset...")
    dataset = ImprovedKWordsDataset(
        vocab_size=args.vocab_size,
        min_count=args.min_count,
        k=args.k
    )
    
    print(f"Vocabulary: {dataset.vocab_size} words")
    print(f"Training sequences: {len(dataset):,}")
    
    # Save sequences if requested
    if args.save_sequences:
        save_sequences(dataset, args.sequences_file, args.max_sequences)
        print()
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type in ['cuda', 'mps'])
    )
    
    # Model
    model = ImprovedKWordsToNext(
        vocab_size=dataset.vocab_size,
        k=args.k,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Training
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("\nStarting training...")
    print(f"K={args.k}, Embedding={args.embedding_dim}, Hidden={args.hidden_dim}")
    print(f"Batch size={args.batch_size}, LR={args.learning_rate}")
    print("-" * 50)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args
            }, os.path.join(args.checkpoint_dir, 'k_words.pt'))
        
        # Evaluate
        if epoch % max(1, args.epochs // 5) == 0:
            evaluate_model(model, dataset, device)
        
        # Regular checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args
            }, os.path.join(args.checkpoint_dir, f'k_words_epoch_{epoch}.pt'))
    
    print("\nFinal evaluation:")
    evaluate_model(model, dataset, device)
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()