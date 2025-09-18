import os
import re
import zipfile
import requests
import numpy as np
from collections import Counter
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


class Text8Dataset(Dataset):
    def __init__(self, text_path=None, vocab_size=10000, min_count=5, window_size=4, 
                 num_negative_samples=10, subsample_threshold=1e-3):
        
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples
        self.subsample_threshold = subsample_threshold
        
        # Download and prepare text if needed
        if text_path is None:
            text_path = self._download_text8()
        
        # Read and preprocess text
        self.text = self._read_text(text_path)
        self.words = self._preprocess_text(self.text)
        
        # Build vocabulary
        self.word_to_idx, self.idx_to_word, self.word_counts = self._build_vocabulary(
            self.words, vocab_size, min_count
        )
        
        # Convert words to indices
        self.word_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) 
                             for word in self.words]
        
        # Setup negative sampling distribution
        self.negative_sampling_dist = self._create_negative_sampling_distribution()
        
        # Subsample frequent words
        self.subsampled_indices = self._subsample_frequent_words()
        
        print(f"Dataset initialized:")
        print(f"  Vocabulary size: {len(self.word_to_idx)}")
        print(f"  Total words: {len(self.words)}")
        print(f"  After subsampling: {len(self.subsampled_indices)}")
        
    def _download_text8(self):
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        text8_path = os.path.join(data_dir, 'text8')
        if os.path.exists(text8_path):
            print("Text8 dataset already exists")
            return text8_path
        
        print("Downloading Text8 dataset...")
        url = 'http://mattmahoney.net/dc/text8.zip'
        zip_path = os.path.join(data_dir, 'text8.zip')
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        os.remove(zip_path)
        return text8_path
    
    def _read_text(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    def _preprocess_text(self, text):
        # Convert to lowercase and split into words
        text = text.lower()
        text = re.sub(r'[^a-z ]', ' ', text)
        words = text.split()
        return words
    
    def _build_vocabulary(self, words, vocab_size, min_count):
        word_counts = Counter(words)
        
        # Filter by minimum count
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= min_count}
        
        # Keep most common words
        most_common = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        most_common = most_common[:vocab_size-2]  # Reserve space for special tokens
        
        # Build mappings
        word_to_idx = {'<UNK>': 0, '<PAD>': 1}
        idx_to_word = {0: '<UNK>', 1: '<PAD>'}
        
        for idx, (word, count) in enumerate(most_common, start=2):
            word_to_idx[word] = idx
            idx_to_word[idx] = word
        
        # Update counts to include only vocabulary words
        final_counts = {word: count for word, count in most_common}
        final_counts['<UNK>'] = sum(count for word, count in word_counts.items() 
                                    if word not in word_to_idx)
        final_counts['<PAD>'] = 0
        
        return word_to_idx, idx_to_word, final_counts
    
    def _create_negative_sampling_distribution(self):
        vocab_size = len(self.word_to_idx)
        word_freqs = np.zeros(vocab_size)
        
        total_count = sum(self.word_counts.values())
        
        for word, idx in self.word_to_idx.items():
            if word in self.word_counts:
                word_freqs[idx] = self.word_counts[word] / total_count
        
        # Use frequency^0.75 as suggested in the original paper
        word_freqs = np.power(word_freqs, 0.75)
        word_freqs = word_freqs / word_freqs.sum()
        
        return word_freqs
    
    def _subsample_frequent_words(self):
        if self.subsample_threshold <= 0:
            return list(range(len(self.word_indices)))
        
        total_count = sum(self.word_counts.values())
        subsampled = []
        
        for idx, word_idx in enumerate(self.word_indices):
            word = self.idx_to_word[word_idx]
            if word in self.word_counts:
                freq = self.word_counts[word] / total_count
                # Subsampling probability from the paper
                prob = 1 - np.sqrt(self.subsample_threshold / freq)
                if np.random.random() > prob:
                    subsampled.append(idx)
        
        return subsampled
    
    def sample_negatives(self, exclude_idx, num_samples):
        negatives = []
        while len(negatives) < num_samples:
            neg_idx = np.random.choice(len(self.word_to_idx), p=self.negative_sampling_dist)
            if neg_idx != exclude_idx and neg_idx != 1:  # Exclude PAD token
                negatives.append(neg_idx)
        return negatives
    
    def __len__(self):
        return len(self.subsampled_indices)
    
    def __getitem__(self, idx):
        # Get actual word index from subsampled indices
        word_pos = self.subsampled_indices[idx]
        center_word = self.word_indices[word_pos]
        
        # Get context words for each position
        half_window = self.window_size // 2
        context_words = []
        negative_samples = []
        
        for i in range(1, half_window + 1):
            # Before center word (positions 0 to half_window-1)
            if word_pos - i >= 0:
                context_words.append(self.word_indices[word_pos - i])
            else:
                context_words.append(1)  # PAD token
            
            # After center word (positions half_window to window_size-1)
            if word_pos + i < len(self.word_indices):
                context_words.append(self.word_indices[word_pos + i])
            else:
                context_words.append(1)  # PAD token
        
        # Sample negatives for each position
        for context_word in context_words:
            negs = self.sample_negatives(context_word, self.num_negative_samples)
            negative_samples.append(negs)
        
        return {
            'center_word': torch.LongTensor([center_word]),
            'context_words': torch.LongTensor(context_words),
            'negative_samples': torch.LongTensor(negative_samples)
        }


def create_dataloader(text_path=None, batch_size=128, vocab_size=10000, 
                      window_size=4, num_negative_samples=10, num_workers=4):
    
    dataset = Text8Dataset(
        text_path=text_path,
        vocab_size=vocab_size,
        window_size=window_size,
        num_negative_samples=num_negative_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader, dataset


def collate_fn(batch):
    center_words = torch.cat([item['center_word'] for item in batch])
    context_words = torch.stack([item['context_words'] for item in batch])
    negative_samples = torch.stack([item['negative_samples'] for item in batch])
    
    return {
        'center_word': center_words,
        'context_words': context_words,
        'negative_samples': negative_samples
    }