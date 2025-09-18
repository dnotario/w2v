import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedKWordsToNext(nn.Module):
    """
    Improved Word2Vec variant with better architecture for k-words-to-next prediction.
    
    Features:
    - Separate embeddings for each position
    - Dropout for regularization
    - Layer normalization
    - Multiple hidden layers
    """
    
    def __init__(self, vocab_size, k=3, embedding_dim=300, hidden_dim=512, dropout=0.2):
        super(ImprovedKWordsToNext, self).__init__()
        
        self.vocab_size = vocab_size
        self.k = k
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Separate embedding for each position (position-aware)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) 
            for _ in range(k)
        ])
        
        # Combine embeddings
        self.combine_layer = nn.Linear(k * embedding_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Hidden layers
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer_norm3 = nn.LayerNorm(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim // 2, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        
        for module in [self.combine_layer, self.hidden1, self.hidden2, self.output_layer]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, k_words_indices, target_next_word):
        """
        Forward pass using word indices directly.
        
        Args:
            k_words_indices: Word indices for k words (batch_size, k)
            target_next_word: Target next word indices (batch_size,)
        
        Returns:
            loss: Cross-entropy loss
        """
        batch_size = k_words_indices.size(0)
        
        # Get embeddings for each position
        embeddings_list = []
        for i in range(self.k):
            word_indices = k_words_indices[:, i]
            emb = self.embeddings[i](word_indices)  # (batch_size, embedding_dim)
            embeddings_list.append(emb)
        
        # Concatenate embeddings
        combined = torch.cat(embeddings_list, dim=1)  # (batch_size, k * embedding_dim)
        
        # Process through layers
        x = self.combine_layer(combined)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.hidden1(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.hidden2(x)
        x = self.layer_norm3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output
        logits = self.output_layer(x)  # (batch_size, vocab_size)
        
        # Calculate loss
        loss = F.cross_entropy(logits, target_next_word)
        
        return loss
    
    def predict_next_word(self, k_words_indices):
        """
        Predict the next word given k words.
        
        Args:
            k_words_indices: Indices of k words (list or tensor of length k)
        
        Returns:
            probs: Probability distribution over vocabulary
        """
        device = next(self.parameters()).device
        
        # Convert to tensor if needed
        if not isinstance(k_words_indices, torch.Tensor):
            k_words_indices = torch.tensor(k_words_indices, dtype=torch.long)
        
        # Reshape to batch format
        if k_words_indices.dim() == 1:
            k_words_indices = k_words_indices.unsqueeze(0)
        
        k_words_indices = k_words_indices.to(device)
        
        # Get embeddings
        embeddings_list = []
        for i in range(self.k):
            word_indices = k_words_indices[:, i]
            emb = self.embeddings[i](word_indices)
            embeddings_list.append(emb)
        
        combined = torch.cat(embeddings_list, dim=1)
        
        # Forward pass (no dropout during inference)
        x = F.relu(self.layer_norm1(self.combine_layer(combined)))
        x = F.relu(self.layer_norm2(self.hidden1(x)))
        x = F.relu(self.layer_norm3(self.hidden2(x)))
        
        logits = self.output_layer(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        
        return probs
    
    def most_likely_next_words(self, k_words_indices, top_n=10):
        """
        Get the most likely next words.
        
        Args:
            k_words_indices: Indices of k words
            top_n: Number of top predictions to return
        
        Returns:
            List of (word_idx, probability) tuples
        """
        probs = self.predict_next_word(k_words_indices)
        
        # Get top n predictions
        top_probs, top_indices = torch.topk(probs, min(top_n, len(probs)))
        
        result = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            result.append((idx, prob))
        
        return result