import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # TODO: Create query, key, value projections
        # Hint: nn.Linear(embed_dim, embed_dim)
        
        # TODO: Create output projection
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        B, T, C = x.shape
        
        # TODO: Compute Q, K, V
        # Shape: (batch, seq_len, embed_dim)
        
        # TODO: Split into multiple heads
        # Reshape to (batch, num_heads, seq_len, head_dim)
        
        # TODO: Compute attention scores
        # scores = Q @ K^T / sqrt(head_dim)
        
        # TODO: Apply causal mask (prevent looking at future tokens)
        # Hint: Use torch.triu to create upper triangular mask
        
        # TODO: Apply softmax
        
        # TODO: Apply attention to values
        # output = attention @ V
        
        # TODO: Concatenate heads and project
        # Reshape back to (batch, seq_len, embed_dim)
        
        pass


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        # TODO: Create two linear layers with ReLU in between
        # Hint: embed_dim -> ff_dim -> embed_dim
        # TODO: Add dropout
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        # x -> linear -> relu -> dropout -> linear -> dropout
        pass


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # TODO: Create attention layer
        # TODO: Create feedforward layer
        # TODO: Create two layer norms (one before attention, one before ff)
        # Hint: nn.LayerNorm(embed_dim)
        # TODO: Create dropout
        pass
    
    def forward(self, x):
        # TODO: Attention with residual connection
        # x = x + dropout(attention(layernorm(x)))
        
        # TODO: Feedforward with residual connection  
        # x = x + dropout(feedforward(layernorm(x)))
        
        pass


class TransformerLanguageModel(nn.Module):
    """Transformer-based language model"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # TODO: Create token embedding
        # Hint: nn.Embedding(vocab_size, embed_dim)
        
        # TODO: Create positional embedding
        # Hint: nn.Embedding(max_seq_len, embed_dim)
        
        # TODO: Create transformer blocks
        # Hint: nn.ModuleList([TransformerBlock(...) for _ in range(num_layers)])
        
        # TODO: Create final layer norm
        
        # TODO: Create output head
        # Hint: nn.Linear(embed_dim, vocab_size)
        
        # TODO: Create dropout
        
    def forward(self, idx):
        """
        Args:
            idx: (batch, seq_len)
        Returns:
            logits: (batch, vocab_size)
        """
        B, T = idx.shape
        
        # TODO: Get token embeddings
        
        # TODO: Get positional embeddings
        # Hint: torch.arange(T) for positions
        
        # TODO: Add token + positional embeddings
        
        # TODO: Apply dropout
        
        # TODO: Pass through transformer blocks
        
        # TODO: Apply final layer norm
        
        # TODO: Take last token and project to vocabulary
        # Shape: (batch, vocab_size)
        
        pass
    
    def generate(self, idx, max_new_tokens):
        """Generate text autoregressively"""
        for _ in range(max_new_tokens):
            # TODO: Crop to max sequence length if needed
            # TODO: Get predictions
            # TODO: Sample and append
            pass
        return idx


# Same data loading as before
def get_batch(data, context_length, batch_size):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = data[ix + context_length]
    return x, y


def estimate_loss(model, data, context_length, batch_size, eval_iters=100):
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        X, Y = get_batch(data, context_length, batch_size)
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    steps_per_epoch = 1000
    eval_interval = 2
    learning_rate = 3e-4
    
    # Model hyperparameters
    context_length = 64        # Transformers handle longer sequences well
    embed_dim = 64             # Embedding dimension
    num_heads = 4              # Number of attention heads
    num_layers = 2             # Number of transformer blocks
    ff_dim = embed_dim * 4     # Feedforward dimension (typically 4x embed_dim)
    dropout = 0.1
    
    # ... rest same as other models ...