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
        
        # Create query, key, value projections
        self.q_projection = nn.Linear(embed_dim, embed_dim)
        self.k_projection = nn.Linear(embed_dim, embed_dim)
        self.v_projection = nn.Linear(embed_dim, embed_dim)
        
        # Create output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        B, T, C = x.shape
        
        # Compute Q, K, V
        # Shape: (batch, seq_len, embed_dim)
        Q = self.q_projection(x)
        K = self.k_projection(x)
        V = self.v_projection(x)


        # Split into multiple heads
        # Reshape to (batch, num_heads, seq_len, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        
        # Apply causal mask (prevent looking at future tokens)
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        attention = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = attention @ V
        
        # TODO: Concatenate heads and project
        # Reshape back to (batch, seq_len, embed_dim)
        output = output.transpose(1,2).contiguous()

        output = output.view(B, T, C)

        return self.output_projection(output)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        # Create two linear layers with ReLU in between
        self.linear_layer = nn.Linear(embed_dim, ff_dim)

        self.output_layer = nn.Linear(ff_dim, embed_dim)
        # Hint: embed_dim -> ff_dim -> embed_dim
        # Add dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Implement forward pass
        # x -> linear -> relu -> dropout -> linear -> dropout
        linear_pass = self.dropout(F.relu(self.linear_layer(x)))
        return self.dropout(self.output_layer(linear_pass))


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Create attention layer
        self.attention_layer = MultiHeadAttention(embed_dim, num_heads)
        # Create feedforward layer
        self.feedforward_layer = FeedForward(embed_dim, ff_dim, dropout)

        # Create two layer norms (one before attention, one before ff)
        self.layerNorm_attention = nn.LayerNorm(embed_dim)
        
        self.layerNorm_feedforward = nn.LayerNorm(embed_dim)
        
        # Create dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #Attention with residual connection
        x = x + self.dropout(self.attention_layer(self.layerNorm_attention(x)))
        
        #Feedforward with residual connection  
        x = x + self.dropout(self.feedforward_layer(self.layerNorm_feedforward(x)))
        
        return x


class TransformerLanguageModel(nn.Module):
    """Transformer-based language model"""
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Create positional embedding
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Create transformer blocks
        self.transformer_layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        
        # Create final layer norm
        self.layernorm = nn.LayerNorm(embed_dim)

        # Create output head
        self.output = nn.Linear(embed_dim, vocab_size)
        
        # Create dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self, idx):
        """
        Args:
            idx: (batch, seq_len)
        Returns:
            logits: (batch, vocab_size)
        """
        B, T = idx.shape
        
        # Get token embeddings
        embeddings = self.embedding(idx)
        x = embeddings.view(B, -1)
        
        # TODO: Get positional embeddings
        # Hint: torch.arange(T) for positions
        positions = torch.arange(T)
        pos_embed = self.positional_embedding(positions)
        # TODO: Add token + positional embeddings
        # TODO: Apply dropout
        x = self.dropout(embeddings + pos_embed)
        

        # TODO: Pass through transformer blocks
        for layer in self.transformer_layers:
            x = layer(x)
        # TODO: Apply final layer norm
        x = self.layernorm(x)
        # TODO: Take last token and project to vocabulary
        # Shape: (batch, vocab_size)
        x = x[:, -1, :]
        return self.output(x)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Generate text autoregressively"""
        for _ in range(max_new_tokens):
            # TODO: Crop to max sequence length if needed
            idx_cond = idx if idx.size(1) <= self.positional_embedding.num_embeddings else idx[:, -self.positional_embedding.num_embeddings:]
            # TODO: Get predictions
            logits, _ = self.forward(idx_cond)
            # TODO: Sample and append
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
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
    eval_interval = 100
    learning_rate = 3e-4
    max_iters = num_epochs * steps_per_epoch
    
    # Model hyperparameters
    context_length = 16        # Transformers handle longer sequences well
    embed_dim = 64             # Embedding dimension
    num_heads = 4              # Number of attention heads
    num_layers = 2             # Number of transformer blocks
    ff_dim = embed_dim * 2     # Feedforward dimension (typically 4x embed_dim)
    dropout = 0.1
    
    # Load and encode data
    with open('Dataset/Anne_of_Green_Gables.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {len(train_data)} tokens")
    print(f"Context length: {context_length}")
    print(f"Embedding dim: {embed_dim}")
    
    model = TransformerLanguageModel(vocab_size, embed_dim, num_heads, num_layers, ff_dim, context_length)
    
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {parameter_count}")

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    
    # Training loop
    print("\nTraining...")
    for iter in range(max_iters):
        xb, yb = get_batch(train_data, context_length, batch_size)
        
        logits = model.forward(xb)  # (batch, vocab_size)
        
        loss = F.cross_entropy(logits, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = estimate_loss(model, train_data, context_length, batch_size)
            val_loss = estimate_loss(model, val_data, context_length, batch_size)
            print(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f}")
    
    # Generate
    print("\nGenerated text:")
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, 500)[0].tolist()))