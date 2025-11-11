import torch
import torch.nn as nn
import torch.nn.functional as F

class ModeloPropio(nn.Module):
    """
    MLP-based language model with fixed context window and an rnn layer.
    Uses learned embeddings and hidden layers (unlike n-gram lookup tables).
    """
    def __init__(self, vocab_size, context_length, embed_dim, hidden_dim, hidden_layers=1):
        super().__init__()
        self.context_length = context_length
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Input size: context_length * embed_dim (flattened embeddings)
        # Output size: hidden_dim
        self.layer1 = nn.Linear(context_length * embed_dim, hidden_dim)

        # Input size: hidden_dim
        # Output size: hidden_dim
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
                
        # Input size: hidden_dim
        # Output size: vocab_size (prediction for each possible token)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx):
        """
        Args:
            idx: (batch, context_length) tensor of token indices
        Returns:
            logits: (batch, vocab_size)
        """
        batch_size = idx.shape[0]

        embeddings = self.embedding(idx)
        flattened_embeddings = embeddings.view(batch_size, -1)

        # Shape should be: (batch, context_length * embed_dim)
        hidden_pass = F.relu(self.layer1(flattened_embeddings))

        for layer in self.hidden_layers:
            hidden_pass = F.relu(layer(hidden_pass))
        
        # Return shape: (batch, vocab_size)
        return self.output_layer(hidden_pass)
    
    def generate(self, idx, max_new_tokens):
        """Generate text by sampling from the learned distribution."""
        for _ in range(max_new_tokens):
            # Crop context to last context_length tokens
            idx_cond = idx[:, -self.context_length:]
            
            # Pad if needed (for start of sequence)
            if idx_cond.shape[1] < self.context_length:
                padding = torch.zeros(idx.shape[0], self.context_length - idx_cond.shape[1], 
                                     dtype=torch.long)
                idx_cond = torch.cat([padding, idx_cond], dim=1)
            
            # Get predictions
            logits = self(idx_cond)
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def get_batch(data, context_length, batch_size):
    """
    Sample random batches from data.
    Returns context windows and targets.
    """
    ix = torch.randint(len(data) - context_length, (batch_size,))
    
    x = torch.stack([data[i:i+context_length] for i in ix])
    
    y = data[ix + context_length]
    
    return x, y


def estimate_loss(model, data, context_length, batch_size, eval_iters=100):
    """Evaluate model loss over multiple batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        X, Y = get_batch(data, context_length, batch_size)

        logits = model.forward(X)
        
        loss = F.cross_entropy(logits, Y)
        
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    max_iters = 5000
    eval_interval = 500
    learning_rate = 1e-3
    
    # Model hyperparameters
    context_length = 8      # Look at 8 previous characters original 8
    embed_dim = 24          # Size of token embeddings original 24
    hidden_dim = 128        # Size of hidden layers original 128
    hidden_layers = 2      # Amount of hidden layers original 1
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
    print(f"Hidden dim: {hidden_dim}")
    print(f"Hidden layers: {hidden_layers}")
    
    model = MLPLanguageModel(vocab_size, context_length, embed_dim, hidden_dim, hidden_layers)
    
    
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