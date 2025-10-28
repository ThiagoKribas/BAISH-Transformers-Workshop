import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralBigram(nn.Module):
    """
    Neural network version of bigram model.
    Same as count-based, but learns the probability table via gradient descent.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx):
        """
        Args:
            idx: (batch,) or (batch, 1) tensor of current token indices
        Returns:
            logits: (batch, vocab_size) predictions for next token
        """
        if idx.dim() == 2: 
            idx = idx.squeeze(-1)

        return self.embedding(idx)
    
    def generate(self, idx, max_new_tokens):
        """Generate text by sampling from the learned distribution."""
        for _ in range(max_new_tokens):
            current = idx[:, -1:]
            
            logits = self.forward(current)
            
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def get_batch(data, batch_size):
    """
    Sample random batches from data.
    Returns context (bigram uses 1 token) and targets.
    """
    
    ix = torch.randint(len(data) - 1, (batch_size,)) 
    
    x = data[ix]

    y = data[ix + 1]
    
    return x, y


def estimate_loss(model, data, batch_size, eval_iters=100):
    """Evaluate model loss over multiple batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size)
        
        logits = model.forward(X)
        
        loss = F.cross_entropy(logits, Y)
        
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    max_iters = 5000
    eval_interval = 500
    learning_rate = 4.0
    
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
    n = int(0.8 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {len(train_data)} tokens")
    
    model = NeuralBigram(vocab_size)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    
    # Training loop
    for iter in range(max_iters):

        xb, yb = get_batch(train_data, batch_size)
        
        logits = model.forward(xb)
        
        loss = F.cross_entropy(logits, yb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = estimate_loss(model, train_data, batch_size)
            val_loss = estimate_loss(model, val_data, batch_size)
            print(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f}")
    
    # Generate
    print("\nGenerated text:")
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, 500)[0].tolist()))