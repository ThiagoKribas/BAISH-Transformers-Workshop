import torch
import torch.nn as nn
import torch.nn.functional as F

class ModeloPropio(nn.Module):
    """
    MLP-based language model with fixed context window and an rnn layer.
    Uses learned embeddings and hidden layers (unlike n-gram lookup tables).
    """
    def __init__(self, vocab_size, context_length, embed_dim, mlp_hidden_dim, rnn_hidden_size, mlp_hidden_layers=1):
        super().__init__()
        self.context_length = context_length
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # --- Path 1: MLP Layers ---
        self.mlp_layer1 = nn.Linear(context_length * embed_dim, mlp_hidden_dim)
        self.mlp_hidden_layers = nn.ModuleList([nn.Linear(mlp_hidden_dim, mlp_hidden_dim) for _ in range(mlp_hidden_layers)])
        
        # --- Path 2: RNN Layer ---
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=rnn_hidden_size, batch_first=True)
        
        # --- Combination Layer ---
        # The output layer must accept the concatenated outputs of both paths
        combined_dim = mlp_hidden_dim + rnn_hidden_size
        self.output_layer = nn.Linear(combined_dim, vocab_size)

    def forward(self, idx, rnn_hidden=None):
        """
        Args:
            idx: (batch, context_length) tensor of token indices
            rnn_hidden: Optional initial hidden state for the RNN
        Returns:
            logits: (batch, vocab_size)
            rnn_hidden: The final hidden state from the RNN path
        """
        embeddings = self.embedding(idx) # Shape: (batch, context_length, embed_dim)

        # --- Path 1: MLP Forward Pass ---
        flattened_embeddings = embeddings.view(idx.shape[0], -1) # Shape: (batch, context_length * embed_dim)
        mlp_output = F.relu(self.mlp_layer1(flattened_embeddings))
        for layer in self.mlp_hidden_layers:
            mlp_output = F.relu(layer(mlp_output))
        # Final mlp_output shape: (batch, mlp_hidden_dim)

        # --- Path 2: RNN Forward Pass ---
        rnn_full_output, rnn_hidden = self.rnn(embeddings, rnn_hidden)
        # We take the hidden state of the very last time-step
        rnn_last_output = rnn_full_output[:, -1, :] # Shape: (batch, rnn_hidden_size)
        
        # --- Combine the two paths ---
        combined_output = torch.cat([mlp_output, rnn_last_output], dim=1) # Shape: (batch, mlp_hidden_dim + rnn_hidden_size)
        
        # --- Final prediction ---
        logits = self.output_layer(combined_output)
        
        return logits, rnn_hidden
    
    def generate(self, idx, max_new_tokens):
        """
        Generates text by handling context padding and the RNN hidden state.
        """
        self.eval()
        rnn_hidden = None  # Initialize the hidden state for the RNN

        for _ in range(max_new_tokens):
            # --- FIX IS HERE ---
            
            # 1. Crop the context to the maximum required length.
            idx_cond = idx[:, -self.context_length:]
            
            # 2. Check if the context is shorter than the model requires.
            current_seq_len = idx_cond.shape[1]
            if current_seq_len < self.context_length:
                # If it is, create a tensor of padding zeros.
                padding = torch.zeros((idx_cond.shape[0], self.context_length - current_seq_len), 
                                    dtype=torch.long, device=idx.device)
                # Prepend the padding to the context.
                idx_cond = torch.cat([padding, idx_cond], dim=1)
            
            # Now, idx_cond is guaranteed to have shape (batch, context_length)
            
            # 3. Get the predictions from the model.
            # We pass the hidden state from the previous step to the RNN part.
            logits, rnn_hidden = self.forward(idx_cond, rnn_hidden)
            
            # 4. Sample the next token.
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 5. Append the new token to the sequence.
            idx = torch.cat((idx, idx_next), dim=1)
            
        self.train()
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

        logits, _ = model.forward(X)
        
        loss = F.cross_entropy(logits, Y)
        
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    max_iters = 50000
    eval_interval = 1000
    learning_rate = 1e-3
    
    # Model hyperparameters
    context_length = 12      # Look at previous characters original 8
    embed_dim = 24          # Size of token embeddings original 24
    hidden_dim = 64        # Size of hidden layers original 128
    hidden_layers = 3      # Amount of hidden layers original 1
    hidden_size = 64        # sIZE OFF HIDDEN RNN 
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
    print(f"Context length: {context_length}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Hidden layers: {hidden_layers}")
    
    model = ModeloPropio(vocab_size, context_length, embed_dim, hidden_dim, hidden_size, hidden_layers)
    
    
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {parameter_count}")

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    
    # Training loop
    print("\nTraining...")
    for iter in range(max_iters):
        xb, yb = get_batch(train_data, context_length, batch_size)
        
        logits, _ = model.forward(xb)  # (batch, vocab_size)

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