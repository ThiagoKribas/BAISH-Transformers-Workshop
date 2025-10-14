import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNLanguageModel(nn.Module):
    """
    Vanilla RNN language model.
    Processes sequences recurrently with hidden state.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        
        # TODO: Create embedding layer
        # Hint: nn.Embedding(vocab_size, embed_dim)
        
        # TODO: Create RNN layer
        # Hint: nn.RNN(input_size, hidden_size, batch_first=True)
        # batch_first=True means input/output shapes are (batch, seq, feature)
        
        # TODO: Create output layer
        # Hint: nn.Linear(hidden_size, vocab_size)
    
    def forward(self, idx, hidden=None):
        """
        Args:
            idx: (batch, seq_len) tensor of token indices
            hidden: (1, batch, hidden_size) initial hidden state (optional)
        Returns:
            logits: (batch, vocab_size) prediction for next token
            hidden: (1, batch, hidden_size) final hidden state
        """
        # TODO: Get embeddings
        # Shape: (batch, seq_len, embed_dim)
        
        # TODO: Pass through RNN
        # Hint: output, hidden = self.rnn(embeddings, hidden)
        # output shape: (batch, seq_len, hidden_size)
        # hidden shape: (1, batch, hidden_size)
        
        # TODO: Take last timestep output
        # Hint: output[:, -1, :]
        # Shape: (batch, hidden_size)
        
        # TODO: Pass through output layer
        # Shape: (batch, vocab_size)
        
        pass
    
    def generate(self, idx, max_new_tokens):
        """Generate text by sampling from the learned distribution."""
        hidden = None  # Start with no hidden state
        
        for _ in range(max_new_tokens):
            # TODO: Get predictions (pass hidden state!)
            logits, hidden = None, None  # self(idx, hidden)
            
            # TODO: Convert to probabilities
            probs = None
            
            # TODO: Sample next token
            idx_next = None
            
            # TODO: Append to sequence
            idx = None
        
        return idx


# Same data loading functions as MLP
def get_batch(data, context_length, batch_size):
    """Sample random batches from data."""
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
        # TODO: Get predictions (no need to keep hidden state during eval)
        logits, _ = None, None  # model(X)
        
        # TODO: Calculate loss
        loss = None
        
        losses[k] = loss.item()
    
    model.train()
    return losses.mean()


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    steps_per_epoch = 1000
    eval_interval = 2
    learning_rate = 1e-3
    
    # Model hyperparameters
    context_length = 32        # RNNs can handle longer sequences
    embed_dim = 32
    hidden_size = 128          # RNN hidden state size
    
    # ... rest of training code same as MLP ...