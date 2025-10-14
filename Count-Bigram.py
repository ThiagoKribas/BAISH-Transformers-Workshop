import torch
import torch.nn.functional as F

class CountBasedBigram:
    """
    Count-based bigram model using PyTorch tensors.
    Builds a frequency table, no gradient descent needed!
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.counts = torch.ones([vocab_size, vocab_size])
    
    def train(self, data):
        """
        Count bigram frequencies in the data.
        Args:
            data: torch.LongTensor of token indices
        """
        
        tuple_tensor = torch.tensor(list(zip(data[:-1], data[1:])))
        print(self.counts)
        #self.counts[tuple_tensor[:, 0], tuple_tensor[:, 1]] += 1
        for (current, next) in tuple_tensor:
            self.counts[current, next] += 1
        print(self.counts)
    def get_probabilities(self):
        """
        Convert counts to probabilities.
        Returns: (vocab_size, vocab_size) tensor where [i, j] = P(j | i)
        """
        
        amount_of_current = self.counts.sum(dim=1, keepdim=True)
        return self.counts / amount_of_current

    def generate(self, idx, max_new_tokens):
        """
        Generate tokens by sampling from the count-based distribution.
        Args:
            idx: (batch, time) starting context
            max_new_tokens: number of tokens to generate
        """
        probs = self.get_probabilities()
        
        for _ in range(max_new_tokens):
            current_token = idx[:, -1]

            next_probs = probs[current_token]
            
            idx_next = torch.multinomial(next_probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def calculate_loss(self, data):
        """
        Calculate cross-entropy loss (negative log-likelihood).
        Same metric as neural models!
        """
        probs = self.get_probabilities()
        current_tokens = data[:-1]
        next_tokens = data[1:]
        
        token_probs = probs[current_tokens, next_tokens]
        
        return torch.mean(-torch.log(token_probs))

        


if __name__ == "__main__":
    
    
    # Load and encode data
    print("Loading data")
    with open('Dataset/Anne_of_Green_Gables.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    #Tokenize data
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {char: n for n, char in enumerate(chars)}
    itos = {n: char for n, char in enumerate(chars)}
    data = torch.tensor([stoi[char] for char in text if char in stoi])
    
    # Split data
    n = int(data.size(dim=0) * 0.7)
    print(n)
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Vocab size: {vocab_size}")
    print(f"Training on {len(train_data)} tokens")
    
    model = CountBasedBigram(vocab_size)
    
    print("\nCounting bigrams...")
    model.train(train_data)
    print("Done!")
    
    
    train_loss = model.calculate_loss(train_data[:10000])
    val_loss = model.calculate_loss(val_data[:10000])
    
    print(f"\nTrain loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")
    

    # Generate
    print("\nGenerated text:")
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, 500)
    print(''.join([itos[i] for i in generated[0].tolist()]))

