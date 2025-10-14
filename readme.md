# AI Safety: Introduction to Transformer Architecture Workshop

**A hands-on workshop where you'll build language models from scratch to understand why transformers became the dominant architecture in modern AI.**

## What You'll Build

Starting from the simplest possible model, you'll implement increasingly sophisticated architectures, discovering their strengths and limitations through experimentation. By the end, you'll understand:

- What problems each architecture solves
- Why certain designs won over others
- How the attention mechanism works
- The path from simple n-grams to modern transformers

## Prerequisites

- Python 3.8+
- Basic neural network concepts
- Some PyTorch familiarity (we'll guide you!)

## Setup

```bash
# Clone this repository
git clone <repo-url>
cd transformer-workshop

# Install dependencies
pip install torch

# Dataset location
# Files should be in Dataset/ folder
```

## Workshop Structure

### Part 1: Statistical Baselines (30 minutes)
**Count-Based Models**
- Implement bigram and trigram models using frequency counting
- Understand probabilistic language modeling
- Establish baseline performance

📁 Files: `bigram_count.py` (template + solution)

### Part 2: Neural N-grams (45 minutes)
**From Counting to Learning**
- Implement the same models with gradient descent
- Compare neural vs. statistical approaches
- Learn about embedding layers and lookup tables

📁 Files: `bigram_neural.py`, `trigram.py` (templates + solutions)

### Part 3: Multi-Layer Perceptrons (1 hour)
**Adding Depth and Representation Learning**
- Build an MLP with learned embeddings and hidden layers
- Experiment with context length and model size
- Discover the parallel processing advantage
- Learn about regularization (weight decay, dropout)

📁 Files: `mlp.py` (template + solution)

**Key Question**: Why does this fixed-window approach work so well?

### Part 4: Recurrent Models (1 hour)
**Sequential Processing**
- Implement vanilla RNN
- Upgrade to LSTM with gated cells
- Compare sequential vs. parallel processing
- Experiment with context length

📁 Files: `rnn.py`, `lstm.py` (templates + solutions)

**Key Questions**: 
- Why don't RNNs beat MLPs in our setup?
- What problems do gates solve?
- When would you choose RNN over MLP?

### Part 5: Transformers (1.5 hours)
**The Modern Approach**
- Implement multi-head self-attention
- Build complete transformer blocks
- Add positional encodings
- Compare to all previous models

📁 Files: `transformer.py` (template + solution)

**Key Question**: How does attention solve the limitations of both MLPs and RNNs?

## Activities

### 🎯 Main Tasks

For each model:
1. **Read the template** - understand the structure
2. **Fill in TODOs** - implement key components
3. **Train the model** - run for ~5 minutes
4. **Observe results** - loss curves, generated text
5. **Compare** - how does it stack up against previous models?

### 🔬 Experiments to Try

- **Context length**: What happens with shorter/longer contexts?
- **Model size**: Try different hidden dimensions
- **Regularization**: Add/remove dropout, weight decay
- **Training time**: Does more training always help?
- **Data size**: Try with 1 book vs 2 books

### 💬 Discussion Points

Throughout the workshop, we'll discuss:
- What makes a good language model?
- Trade-offs between different architectures
- When would you choose each approach?
- How do these principles apply to modern LLMs?

## Code Structure

All models follow this pattern:

```python
class ModelName(nn.Module):
    def __init__(self, vocab_size, ...):
        # TODO: Define layers
        pass
    
    def forward(self, idx):
        # TODO: Implement forward pass
        return logits
    
    def generate(self, idx, max_new_tokens):
        # Autoregressive text generation
        return idx

# Training loop (mostly complete, you focus on the model!)
# Evaluation and visualization
```

**Your job**: Fill in the TODOs to make each model work!

## Expected Timeline

| Section | Time | Activity |
|---------|------|----------|
| Setup & Intro | 15 min | Installation, overview |
| Count-based | 30 min | Bigram/trigram statistics |
| Neural N-grams | 45 min | Gradient descent versions |
| **Break** | 15 min | ☕ |
| MLP | 60 min | Fixed-context neural model |
| RNN/LSTM | 60 min | Sequential processing |
| **Break** | 15 min | ☕ |
| Transformer | 90 min | Attention mechanism |
| Wrap-up | 30 min | Comparison, Q&A |

**Total: ~6 hours** (with breaks)

## Tips for Success

### 🐛 Debugging
- Check tensor shapes frequently
- Use small models first to debug faster
- Print intermediate outputs
- Compare with solution if stuck

### ⚡ Performance
- Start with small models (faster iteration)
- Use `steps_per_epoch=500` for quick experiments
- GPU not required but helps
- Watch your CPU temperature!

### 📊 Comparing Models
- Keep hyperparameters consistent across models
- Look at both train AND validation loss
- Generated text quality matters too
- Training time is a real constraint

### 🤔 When Stuck
1. Check the hints in code comments
2. Verify tensor shapes match expected dimensions
3. Look at the solution for guidance
4. Ask instructors!

## Common Questions

**Q: Do I need a GPU?**
A: No! All models train in minutes on CPU.

**Q: What if my model performs worse than expected?**
A: That's part of learning! We'll discuss why certain architectures struggle in certain scenarios.

**Q: Can I use different hyperparameters?**
A: Absolutely! Experimentation is encouraged.

**Q: How do I know if my implementation is correct?**
A: If it trains (loss goes down) and generates somewhat coherent text, you're on the right track!

## Competition (Optional)

Want to compete? Try to achieve the **lowest validation loss** by the end of the week!

**Rules:**
- Use the provided training data only
- Any architecture modifications allowed
- Document your approach

**Submit:**
- Your best model code
- Final train/val loss
- Brief explanation of what you tried

**Categories:**
- Overall best validation loss
- Most creative architecture
- Best efficiency (loss per parameter)

## Resources

### During Workshop
- Template files with TODOs and hints
- Solution files (try templates first!)
- Instructor support

### For Later Study
- Bengio et al. (2003) - Neural Language Models
- Vaswani et al. (2017) - "Attention Is All You Need"
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Karpathy's "Unreasonable Effectiveness of RNNs"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## What You'll Learn

By the end of this workshop, you'll be able to:

✅ Explain how language models work from first principles
✅ Implement key architectures from scratch
✅ Understand the attention mechanism intuitively
✅ Recognize trade-offs between different approaches
✅ Debug and experiment with neural architectures
✅ Connect historical developments to modern AI systems

## After the Workshop

- Complete any unfinished implementations
- Try the optional competition
- Experiment with different datasets
- Read the referenced papers
- Apply these concepts to your own projects

## Questions?

Reach out during the workshop or open an issue on GitHub!

---

**Let's build some transformers! 🚀**