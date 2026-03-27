# Direct Baseline Integration Guide

## Quick Start

### 1. Installation & Setup

The direct baseline is part of the hyperdiffusion package. No additional installation needed:

```python
from hyperdiffusion.direct_baseline import DirectPredictor, DirectSystem
```

### 2. Basic Usage

```python
import torch
from hyperdiffusion.direct_baseline import DirectPredictor

# Create model
model = DirectPredictor(x_dim=2, y_dim=1)

# Create sample data
support_x = torch.randn(batch_size=4, support_size=5, x_dim=2)
support_y = torch.randn(batch_size=4, support_size=5, y_dim=1)
query_x = torch.randn(batch_size=4, query_size=10, x_dim=2)

# Predict
predictions = model(support_x, support_y, query_x)
print(predictions.shape)  # torch.Size([4, 10, 1])
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Support Set (x, y)                                     │
│  Shape: (batch, support_size, x_dim/y_dim)              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│  AttentionSetEncoder                                    │
│  • Embeds support points                                │
│  • Pools with attention mechanism                       │
│  • Produces context vector                              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
         Context: (batch, cond_dim)
                 │
                 ├─────────────────────────────────────────┐
                 │                                         │
                 ▼                                         ▼
          [Expand to queries]              Query X: (batch, query_size, x_dim)
                 │                                         │
                 └──────────────────┬──────────────────────┘
                                    │
                                    ▼
                        [Concatenate: Context + Query X]
                        Shape: (batch, query_size, cond_dim + x_dim)
                                    │
                                    ▼
                        ┌─────────────────────────────────┐
                        │  Prediction Head (MLP)          │
                        │  • Hidden layer with SiLU       │
                        │  • Output layer                 │
                        └─────────────────────────────────┘
                                    │
                                    ▼
                        Predictions: (batch, query_size, y_dim)
```

## Component Details

### DirectPredictor

The core prediction model. Takes support examples and generates predictions for query points.

**Key features:**
- Single forward pass
- Deterministic predictions
- Efficient computation
- Stable gradient flow

**Configuration:**

```python
model = DirectPredictor(
    x_dim=2,              # Input feature dimension
    y_dim=1,              # Output dimension
    encoder_hidden=128,   # Width of encoder MLP
    cond_dim=64,          # Context vector dimension
    attention_heads=4,    # Number of attention heads
    attention_layers=3,   # Number of transformer layers
)
```

### DirectTextProjector

Projects external text embeddings (e.g., from BERT) to the context space.

**Usage:**

```python
text_projector = DirectTextProjector(text_dim=768, cond_dim=64)
text_embed = torch.randn(batch_size, 768)  # E.g., from BERT
context = text_projector(text_embed)       # (batch_size, 64)
```

### DirectSystem

Unified interface supporting multiple encoding modes and modalities.

**Supported modes:**
1. **Support set mode**: Encode from support examples
2. **Text mode**: Encode from text embeddings
3. **Oracle mode**: Encode from ground-truth family indicators
4. **Hybrid mode**: Mix multiple encodings

**API:**

```python
system = DirectSystem(
    x_dim=2,
    y_dim=1,
    encoder_hidden=128,
    cond_dim=64,
    attention_heads=4,
    attention_layers=3,
    text_dim=768,
    text_mix_alpha=0.5,  # Mixing weight for hybrid mode
)

# Encoding operations
context_support = system.encode_support(support_x, support_y)
context_text = system.encode_text(text_embed)
context_oracle = system.encode_oracle(one_hot_family)

# Prediction with different contexts
pred1 = system(support_x, support_y, query_x)  # Auto-encode support
pred2 = system(None, None, query_x, context=context_text)  # Use text context
pred3 = system(None, None, query_x, context=context_oracle)  # Use oracle context
```

## Training

### Standard Training Loop

```python
import torch.nn as nn
import torch.optim as optim

model = DirectPredictor(x_dim=2, y_dim=1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        support_x, support_y, query_x, query_y = batch
        
        # Forward
        pred = model(support_x, support_y, query_x)
        loss = criterion(pred, query_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

### Multi-Task Training

```python
# Generate multiple diverse tasks in each batch
for epoch in range(num_epochs):
    total_loss = 0
    
    for task_batch in task_generator(batch_size=32):
        support_x, support_y, query_x, query_y = task_batch
        
        # Forward pass on multiple tasks
        pred = model(support_x, support_y, query_x)
        loss = criterion(pred, query_y)
        
        # Aggregate loss over tasks
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss / num_batches}")
```

### Hybrid Training with Text

```python
system = DirectSystem(text_mix_alpha=0.5)
optimizer = optim.Adam(system.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for support_x, support_y, query_x, query_y, text_embed in data:
        # Loss from support set encoding
        pred_support = system(support_x, support_y, query_x)
        loss_support = criterion(pred_support, query_y)
        
        # Loss from text encoding
        context_text = system.encode_text(text_embed)
        pred_text = system(None, None, query_x, context=context_text)
        loss_text = criterion(pred_text, query_y)
        
        # Combined loss (weighted by text_mix_alpha)
        loss = (1 - system.text_mix_alpha) * loss_support + system.text_mix_alpha * loss_text
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Evaluation

### Basic Evaluation

```python
model.eval()

with torch.no_grad():
    for support_x, support_y, query_x, query_y in test_loader:
        predictions = model(support_x, support_y, query_x)
        
        # Compute metrics
        mse = torch.mean((predictions - query_y) ** 2)
        rmse = torch.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - query_y))
        
        print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
```

### Few-Shot Learning Evaluation

```python
def evaluate_few_shot(model, episodes, k_shot, k_query):
    """Evaluate on few-shot learning episodes."""
    model.eval()
    accuracies = []
    
    with torch.no_grad():
        for support, query in episodes:
            support_x, support_y = support
            query_x, query_y = query
            
            # Add batch dimension
            support_x = support_x.unsqueeze(0)
            support_y = support_y.unsqueeze(0)
            query_x = query_x.unsqueeze(0)
            
            # Predict
            pred = model(support_x, support_y, query_x)
            
            # Compute accuracy
            acc = (pred.argmax(-1) == query_y.argmax(-1)).float().mean()
            accuracies.append(acc.item())
    
    return {
        'mean_acc': sum(accuracies) / len(accuracies),
        'std_acc': np.std(accuracies),
    }
```

## Comparison with Full Hyperdiffusion

The direct baseline is intentionally simple to serve as a reference point.

| Feature | Direct | Full Hyperdiffusion |
|---------|--------|------------------|
| Context type | Fixed global | Adaptive/Per-query |
| Weight generation | None | Hypernetwork |
| Predictions | Deterministic | Can be stochastic |
| Training complexity | Simple | Complex |
| Inference speed | Fast | Slower |
| Max expressiveness | Limited | Very high |

**When to use direct baseline:**
- Fast prototyping
- Understanding model behavior
- Establishing performance baselines
- Limited computational resources
- Interpretability requirements

**When to use full hyperdiffusion:**
- Maximizing accuracy
- Complex task distributions
- Per-query adaptation needed
- Sufficient computational resources

## Integration with Existing Code

### Replacing a Baseline

```python
# Old: Simple MLP baseline
class MLPBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.mlp(x)

# New: Direct baseline (much better!)
from hyperdiffusion.direct_baseline import DirectPredictor

baseline = DirectPredictor(x_dim=2, y_dim=1)
```

### Combining with Hyperparameter Search

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    encoder_hidden = trial.suggest_int('encoder_hidden', 32, 256, step=32)
    cond_dim = trial.suggest_int('cond_dim', 16, 128, step=16)
    attention_heads = trial.suggest_int('attention_heads', 1, 8)
    attention_layers = trial.suggest_int('attention_layers', 1, 4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    # Create model
    model = DirectPredictor(
        x_dim=2, y_dim=1,
        encoder_hidden=encoder_hidden,
        cond_dim=cond_dim,
        attention_heads=attention_heads,
        attention_layers=attention_layers,
    )
    
    # Train and evaluate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            pred = model(*batch[:3])
            loss = criterion(pred, batch[3])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        val_loss = evaluate(model, val_loader)
        best_val = min(best_val, val_loss)
    
    return best_val

# Run hyperparameter search
study = optuna.create_study()
study.optimize(objective, n_trials=20)
```

## Optimization Tips

### 1. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. Learning Rate Scheduling

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
for epoch in range(num_epochs):
    train(model, train_loader)
    scheduler.step()
```

### 3. Batch Normalization (Optional)

The model includes SiLU activations which are generally stable. Batch norm may help with very large batches:

```python
class DirectPredictorWithBN(DirectPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add batch norm if needed
```

### 4. Regularization

```python
# L2 regularization
l2_loss = sum(p.pow(2).sum() for p in model.parameters()) * 0.0001

# Total loss
total_loss = mse_loss + l2_loss
```

## Testing & Validation

Run the comprehensive test suite:

```bash
# All tests
pytest tests/test_direct_baseline.py -v

# Specific test class
pytest tests/test_direct_baseline.py::TestDirectPredictor -v

# With coverage
pytest tests/test_direct_baseline.py --cov=hyperdiffusion.direct_baseline
```

## Common Issues & Solutions

### Issue: Training doesn't converge
**Solution:**
- Reduce learning rate
- Increase context dimension (`cond_dim`)
- Use gradient clipping
- Check data preprocessing

### Issue: High validation error
**Solution:**
- Increase model capacity (encoder_hidden, attention_layers)
- Use more support examples
- Check for data leakage
- Verify normalization

### Issue: Memory overflow
**Solution:**
- Reduce batch size
- Reduce context dimension
- Reduce encoder hidden dimension
- Use gradient accumulation

### Issue: NaN in losses
**Solution:**
- Check input value ranges
- Use gradient clipping
- Reduce learning rate
- Verify data contains no NaN

## Performance Monitoring

### Tensorboard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        pred = model(*batch[:3])
        loss = criterion(pred, batch[3])
        
        writer.add_scalar('loss/train', loss, epoch * len(train_loader) + step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

writer.close()
```

## Next Steps

1. **Try the example**: `python examples/direct_baseline_example.py`
2. **Run tests**: `pytest tests/test_direct_baseline.py -v`
3. **Read documentation**: See `docs/direct_baseline.md`
4. **Adapt to your task**: Modify input/output dimensions
5. **Compare with full model**: Benchmark against hyperdiffusion

## Support & Debugging

For detailed architecture information, see [direct_baseline.py](../hyperdiffusion/direct_baseline.py).

For full documentation, see [docs/direct_baseline.md](direct_baseline.md).
