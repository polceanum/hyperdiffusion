"""
Example: Using DirectPredictor for Function Regression

This example demonstrates how to use the DirectPredictor for meta-learning on
synthetic function regression tasks. The model learns to quickly adapt to new
functions from just a few examples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

from hyperweights.direct_baseline import DirectPredictor


class SinusoidDataset(IterableDataset):
    """Generates synthetic sinusoid tasks for meta-learning."""

    def __init__(self, task_size: int = 100, support_size: int = 5, query_size: int = 10):
        self.task_size = task_size
        self.support_size = support_size
        self.query_size = query_size

    def __iter__(self):
        for _ in range(self.task_size):
            # Sample random amplitude and phase for this task
            amplitude = torch.rand(1).item() * 0.5 + 0.5  # [0.5, 1.0]
            phase = torch.rand(1).item() * 2 * 3.14159  # [0, 2π]
            
            # Generate support points
            support_x = torch.rand(self.support_size, 1) * 2 * 3.14159
            support_y = amplitude * torch.sin(support_x + phase)
            
            # Generate query points
            query_x = torch.rand(self.query_size, 1) * 2 * 3.14159
            query_y = amplitude * torch.sin(query_x + phase)
            
            yield support_x, support_y, query_x, query_y


class MetaLearningTrainer:
    """Trains a meta-learning model on sinusoid regression."""

    def __init__(self, model: DirectPredictor, learning_rate: float = 1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def train_epoch(self, dataloader, num_tasks: int = 100) -> float:
        """Train for one epoch on multiple tasks."""
        self.model.train()
        total_loss = 0.0
        
        for i, (support_x, support_y, query_x, query_y) in enumerate(dataloader):
            if i >= num_tasks:
                break
            
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Forward pass
            predictions = self.model(support_x, support_y, query_x)
            loss = self.criterion(predictions, query_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"  Task {i+1}/{num_tasks}, Loss: {loss.item():.6f}")
        
        return total_loss / num_tasks

    @torch.no_grad()
    def evaluate(self, dataloader, num_tasks: int = 100) -> float:
        """Evaluate on multiple tasks."""
        self.model.eval()
        total_loss = 0.0
        
        for i, (support_x, support_y, query_x, query_y) in enumerate(dataloader):
            if i >= num_tasks:
                break
            
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            predictions = self.model(support_x, support_y, query_x)
            loss = self.criterion(predictions, query_y)
            total_loss += loss.item()
        
        return total_loss / num_tasks


def main():
    """Main training loop."""
    print("=" * 60)
    print("DirectPredictor Meta-Learning Example")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating model...")
    model = DirectPredictor(
        x_dim=1,
        y_dim=1,
        encoder_hidden=64,
        cond_dim=32,
        attention_heads=2,
        attention_layers=2,
    )
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    print("\n2. Creating datasets...")
    train_dataset = SinusoidDataset(task_size=1000, support_size=5, query_size=10)
    val_dataset = SinusoidDataset(task_size=100, support_size=5, query_size=10)
    
    train_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=1)
    print("   Train and validation datasets created")
    
    # Create trainer
    print("\n3. Starting training...")
    trainer = MetaLearningTrainer(model, learning_rate=1e-3)
    
    # Training loop
    num_epochs = 5
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, num_tasks=100)
        print(f"  Average train loss: {train_loss:.6f}")
        
        # Validate
        val_loss = trainer.evaluate(val_loader, num_tasks=50)
        print(f"  Average val loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "direct_baseline_best.pt")
            print("  → Saved best model")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("=" * 60)
    
    # Demonstrate inference
    print("\n4. Demonstrating inference on new task...")
    model.eval()
    
    with torch.no_grad():
        # Create a new task with amplitude=0.7, phase=1.0
        support_x = torch.tensor([[0.5], [1.5], [2.5], [3.5], [4.5]])
        support_y = 0.7 * torch.sin(support_x + 1.0)
        
        query_x = torch.tensor([[0.2], [1.2], [2.2], [3.2], [4.2]])
        query_y_true = 0.7 * torch.sin(query_x + 1.0)
        
        # Add batch dimension
        support_x = support_x.unsqueeze(0)
        support_y = support_y.unsqueeze(0)
        query_x = query_x.unsqueeze(0)
        
        # Predict
        predictions = model(support_x, support_y, query_x)
        predictions = predictions.squeeze()
        query_y_true = query_y_true.squeeze()
        
        # Compute error
        rmse = torch.sqrt(torch.mean((predictions - query_y_true) ** 2))
        
        print(f"\n   Support set:")
        for x, y in zip(support_x.squeeze(), support_y.squeeze()):
            print(f"     x={x:.3f}, y={y:.3f}")
        
        print(f"\n   Query predictions vs ground truth:")
        for i, (x, pred, true) in enumerate(zip(query_x.squeeze(), predictions, query_y_true)):
            print(f"     x={x:.3f}: pred={pred:.3f}, true={true:.3f}")
        
        print(f"\n   RMSE: {rmse:.6f}")


class EvaluationExample:
    """Example showing how to evaluate direct baseline on a custom dataset."""
    
    @staticmethod
    def evaluate_on_custom_data(
        model: DirectPredictor,
        test_tasks: list,
        support_size: int = 5,
    ) -> dict:
        """
        Evaluate model on custom test tasks.
        
        Args:
            model: DirectPredictor model
            test_tasks: List of (support_x, support_y, query_x, query_y) tuples
            support_size: Number of support examples (for filtering)
        
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        all_losses = []
        all_rmses = []
        
        with torch.no_grad():
            for support_x, support_y, query_x, query_y in test_tasks:
                # Ensure correct sizes
                if support_x.shape[1] != support_size:
                    indices = torch.randperm(support_x.shape[1])[:support_size]
                    support_x = support_x[:, indices]
                    support_y = support_y[:, indices]
                
                # Forward pass
                pred = model(support_x, support_y, query_x)
                
                # Compute metrics
                mse_loss = torch.mean((pred - query_y) ** 2)
                rmse = torch.sqrt(mse_loss)
                
                all_losses.append(mse_loss.item())
                all_rmses.append(rmse.item())
        
        return {
            "mean_mse": sum(all_losses) / len(all_losses),
            "mean_rmse": sum(all_rmses) / len(all_rmses),
            "std_rmse": torch.std(torch.tensor(all_rmses)).item(),
        }


if __name__ == "__main__":
    main()
