# -*- coding: utf-8 -*-
"""
Milestone Project: Simple Adding Machine Neural Network

This script implements a feedforward neural network that learns to add two integers
between -10 and 10. The network takes two numbers as input and outputs their sum.

This is a regression problem, not classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()  # Turn off interactive mode

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

"""Data Generation"""

# Generate dataset
N = 1200  #  dataset size for better training

# Create input pairs: two random integers between -10 and 10
entrada = torch.randint(-10, 11, (N, 2)).float()
# Create targets: sum of the two inputs
saida = torch.sum(entrada, axis=1, keepdim=True)


"""**Conjunto de treinamento e teste usando DataLoader**"""

# Dividir o conjunto de dados
train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    entrada, saida, test_size=0.2, random_state=42
)

print(f"Training set size: {len(train_inputs)}")
print(f"Test set size: {len(test_inputs)}")

# Create PyTorch datasets
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)

# Create data loaders
batch_size = 16  # Increased batch size for more stable training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))  # Use full test set for evaluation

"""Neural Network Model Definition"""

class AddingMachine(nn.Module):
    """
    Simple feedforward neural network for learning addition.

    Args:
        n_units: Number of units in hidden layers
        n_layers: Number of hidden layers
    """
    def __init__(self, n_units=16, n_layers=2):
        super().__init__()

        self.n_layers = n_layers

        # Input layer: 2 inputs -> n_units
        self.input_layer = nn.Linear(2, n_units)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_units, n_units) for _ in range(n_layers)
        ])

        # Output layer: n_units -> 1 (the sum)
        self.output_layer = nn.Linear(n_units, 1)

    def forward(self, x):
        # Input layer with ReLU activation
        x = F.relu(self.input_layer(x))

        # Hidden layers with ReLU activation
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        # Output layer (no activation for regression)
        x = self.output_layer(x)
        return x

# a function to create the model with default values
def create_model(n_units=16, n_layers=2, learning_rate=0.01):
    """
    Create and return model, loss function, and optimizer.

    Args:
        n_units: Number of units in hidden layers
        n_layers: Number of hidden layers
        learning_rate: Learning rate for optimizer

    Returns:
        model, loss_function, optimizer
    """
    model = AddingMachine(n_units, n_layers)

    # Mean Square Error loss is appropriate for regression
    loss_function = nn.MSELoss()

    # Adam optimizer often works better than SGD
    # optimizer = torch.optim.SGD(net.parameters(),lr=.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_function, optimizer

"""Model Testing and Verification"""

# Test model creation
n_units_per_layer = 16
n_layers = 2
# use the create_model to create an instance
model, loss_fn, optimizer = create_model(n_units_per_layer, n_layers)

print("Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# Test forward pass
test_input = torch.tensor([[5.0, 3.0], [-2.0, 7.0]])
test_output = model(test_input)
print(f"\nTest forward pass:")
print(f"Input: {test_input}")
print(f"Output: {round(test_output)}")
print(f"Expected: {torch.sum(test_input, dim=1, keepdim=True)}")

"""Training Function"""

def train_model(n_units=16, n_layers=2, num_epochs=60, learning_rate=0.01, verbose=True):
    """
    Train the adding machine neural network.

    Args:
        n_units: Number of units in hidden layers
        n_layers: Number of hidden layers
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        verbose: Whether to print training progress

    Returns:
        train_losses, test_losses, train_maes, test_maes, trained_model
    """

    # Create model
    model, loss_fn, optimizer = create_model(n_units, n_layers, learning_rate)

    # Initialize tracking lists
    train_losses = []
    test_losses = []
    train_maes = []  # Mean Absolute Error
    test_maes = []

    # Training loop
    for epoch in range(num_epochs):

        # Training phase
        model.train()
        batch_losses = []
        batch_maes = []

        for X, y in train_loader:
            # Forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            batch_losses.append(loss.item())
            mae = torch.mean(torch.abs(y_pred - y))
            batch_maes.append(mae.item())

        # Average training metrics for this epoch
        avg_train_loss = np.mean(batch_losses)
        avg_train_mae = np.mean(batch_maes)
        train_losses.append(avg_train_loss)
        train_maes.append(avg_train_mae)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            X_test, y_test = next(iter(test_loader))
            y_pred_test = model(X_test)
            test_loss = loss_fn(y_pred_test, y_test).item()
            test_mae = torch.mean(torch.abs(y_pred_test - y_test)).item()

            test_losses.append(test_loss)
            test_maes.append(test_mae)

        # Print progress
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
            print(f"  Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    return train_losses, test_losses, train_maes, test_maes, model

"""Model Evaluation and Testing"""

def test_model(model, num_tests=10):
    """Test the trained model with specific examples."""
    model.eval()
    print("Testing trained model:")
    print("Input (a, b) -> Predicted Sum | Actual Sum | Error")
    print("-" * 50)

    total_error = 0
    with torch.no_grad():
        for _ in range(num_tests):
            # Generate random test inputs
            a, b = np.random.randint(-10, 11, 2)
            test_input = torch.tensor([[float(a), float(b)]])
            predicted = model(test_input).item()
            actual = a + b
            error = abs(predicted - actual)
            total_error += error

            print(f"({a:3d}, {b:3d}) -> {predicted:8.2f} | {actual:8d} | {error:.2f}")

    avg_error = total_error / num_tests
    print(f"\nAverage absolute error: {avg_error:.3f}")
    return avg_error

"""Run Training and Evaluation"""

print("Training the Adding Machine Neural Network...")
print("=" * 60)

# Train a single model
train_losses, test_losses, train_maes, test_maes, trained_model = train_model(
    n_units=16,
    n_layers=2,
    num_epochs=150,
    learning_rate=0.01,
    verbose=True
)

print("\n" + "=" * 60)
print("Training completed!")

# Test the trained model
print("\n" + "=" * 60)
test_model(trained_model, num_tests=15)

# Plot training progress
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Loss plots
epochs = range(1, len(train_losses) + 1)
ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
ax1.plot(epochs, test_losses, label='Test Loss', color='red')
ax1.set_title('Training and Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.legend()
ax1.grid(True)

# MAE plots
ax2.plot(epochs, train_maes, label='Train MAE', color='blue')
ax2.plot(epochs, test_maes, label='Test MAE', color='red')
ax2.set_title('Training and Test Mean Absolute Error')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE')
ax2.legend()
ax2.grid(True)

# Final performance comparison
final_metrics = ['Train Loss', 'Test Loss', 'Train MAE', 'Test MAE']
final_values = [train_losses[-1], test_losses[-1], train_maes[-1], test_maes[-1]]
colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']

ax3.bar(final_metrics, final_values, color=colors)
ax3.set_title('Final Model Performance')
ax3.set_ylabel('Error Value')
ax3.tick_params(axis='x', rotation=45)

# Test on a grid of values
test_range = range(-5, 6)
predictions_grid = []
actuals_grid = []

trained_model.eval()
with torch.no_grad():
    for a in test_range:
        for b in test_range:
            test_input = torch.tensor([[float(a), float(b)]])
            pred = trained_model(test_input).item()
            actual = a + b
            predictions_grid.append(pred)
            actuals_grid.append(actual)

ax4.scatter(actuals_grid, predictions_grid, alpha=0.6)
ax4.plot([-10, 10], [-10, 10], 'r--', label='Perfect Prediction')
ax4.set_xlabel('Actual Sum')
ax4.set_ylabel('Predicted Sum')
ax4.set_title('Predicted vs Actual (Grid Test)')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Training results saved to 'training_results.png'")

print(f"\nFinal Results:")
print(f"Final Train Loss: {train_losses[-1]:.4f}")
print(f"Final Test Loss: {test_losses[-1]:.4f}")
print(f"Final Train MAE: {train_maes[-1]:.4f}")
print(f"Final Test MAE: {test_maes[-1]:.4f}")

"""Multiple Runs for Reproducibility Analysis"""

def run_multiple_experiments(num_runs=5, n_units=16, n_layers=2, num_epochs=100):
    """
    Run multiple training experiments to analyze reproducibility and performance consistency.

    Args:
        num_runs: Number of independent training runs
        n_units: Number of units in hidden layers
        n_layers: Number of hidden layers
        num_epochs: Number of training epochs per run

    Returns:
        Dictionary with aggregated results
    """
    print(f"\nRunning {num_runs} independent experiments...")
    print("=" * 60)

    final_train_losses = []
    final_test_losses = []
    final_train_maes = []
    final_test_maes = []

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        # Train model (with different random initialization each time)
        train_losses, test_losses, train_maes, test_maes, model = train_model(
            n_units=n_units,
            n_layers=n_layers,
            num_epochs=num_epochs,
            learning_rate=0.01,
            verbose=False
        )

        # Store final metrics
        final_train_losses.append(train_losses[-1])
        final_test_losses.append(test_losses[-1])
        final_train_maes.append(train_maes[-1])
        final_test_maes.append(test_maes[-1])

        print(f"  Final Test MAE: {test_maes[-1]:.4f}")

    # Calculate statistics
    results = {
        'train_loss': {
            'mean': np.mean(final_train_losses),
            'std': np.std(final_train_losses),
            'values': final_train_losses
        },
        'test_loss': {
            'mean': np.mean(final_test_losses),
            'std': np.std(final_test_losses),
            'values': final_test_losses
        },
        'train_mae': {
            'mean': np.mean(final_train_maes),
            'std': np.std(final_train_maes),
            'values': final_train_maes
        },
        'test_mae': {
            'mean': np.mean(final_test_maes),
            'std': np.std(final_test_maes),
            'values': final_test_maes
        }
    }

    return results

# Run multiple experiments
print("\n" + "=" * 60)
print("REPRODUCIBILITY ANALYSIS")
print("=" * 60)

experiment_results = run_multiple_experiments(num_runs=5, num_epochs=100)

# Print summary statistics
print(f"\nSummary of {len(experiment_results['test_mae']['values'])} runs:")
print("-" * 40)
print(f"Test MAE: {experiment_results['test_mae']['mean']:.4f} ± {experiment_results['test_mae']['std']:.4f}")
print(f"Test Loss: {experiment_results['test_loss']['mean']:.4f} ± {experiment_results['test_loss']['std']:.4f}")
print(f"Train MAE: {experiment_results['train_mae']['mean']:.4f} ± {experiment_results['train_mae']['std']:.4f}")
print(f"Train Loss: {experiment_results['train_loss']['mean']:.4f} ± {experiment_results['train_loss']['std']:.4f}")

# Plot results of multiple runs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# MAE comparison
runs = range(1, len(experiment_results['test_mae']['values']) + 1)
ax1.plot(runs, experiment_results['train_mae']['values'], 'o-', label='Train MAE', color='blue')
ax1.plot(runs, experiment_results['test_mae']['values'], 's-', label='Test MAE', color='red')
ax1.set_title('MAE Across Multiple Runs')
ax1.set_xlabel('Run Number')
ax1.set_ylabel('Mean Absolute Error')
ax1.legend()
ax1.grid(True)

# Box plot of test MAE
ax2.boxplot([experiment_results['test_mae']['values']], labels=['Test MAE'])
ax2.set_title('Test MAE Distribution')
ax2.set_ylabel('Mean Absolute Error')
ax2.grid(True)

plt.tight_layout()
plt.savefig('reproducibility_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Reproducibility analysis saved to 'reproducibility_analysis.png'")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)