#!/usr/bin/env python3
"""
Test script for the Adding Machine Neural Network
This version runs without matplotlib to test core functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("Testing Adding Machine Neural Network")
print("=" * 50)

# Generate dataset
N = 1000
inputs = torch.randint(-10, 11, (N, 2)).float()
targets = torch.sum(inputs, axis=1, keepdim=True)

print(f"Dataset created with {N} samples")
print(f"Sample data points:")
for i in range(5):
    print(f"  {inputs[i].numpy()} -> {targets[i].item()}")

# Split data
train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    inputs, targets, test_size=0.2, random_state=42
)

# Create data loaders
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

class AddingMachine(nn.Module):
    def __init__(self, n_units=16, n_layers=2):
        super().__init__()
        self.input_layer = nn.Linear(2, n_units)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(n_units, n_units) for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(n_units, 1)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x

# Create model
model = AddingMachine(16, 2)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

# Training
num_epochs = 50
print(f"\nTraining for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for X, y in train_loader:
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        X_test, y_test = next(iter(test_loader))
        y_pred_test = model(X_test)
        test_loss = loss_fn(y_pred_test, y_test).item()
        test_mae = torch.mean(torch.abs(y_pred_test - y_test)).item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

print("\nTraining completed!")

# Test the model
print("\nTesting the trained model:")
print("Input (a, b) -> Predicted | Actual | Error")
print("-" * 40)

model.eval()
total_error = 0
num_tests = 10

with torch.no_grad():
    for _ in range(num_tests):
        a, b = np.random.randint(-10, 11, 2)
        test_input = torch.tensor([[float(a), float(b)]])
        predicted = model(test_input).item()
        actual = a + b
        error = abs(predicted - actual)
        total_error += error
        
        print(f"({a:3d}, {b:3d}) -> {predicted:7.2f} | {actual:7d} | {error:.2f}")

avg_error = total_error / num_tests
print(f"\nAverage absolute error: {avg_error:.3f}")

# Test on edge cases
print("\nTesting edge cases:")
edge_cases = [(-10, -10), (-10, 10), (10, -10), (10, 10), (0, 0)]
with torch.no_grad():
    for a, b in edge_cases:
        test_input = torch.tensor([[float(a), float(b)]])
        predicted = model(test_input).item()
        actual = a + b
        error = abs(predicted - actual)
        print(f"({a:3d}, {b:3d}) -> {predicted:7.2f} | {actual:7d} | {error:.2f}")

print("\nTest completed successfully!")
