"""
06_Empirical_Experiment/LSTM/train_stacked_lstm.py
Layer 3 (Dynamic Branch): 2-Layer Stacked LSTM Neural Network for LMS Sequence Trajectories.
Processes weekly interaction sequences (logins, submissions, forum posts) up to horizon T.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class StackedLSTMBranch(nn.Module):
    """
    2-Layer Stacked LSTM for dynamic LMS interaction sequences.
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super(StackedLSTMBranch, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take last time step output
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        prob = self.sigmoid(out)
        return prob

def train_lstm_dynamic_branch(X_seq: torch.Tensor, y_train: torch.Tensor, epochs: int = 20) -> StackedLSTMBranch:
    """
    Trains the PyTorch 2-layer Stacked LSTM dynamic sequence branch.
    """
    model = StackedLSTMBranch(input_dim=X_seq.shape[2], hidden_dim=64, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_seq).squeeze()
        loss = criterion(predictions, y_train.float())
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"[Stacked LSTM] Epoch {epoch+1}/{epochs} - BCELoss: {loss.item():.4f}")
            
    return model

if __name__ == "__main__":
    # Batch size 32, 12 weeks sequence length, 5 LMS features per week
    dummy_seq = torch.randn(32, 12, 5)
    dummy_labels = torch.randint(0, 2, (32,))
    lstm_model = train_lstm_dynamic_branch(dummy_seq, dummy_labels, epochs=5)
