import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, nhead=4, ff_dim=256, num_layers=2, dropout=0.1, pred_steps=60):
        super().__init__()
        self.pred_steps = pred_steps
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.posenc = SinusoidalPositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True, activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 2 * pred_steps)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.posenc(x)
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out.view(-1, self.pred_steps, 2)
