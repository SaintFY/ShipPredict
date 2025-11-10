import torch
import torch.nn as nn

class BiGRUModel(nn.Module):
    """双向 GRU"""
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, bidirectional=True, pred_steps=60):
        super().__init__()
        self.pred_steps = pred_steps
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_hidden = hidden_size * 2
        self.fc = nn.Linear(out_hidden, 2 * pred_steps)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        out = self.fc(last)
        return out.view(-1, self.pred_steps, 2)
