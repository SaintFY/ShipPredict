import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """单向 LSTM 回归未来经纬度序列"""
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, bidirectional=False, pred_steps=60):
        super().__init__()
        self.pred_steps = pred_steps
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_hidden = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_hidden, 2 * pred_steps)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.fc(last)
        print(f"[DEBUG] fc output shape: {out.shape}, pred_steps={self.pred_steps}")
        out = out.view(-1, self.pred_steps, 2)
        return out

