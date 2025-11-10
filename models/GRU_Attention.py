import torch
import torch.nn as nn

class GRUAttention(nn.Module):
    """GRU + 注意力机制"""
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, bidirectional=False, pred_steps=60):
        super().__init__()
        self.pred_steps = pred_steps
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_hidden = hidden_size * (2 if bidirectional else 1)
        self.attn = nn.Linear(out_hidden, 1)
        self.fc = nn.Linear(out_hidden, 2 * pred_steps)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_w = torch.softmax(self.attn(gru_out).squeeze(-1), dim=1)
        context = torch.sum(gru_out * attn_w.unsqueeze(-1), dim=1)
        out = self.fc(context)
        return out.view(-1, self.pred_steps, 2)
