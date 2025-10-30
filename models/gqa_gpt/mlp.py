import torch.nn as nn

from .config import GQAGPTConfig


class MLP(nn.Module):
    def __init__(self, config: GQAGPTConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=True)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=True)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x
