import torch.nn as nn
from typing import Dict, Any


class MLP(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'])
        self.c_proj.NANO_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
