import torch.nn as nn

from typing import Dict, Any

from .MLP import MLP
from .CausalSelfAttention import CausalSelfAttention


class Block(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x