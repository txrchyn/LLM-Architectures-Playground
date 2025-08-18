import torch.nn as nn

from typing import Dict, Any

from .MLP import MLP
from .CausalSelfAttention import CausalSelfAttention
from .gpt_2_vanilla_config import gpt_2_vanilla_config


class Block(nn.Module):
    def __init__(self, config: gpt_2_vanilla_config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x