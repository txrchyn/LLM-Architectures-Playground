from typing import Dict, Any

import torch
import torch.nn as nn

from .gpt_2_modern_config import gpt_2_modern_config

from .CausalSelfAttention import CausalSelfAttention
from .MLP import MLP


class Block(nn.Module):
    """
    A single Transformer block, which combines a multi-head self-attention layer
    with a feed-forward MLP, using pre-layer normalization, RMSNorm, and residual connections.
    """
    def __init__(self, config: gpt_2_modern_config) -> None:
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the Transformer block.

        Args:
            x: The input tensor of shape (B, T, D).

        Returns:
            The output tensor of shape (B, T, D).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x