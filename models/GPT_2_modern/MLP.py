from typing import Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from .gpt_2_modern_config import gpt_2_modern_config


class MLP(nn.Module):
    """
    A SwiGLU-based MLP module, following modern LLM architecture practices.
    """
    def __init__(self, config: gpt_2_modern_config) -> None:
        super().__init__()
        
        # Llama-style hidden dimension calculation for SwiGLU
        hidden_dim = int(2/3 * (4 * config.n_embd))

        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)

        # Retain custom initialization scheme for the projection layer
        self.w2.NANO_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SwiGLU MLP.
        
        Args:
            x: Input tensor of shape (B, T, n_embd).
        
        Returns:
            Output tensor of shape (B, T, n_embd).
        """
        # Apply the SwiGLU activation: F.silu(w1(x)) * w3(x)
        gated_output = F.silu(self.w1(x)) * self.w3(x)
        
        return self.w2(gated_output)