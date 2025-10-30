import torch.nn as nn
from .moe_config import MoEConfig

class FeedForwardExpert(nn.Module):
    """
    Один эксперт: стандартный GPT-FFN (GELU MLP).
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=True)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=True)
        self.c_proj.NANO_SCALE_INIT = 1

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
