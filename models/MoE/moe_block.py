import torch.nn as nn
from .moe_config import MoEConfig
from .moe_attention import CausalSelfAttention
from .moe_mlp import MoEFeedForward

class MoEBlock(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn  = MoEFeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        ffn_out, aux_loss = self.ffn(self.ln_2(x))
        x = x + ffn_out
        return x, aux_loss
