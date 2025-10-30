import torch.nn as nn

from .linformer_config import LinformerConfig
from .LinformerAttention import LinformerSelfAttention
from .LinformerMLP import LinformerMLP


class LinformerBlock(nn.Module):
    def __init__(self, config: LinformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = LinformerSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = LinformerMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
