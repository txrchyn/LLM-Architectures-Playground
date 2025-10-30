import torch.nn as nn

from .state_space_config import StateSpaceConfig
from .state_space_layer import StateSpaceLayer
from .state_space_mlp import StateSpaceMLP


class StateSpaceBlock(nn.Module):
    def __init__(self, config: StateSpaceConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ssm = StateSpaceLayer(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = StateSpaceMLP(config)

    def forward(self, x):
        x = x + self.ssm(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
