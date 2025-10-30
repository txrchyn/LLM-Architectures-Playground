import torch.nn as nn

from .state_space_config import StateSpaceConfig


class StateSpaceMLP(nn.Module):
    def __init__(self, config: StateSpaceConfig) -> None:
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
