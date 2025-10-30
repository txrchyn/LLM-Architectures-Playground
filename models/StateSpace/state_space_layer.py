import torch
import torch.nn as nn

from .state_space_config import StateSpaceConfig


@torch.jit.script
def _ssm_scan(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    B, T, D = x.shape
    state = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    outputs = torch.empty(B, T, D, device=x.device, dtype=x.dtype)
    for t in range(T):
        u_t = x[:, t, :]
        state = state * a + u_t * b
        outputs[:, t, :] = state * c + u_t * d
    return outputs


class StateSpaceLayer(nn.Module):
    """
    Lightweight diagonal state-space layer:
        s_t = a * s_{t-1} + b * u_t
        y_t = c * s_t + d * u_t
    Parameters (a, b, c, d) are learned per feature. The diagonal form keeps the update cheap
    while still modelling long-range dependencies when |a| < 1.
    """
    def __init__(self, config: StateSpaceConfig) -> None:
        super().__init__()
        hidden = config.n_embd
        state_dim = config.state_dim
        assert state_dim == hidden, "This simple diagonal SSM assumes state_dim == n_embd."

        self.hidden = hidden

        self.logit_a = nn.Parameter(torch.zeros(hidden))
        self.b = nn.Parameter(torch.randn(hidden) * 0.1)
        self.c = nn.Parameter(torch.randn(hidden) * 0.1)
        self.d = nn.Parameter(torch.zeros(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, hidden)
        """
        B, T, D = x.shape
        assert D == self.hidden

        a = torch.tanh(self.logit_a)
        return _ssm_scan(x, a, self.b, self.c, self.d)
