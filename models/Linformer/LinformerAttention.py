import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linformer_config import LinformerConfig


class LinformerSelfAttention(nn.Module):
    """
    Linformer-style self-attention with learned projection of keys/values along the sequence axis.
    """
    def __init__(self, config: LinformerConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert config.proj_dim <= config.block_size, "proj_dim must be <= block_size"

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.proj_dim = config.proj_dim
        self.block_size = config.block_size

        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out_proj.NANOGPT_SCALE_INIT = 1

        # sequence projection matrices (shared across heads)
        self.E_k = nn.Parameter(torch.randn(config.block_size, self.proj_dim) / math.sqrt(self.proj_dim))
        self.E_v = nn.Parameter(torch.randn(config.block_size, self.proj_dim) / math.sqrt(self.proj_dim))

    def _causal_linformer_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Build a coarse causal mask in projected space: bucket j summarizes the first
        ceil((j+1)/proj_dim * T) tokens, so query position t may only attend to buckets
        whose boundary is <= t.
        """
        t_idx = torch.arange(T, device=device)
        bucket_limits = torch.ceil(
            (torch.arange(self.proj_dim, device=device) + 1) * (T / self.proj_dim)
        ).long() - 1
        bucket_limits = torch.clamp(bucket_limits, max=T - 1)
        return t_idx.unsqueeze(-1) >= bucket_limits.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        E_k = self.E_k[:T, :]  # (T, P)
        E_v = self.E_v[:T, :]

        k_proj = torch.einsum('bnth,tp->bnph', k, E_k)  # (B, H, P, Dh)
        v_proj = torch.einsum('bnth,tp->bnph', v, E_v)  # (B, H, P, Dh)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.einsum('bnth,bnph->bntp', q, k_proj) * scale  # (B, H, T, P)

        causal_mask = self._causal_linformer_mask(T, x.device)
        att = att.masked_fill(~causal_mask.view(1, 1, T, self.proj_dim), float('-inf'))

        att = F.softmax(att, dim=-1)
        y = torch.einsum('bntp,bnph->bnth', att, v_proj)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        y = self.out_proj(y)
        return y
