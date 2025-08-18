import inspect
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings

from .gpt_2_modern_config import gpt_2_modern_config


class CausalSelfAttention(nn.Module):
    """
    A multi-head causal self-attention module that integrates Rotary Positional
    Embeddings (RoPE) using the torchtune library implementation.
    """
    def __init__(self, config: gpt_2_modern_config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Instantiate the RoPE module from torchtune
        self.head_size = self.n_embd // self.n_head
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_size,
            max_seq_len=config.block_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the attention module.

        Args:
            x: The input tensor of shape (B, T, D).

        Returns:
            The output tensor of shape (B, T, D).
        """
        B, T, D = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)

        # Generate position IDs for RoPE
        positions = torch.arange(0, T, device=x.device)
        
        # Apply RoPE to q and k
        q = self.rope(q, input_pos=positions)
        k = self.rope(k, input_pos=positions)
        
        # Transpose for multi-head attention calculation
        q = q.transpose(1, 2) # (B, n_head, T, head_size)
        k = k.transpose(1, 2) # (B, n_head, T, head_size)
        v = v.transpose(1, 2) # (B, n_head, T, head_size)
        
        # Use scaled_dot_product_attention for efficiency
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.c_proj(y)
        
        return y


    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx


    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device: str
    ) -> torch.optim.AdamW:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer