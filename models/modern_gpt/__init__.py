# model/__init__.py
from .model import ModernGPT
from .block import Block
from .mlp import MLP
from .attention import CausalSelfAttention
from .config import ModernGPTConfig, config

__all__ = ['ModernGPT', 'Block', 'MLP', 'CausalSelfAttention', 'ModernGPTConfig', 'config']
