from dataclasses import dataclass


@dataclass
class MicroGPTConfig:
    block_size: int = 1024   
    vocab_size: int = 50304  
    n_layer:    int = 8   
    n_head:     int = 8
    n_embd:     int = 512      