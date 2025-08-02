from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024      # max seq len
    vocab_size: int = 50304     # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token + 47 to make a "beauty" number
    n_layer:    int = 12        # num of layers
    n_head:     int = 12        # num of heads
    n_embd:     int = 768       # embedding dimension
