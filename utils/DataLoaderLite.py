import torch
import os
import numpy as np


def load_tokens(filename):
    npt = np.fromfile(filename, dtype=np.uint16) 
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, dataset_name='fineweb'):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.dataset_name = dataset_name

        if self.dataset_name == 'fineweb':
            data_root = "data/edu_fineweb10B"
            shards = os.listdir(data_root)
            shards = [s for s in shards if split in s]
            shards = sorted(shards)
            shards = [os.path.join(data_root, s) for s in shards]
            self.shards = shards
            assert len(shards) > 0, f"no shards found for split {split}"
            if self.process_rank == 0:
                print(f"found {len(shards)} shards for split {split}")
        
        elif self.dataset_name == 'shakespeare':
            data_root = "data"
            text_file = os.path.join(data_root, 'tinyshakespeare.txt')
            with open(text_file, 'r') as f:
                text = f.read()
            chars = sorted(list(set(text)))
            vocab_size = len(chars)
            stoi = { ch:i for i,ch in enumerate(chars) }
            itos = { i:ch for i,ch in enumerate(chars) }
            encode = lambda s: [stoi[c] for c in s]
            all_tokens = torch.tensor(encode(text), dtype=torch.long)
            n = len(all_tokens)
            train_size = int(n * 0.9)
            if self.split == 'train':
                self.tokens = all_tokens[:train_size]
            else:
                self.tokens = all_tokens[train_size:]
            if self.process_rank == 0:
                print(f"Loaded {len(self.tokens)} tokens for split {split} from tinyshakespeare")

        elif self.dataset_name == 'tinystories':
            data_root = "data/TinyStoriesV2"
            token_file = os.path.join(data_root, f"{self.split}.bin")
            self.tokens = load_tokens(token_file)
            if self.process_rank == 0:
                print(f"Loaded {len(self.tokens)} tokens for split {split} from {token_file}")

        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        self.reset()

    def reset(self):
        if self.dataset_name == 'fineweb':
            self.current_shard = 0
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        elif self.dataset_name == 'shakespeare' or self.dataset_name == 'tinystories':
            self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes

        if self.dataset_name == 'fineweb':
            if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
                self.current_position = B * T * self.process_rank
        elif self.dataset_name == 'shakespeare' or self.dataset_name == 'tinystories':
            if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
                self.current_position = self.B * self.T * self.process_rank

        return x, y
