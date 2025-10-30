from dataclasses import dataclass


@dataclass
class LinformerConfig:
    # --- Wandb Config ---
    wandb_log: bool = False
    wandb_run_name: str = 'linformer-gpt'

    # --- Model Config ---
    model_name: str = 'linformer'
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50304
    bias: bool = False
    proj_dim: int = 256  # sequence projection length (k in Linformer)

    # --- Training Config ---
    batch_size: int = 8
    sequence_length: int = 1024
    total_batch_size: int = 65536
    max_steps: int = 1000
    max_lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps_percentage: float = 0.07
    beta1: float = 0.9
    beta2: float = 0.95

    # --- I/O Config ---
    dataset_name: str = 'tinystories'
    out_dir: str = 'checkpoints'
    save_checkpoints: bool = False
    every_n_steps_save: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20


config = LinformerConfig()
