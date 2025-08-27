from dataclasses import dataclass

@dataclass
class gpt_2_modern_config:
    # --- Wandb Config ---
    wandb_log: bool = False
    wandb_run_name: str = 'modern-gpt-124M'

    # --- Model Config ---
    model_name: str = 'modern'
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50304
    bias: bool = False

    # --- Training Config ---
    batch_size: int = 16
    sequence_length: int = 1024
    total_batch_size: int = 65536
    max_steps: int = 1000 #19073
    max_lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps_percentage: float = 0.07  # 7% of total steps
    beta1: float = 0.9
    beta2: float = 0.95

    # --- I/O Config ---
    dataset_name: str = 'tinystories' # 'shakespeare', 'tinystories', 'fineweb'
    out_dir: str = 'checkpoints'
    save_checkpoints: bool = False
    every_n_steps_save: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20


config = gpt_2_modern_config()