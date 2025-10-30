from dataclasses import dataclass

@dataclass
class MoEConfig:
    # --- Wandb/Logging ---
    wandb_log: bool = True
    wandb_run_name: str = 'moe-gpt-124M'

    # --- Model ---
    model_name: str = 'moe'
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50304
    bias: bool = False
    share_input_output_embeddings: bool = True

    # --- MoE ---
    num_experts: int = 2
    top_k: int = 1                # сколько экспертов активировать на токен
    capacity_factor: float = 1.0  # емкость на эксперта = ceil(capacity_factor * N * top_k / E)
    router_jitter: float = 0.0    # шум для стабильности (0.0 = без шума)
    aux_loss_weight: float = 0.01 # коэффициент потери балансировки нагрузки

    # --- Training ---
    dataset_name: str = 'tinystories' # 'shakespeare', 'tinystories', 'fineweb'
    out_dir: str = 'checkpoints'
    save_checkpoints: bool = False
    batch_size: int = 4
    sequence_length: int = 1024
    total_batch_size: int = 65536
    max_steps: int = 1000
    max_lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps_percentage: float = 0.07
    beta1: float = 0.9
    beta2: float = 0.95
    every_n_steps_save: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20

config = MoEConfig()