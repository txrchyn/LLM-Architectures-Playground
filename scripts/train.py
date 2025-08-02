import os
import time
import math

import wandb
import torch
from dataclasses import asdict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from ..models.GPT_2_modern import GPT, MicroGPTConfig, GPTConfig
from ..utils import DataLoaderLite


torch.set_float32_matmul_precision('high')
# torchrun command sets the env variables RANK = Unique rank of the process across all nodes
#                                         LOCAL_RANK = Unique rank of the process on the current node
#                                         WORLD_SIZE =  Total number of gpus across all nodes
ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1

CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if ddp:
    assert torch.cuda.is_available(), 'DDP requires CUDA support'
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

# The dataset used here is a FineWeb_Edu 10B tokens
B = 24
T = 1024 # 1024ё
total_batch_size = B * T # 524288 = 2^19, ~0.5M, in number of tokens
max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 20000 # 19073 steps is ~1 epoch, ~10B tokens of FineWeb_Edu dataset
warmup_steps = max_steps * 0.07 # in gpt3 paper was used the same proportion = 715/19073


assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = int(total_batch_size // (B * T * ddp_world_size))


if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"Calculatted gradient accumulation steps: {grad_accum_steps}")

if master_process:
    wandb.init(                                            
        project="nanogpt2", 
        entity="kyrylo-turchyn-",
        name=f"+ RMSNormm",
        config={
            "model": asdict(MicroGPTConfig()),
            "total_batch_size": total_batch_size,
            "batch_size_per_gpu": B,
            "sequ e_length": T,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "grad_accum_steps": grad_accum_steps,
            "world_size": ddp_world_size,
        }
    )

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', dataset_name='tinystories')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', dataset_name='tinystories')

model = GPT(MicroGPTConfig())
model.to(device)
model = torch.compile(model)

if master_process:
    wandb.watch(model, log="all", log_freq=100)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model
# ---- CHECKPOINT LOADING ----
start_step = 0
latest_checkpoint_path = None
if os.path.exists(CHECKPOINT_DIR) and len(os.listdir(CHECKPOINT_DIR)) > 0:
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_step_") and f.endswith(".pth")]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])

if latest_checkpoint_path:
    print(f"Resuming from checkpoint: {latest_checkpoint_path}")
    checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
    # When loading a DDP model checkpoint onto a non-DDP model (or vice-versa)
    # the keys in state_dict might not match. We need to adjust them.
    state_dict = checkpoint['model']
    unwrapped_model = model.module if ddp else model
    # Fix for DDP model saved without DDP
    if ddp and not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    # Fix for non-DDP model saved with DDP
    elif not ddp and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    unwrapped_model.load_state_dict(state_dict)
    if 'optimizer' in checkpoint:
        optimizer_state_dict = checkpoint['optimizer']
        # Ensure the optimizer is created before loading its state
        # We'll create the optimizer a bit later in the code, so we'll load its state then.
        # For now, we'll store it.
        loaded_optimizer_state_dict = optimizer_state_dict
    else:
        loaded_optimizer_state_dict = None
    start_step = checkpoint.get('step', 0) + 1
    print(f"Loaded model from step {start_step -1}. Resuming training from step {start_step}")
else:
    print("No checkpoint found, starting training from scratch.")
    loaded_optimizer_state_dict = None


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# logits, loss = model(x, y)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, betas=(0.9, 0.95), device=device)

# ---- LOAD OPTIMIZER STATE ----
if loaded_optimizer_state_dict:
    optimizer.load_state_dict(loaded_optimizer_state_dict)
    print("Loaded optimizer state from checkpoint.")

# ---- SAVE CHECKPOINT FUNCTION ----
def save_checkpoint(step, model_to_save, opt, current_loss):
    
    if master_process:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ckpt_step_{step}.pth")
        checkpoint = {
            'model': model_to_save.state_dict(),
            'optimizer': opt.state_dict(),
            'step': step,
            'loss': current_loss,
            'config': model_to_save.config if hasattr(model_to_save, 'config') else None,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path} at step {step}")

        # Keep only the last 5 checkpoints
        all_checkpoints = sorted(
            [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.startswith("ckpt_step_") and f.endswith(".pth")],
            key=lambda x: int(x.split("_")[2].split(".")[0])
        )
        if len(all_checkpoints) > 5:
            for old_ckpt in all_checkpoints[:-5]:
                os.remove(old_ckpt)
                print(f"Removed old checkpoint: {old_ckpt}")

for step in range(start_step, max_steps):
    t0 = time.time()
    # once in a while evaluate our validation loss
    if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            wandb.log({"train/val_loss": val_loss_accum.item()}, step=step)


    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)
    ppl  = torch.exp(loss)

    # once in a while save checkpoint
    if step > 0 and step % 1000 == 0: # Save every 1000 steps
         save_checkpoint(step, raw_model, optimizer, loss_accum.item() if master_process else None)

    print(f"iter: {step+1:4d}  |  loss: {loss_accum.item():.6f}  |  ppl: {ppl:.6f}  |  lr {lr:.4e}  |  norm {norm:.4f}  |  time: {dt:.2f}ms  |  tok/sec: {tokens_per_sec:.2f}")
    wandb.log({
            "train/loss": loss_accum.item(),
            "train/perplexity": ppl.item(),
            "train/grad_norm": norm.item(),
            "train/tokens_per_sec": tokens_per_sec,
            "train/time_ms": dt,
        }, step=step)

# Save final checkpoint
save_checkpoint(max_steps -1, raw_model, optimizer, losses[-1] if losses else None)
wandb.finish()
if ddp:
    destroy_process_group()
