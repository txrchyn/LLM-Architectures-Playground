import os
import sys
import time
import math
import importlib.util
from dotenv import load_dotenv
from argparse import ArgumentParser

import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils import DataLoaderLite
from models import create_model

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def setup_ddp():
    ddp = int(os.environ.get('LOCAL_RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), 'DDP requires CUDA support'
        init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        master_process = (rank == 0)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        master_process = True
        print(f"Device: {device}")
    return ddp, rank, local_rank, world_size, device, master_process

def load_config_from_py_file(config_path):
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the config file from {config_path}")
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module.config

def get_lr(it, config):
    warmup_steps = config.max_steps * config.warmup_steps_percentage
    min_lr = config.max_lr * config.weight_decay
    if it < warmup_steps:
        return config.max_lr * (it + 1) / warmup_steps
    if it > config.max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (config.max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (config.max_lr - min_lr)

@torch.no_grad()
def evaluate_loss(model, loader, device, config):
    model.eval()
    loader.reset()
    autocast_context = torch.autocast(device_type=device, dtype=torch.bfloat16)
    val_loss_accum = torch.zeros(1, device=device)
    for _ in range(config.eval_iters):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with autocast_context:
            _, loss = model(x, y)
        val_loss_accum += loss / config.eval_iters
    model.train()
    return val_loss_accum

def save_checkpoint(step, model, optimizer, loss, config, is_master):
    if not is_master:
        return
    
    checkpoint_dir = f"checkpoints/{config.model_name}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    path = os.path.join(checkpoint_dir, f"ckpt_step_{step}.pth")
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
        'loss': loss,
        'config': config,
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path} at step {step}")

    # Cleanup old checkpoints
    all_checkpoints = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pth")],
        key=lambda x: int(x.split("_")[2].split(".")[0])
    )
    if len(all_checkpoints) > 5:
        for old_ckpt in all_checkpoints[:-5]:
            os.remove(old_ckpt)
            print(f"Removed old checkpoint: {old_ckpt}")

# -----------------------------------------------------------------------------
# Main execution block
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    load_dotenv()
    torch.set_float32_matmul_precision('high')

    parser = ArgumentParser(description='Train a GPT model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the Python config file.')
    args = parser.parse_args()

    config = load_config_from_py_file(args.config)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_ddp()

    tokens_per_iter = config.batch_size * config.sequence_length * ddp_world_size
    grad_accum_steps = config.total_batch_size // tokens_per_iter
    assert config.total_batch_size % tokens_per_iter == 0

    if master_process:
        print(f"Total desired batch size: {config.total_batch_size}")
        print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

    if config.wandb_log and master_process:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT_NAME"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=config.wandb_run_name,
            config=vars(config)
        )

    train_loader = DataLoaderLite(B=config.batch_size, T=config.sequence_length, process_rank=ddp_rank,
                                  num_processes=ddp_world_size, split='train', dataset_name=config.dataset_name)
    val_loader = DataLoaderLite(B=config.batch_size, T=config.sequence_length, process_rank=ddp_rank,
                                num_processes=ddp_world_size, split='val', dataset_name=config.dataset_name)

    model = create_model(config.model_name, config)
    model.to(device)
    try:
        model = torch.compile(model)
    except Exception:
        pass

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.max_lr,
        betas=(config.beta1, config.beta2),
        device=device
     )

    start_step = 0
    checkpoint_dir = f"checkpoints/{config.model_name}"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if checkpoint_files:
            latest_file = max(checkpoint_files, key=lambda x: int(x.split("_")[2].split(".")[0]))
            path = os.path.join(checkpoint_dir, latest_file)
            print(f"Resuming from checkpoint: {path}")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            raw_model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint.get('step', 0) + 1

    autocast_context = torch.autocast(device_type=device, dtype=torch.bfloat16)

    for step in range(start_step, config.max_steps):
        t0 = time.time()

        if step > 0 and step % config.eval_interval == 0:
            val_loss = evaluate_loss(model, val_loader, device, config)
            if ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation loss: {val_loss.item():.4f}")
                if config.wandb_log:
                    wandb.log({"val_loss": val_loss.item()}, step=step)

        optimizer.zero_grad()
        loss_accum = torch.zeros(1, device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with autocast_context:
                _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        torch.cuda.synchronize()
        t1 = time.time()

        if master_process:
            dt = (t1 - t0) * 1000
            tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)
            
            print(f"iter: {step+1:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm {norm:.4f} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            if config.wandb_log:
                wandb.log({
                    "loss": loss_accum.item(),
                    "grad_norm": norm.item(),
                    "tokens_per_sec": tokens_per_sec,
                    "time_ms": dt,
                    "lr": lr,
                }, step=step)

        if config.save_checkpoints and step > 0 and step % config.every_n_steps_save == 0:
            save_checkpoint(step, raw_model, optimizer, loss_accum.item(), master_process)

    save_checkpoint(config.max_steps - 1, raw_model, optimizer, loss_accum.item(), master_process)
    if config.wandb_log and master_process:
        wandb.finish()
    if ddp:
        destroy_process_group()