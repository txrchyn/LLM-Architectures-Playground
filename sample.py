import os
import yaml
import torch
import tiktoken
from argparse import ArgumentParser

from models import create_model

parser = ArgumentParser(description='Generate text from a trained GPT model.')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to a checkpoint file or a directory containing checkpoints.')
parser.add_argument('--prompt', type=str, default="Hello, I am a language model,", help='The prompt to start generation from.')
parser.add_argument('--max_new_tokens', type=int, default=100, help='Number of new tokens to generate.')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
torch.set_float32_matmul_precision('high')

checkpoint_path = args.checkpoint_path

if os.path.isdir(checkpoint_path):
    print(f"Directory provided. Searching for the latest checkpoint in {checkpoint_path}...")
    checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoint files found in directory: {checkpoint_path}")
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
elif not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"The provided path is not a valid file or directory: {checkpoint_path}")

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

config = checkpoint.get('config')
if config is None:
    raise ValueError("Config not found in checkpoint. Please retrain and save with config.")

model_name = config.get('model_name')
print(f"Initializing model: {model_name}")

model = create_model(model_name, config)

state_dict = checkpoint['model']
state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()
model.to(device)
print("Model loaded successfully.")

tokenizer_name = config.get('dataset_name', 'tinystories')
print(f"Initializing tokenizer for: {tokenizer_name}")
if 'shakespeare' in tokenizer_name:
    with open('data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

start_tokens = encode(args.prompt)
x = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

with torch.no_grad():
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        generated_tokens_tensor = model.generate(
            idx=x,
            max_new_tokens=args.max_new_tokens
        )

generated_tokens = generated_tokens_tensor[0].tolist()
completion = decode(generated_tokens)

print("\n--- Completion ---")
print(completion)
print("------------------\n")