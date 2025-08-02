import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tiktoken
import torch.nn.functional as F
from models.GPT_2_modern import GPT, MicroGPTConfig, GPTConfig 

checkpoint_path = "checkpoints_main/ckpt_step_19072.pth" 

used_config = GPTConfig()   

tokenizer_type = 'tinystories' 

max_new_tokens = 200
temperature = 0.8
top_k = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

model = GPT(used_config)
state_dict = checkpoint['model']

state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

unwrapped_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model.load_state_dict(unwrapped_state_dict)
model.eval()
model.to(device)
print("Model loaded successfully.")


print(f"Initializing tokenizer: {tokenizer_type}")
if tokenizer_type == 'shakespeare':
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

# ==============================================================================


print("\n--- Start Chatting (press Enter on an empty line to exit) ---")
while True:
    prompt = input("Prompt: ")
    if not prompt:
        break

    start_tokens = encode(prompt)
    x = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated_tokens_tensor = model.generate(
        idx=x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )

    # Декодируем и печатаем результат
    generated_tokens = generated_tokens_tensor[0].tolist()
    completion = decode(generated_tokens)
    
    print("\n--- Completion ---")
    print(completion)
    print("------------------\n")