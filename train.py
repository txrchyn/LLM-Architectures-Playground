import tiktoken
import torch
import tiktoken
import sys
import time
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.functional import F

from GPT import GPT
from GPTConfig import GPTConfig
from DataLoaderLite import DataLoaderLite


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

num_return_sequence = 5
max_length = 30

# get a data batch
train_loader = DataLoaderLite(B=8, T=1024)

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig)
model.to(device)

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# logits, loss = model(x, y)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
losses = []

torch.cuda.empty_cache()

for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    losses.append(loss.item())
    print(f"iter: {i}, loss: {loss.item()}, time: {dt:.2f}ms")

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss over Iterations")
plt.grid(True)
plt.savefig("assets/loss.png")

print(f"loss: {loss}, min loss: {min(losses)}")
sys.exit(0)
model.eval()


enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm language model, ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)
x = tokens.to('cuda')

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (B, 50), topk_indices is (B, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequence):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print("> " + decoded + "\n")
