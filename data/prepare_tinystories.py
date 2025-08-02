import os
import tiktoken
import numpy as np

input_dir = "data"
output_dir = "data/TinyStoriesV2"
train_input_file = os.path.join(input_dir, "TinyStoriesV2-GPT4-train.txt")
val_input_file = os.path.join(input_dir, "TinyStoriesV2-GPT4-valid.txt")

os.makedirs(output_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
print(f"Tokenizer vocab size: {enc.n_vocab}")

def tokenize_file_chunked(input_path, output_path, chunk_size=10 * 1024 * 1024): # 10 MB chunks
    print(f"Tokenizing {input_path} -> {output_path}...")
    
    with open(output_path, 'wb') as f_out:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            total_tokens = 0
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                
                tokens = enc.encode(chunk, allowed_special={"<|endoftext|>"})
                
                tokens_np = np.array(tokens, dtype=np.uint16)
                
                f_out.write(tokens_np.tobytes())
                
                total_tokens += len(tokens_np)
                print(f"\rProcessed {total_tokens // 1000}k tokens...", end='', flush=True)

    print(f"\nSaved {total_tokens} tokens to {output_path}")

tokenize_file_chunked(train_input_file, os.path.join(output_dir, "train.bin"))
tokenize_file_chunked(val_input_file, os.path.join(output_dir, "val.bin"))

print("Done.")