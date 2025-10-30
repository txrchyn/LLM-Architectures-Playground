# LLM Architectures Playground

This repository is a playground for implementing, training, and experimenting with modern, nano-sized Large Language Model architectures. The project is heavily inspired by Andrej Karpathy's renowned [nanoGPT](https://github.com/karpathy/nanogpt) but focuses on integrating more contemporary and efficient components found in state-of-the-art models like Llama and Mistral. It also explores different architectural decisions, such as Grouped-Query Attention, Multi-Head Latent Attention, and models with KV-caching.
## Project Structure

```
LLM-ARCHITECTURES-PLAYGROUND/
├── checkpoints/               # Directory for saving model checkpoints
├── data/                       # Datasets and preparation scripts
│   ├── edu_fineweb10B/         # Shards for the FineWeb dataset
│   ├── TinyStoriesV2/          # The TinyStories dataset
│   ├── prepare_fineweb.py      # Script to download and tokenize FineWeb
│   ├── prepare_tinystories.py  # Script to prepare TinyStories
│   └── tinyshakespeare.txt     # A classic dataset for testing
├── models/                     # Model source code
│   ├── modern_gpt/             # Modern GPT implementation (RoPE + SwiGLU)
│   ├── vanilla_gpt/            # Classic GPT-2 baseline
│   ├── gqa_gpt/                # Grouped-Query GPT variant
│   ├── Linformer/              # Linformer-style efficient transformer
│   ├── StateSpace/             # State-space language model
│   └── MoE/                    # Mixture-of-Experts transformer
├── utils/                      # Helper utilities
│   └── DataLoaderLite.py       # An efficient data loader
├── .env                        # Environment variables file (for WandB keys)
├── .gitignore
├── environment.yml             # Conda environment file
├── LICENSE
├── README.md
├── sample.py                   # Script for text generation (inference)
└── train.py                    # Main script for model training
```

## Getting Started

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LLM-Architectures-Playground.git
    cd LLM-Architectures-Playground
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate llm-playground
    ```

3.  **Set up environment variables (optional, for WandB):**
    Create a `.env` file in the root directory and add your Weights & Biases credentials
    ```
    WANDB_API_KEY="your_wandb_api_key"
    WANDB_PROJECT_NAME="llm-playground"
    WANDB_ENTITY="your_wandb_username"
    ```

4.  **Prepare the data:**
    Download and tokenize the datasets using the provided scripts. For example, to prepare TinyStories:
    ```bash
    python data/prepare_tinystories.py
    ```

## Usage

### Training a Model

The `train.py` script is used to launch training runs. It requires a path to a configuration file.

**Example: Training on a single GPU (Modern GPT):**
```bash
python train.py --config models/modern_gpt/config.py
```

**Example: Distributed training on 2 GPUs:**
```bash
torchrun --standalone --nproc_per_node=2 train.py --config models/modern_gpt/config.py
```
All training hyperparameters, such as batch size, learning rate, model architecture, and dataset, are defined within the specified config file.

Other available configs:

- `models/vanilla_gpt/config.py` – classic GPT-2 baseline  
- `models/gqa_gpt/config.py` – grouped-query attention GPT  
- `models/Linformer/linformer_config.py` – Linformer language model  
- `models/StateSpace/state_space_config.py` – state-space language model  
- `models/MoE/moe_config.py` – Mixture-of-Experts GPT

### Inference

Use the `sample.py` script to generate text from a trained model checkpoint.

```bash
python sample.py \
    --checkpoint_path="checkpoints/modern/best_checkpoint.pth" \
    --prompt="Once upon a time" \
    --max_new_tokens=100
```

### Benchmarking

This repo ships with a thin adapter for [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Install the harness (`pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@main`) and run:

```bash
python run_benchmarks.py \
  --model-name modern \
  --checkpoint checkpoints/modern/ckpt_step_999.pth \
  --tasks hellaswag,arc_easy \
  --batch-size 4 \
  --dtype bfloat16 \
  --device cuda:0
```

The script bootstraps our local adapter, forwards configuration to `lm-evaluation-harness`, and prints the aggregated JSON results. Use `--output-json metrics.json` to persist the full report.

## Architectures

This repository contains several GPT architectures designed for comparative analysis. The main goal is to demonstrate the impact of modern architectural improvements over a classic baseline. As new architectures and benchmarks are added, they will be included in the comparison below.

### Architectural Comparison

This table highlights the core component differences between the implemented models.

| Architecture | Key Components | Description / Purpose |
| :--- | :--- | :--- |
<<<<<<< HEAD
| **`GPT-2 (Vanilla)`** | • Learned Absolute Positional Embeddings<br>• LayerNorm<br>• GELU Activation<br>• Manual Attention Implementation | A faithful implementation of the original GPT-2 architecture. It serves as a strong, well-understood baseline for all experiments. |
| **`GPT-2 (Modern)`** | • **Rotary Positional Embeddings (RoPE)**<br>• **RMSNorm**<br>• **SwiGLU** Activation<br>• **Flash Attention** (`F.sdpa`) | An upgraded architecture incorporating modern, highly efficient techniques from models like Llama and Mistral for improved performance and training stability. |
| *`NextModel (TBD)`* | *• Grouped-Query Attention<br>• ...* | *A future implementation to test...* |
=======
| **Vanilla GPT** | • Learned absolute positional embeddings<br>• LayerNorm<br>• GELU MLP | Faithful GPT-2 baseline for apples-to-apples comparisons. |
| **Modern GPT** | • Rotary Positional Embeddings (RoPE)<br>• RMSNorm<br>• SwiGLU MLP<br>• Flash attention (`F.sdpa`) | Modernized GPT stack inspired by Llama/Mistral for better efficiency and stability. |
| **GQA GPT** | • Grouped-query attention (shared K/V heads)<br>• LayerNorm<br>• GELU MLP | Reduces KV cache size and speeds up decoding by sharing keys/values. |
| **Linformer GPT** | • Learned low-rank sequence projections<br>• LayerNorm<br>• GELU MLP | Linear-time attention via Linformer-style projections for long contexts. |
| **State-Space LM** | • Diagonal state-space residual blocks<br>• LayerNorm<br>• GELU MLP | Captures long-range patterns with recurrent state-space dynamics. |
| **MoE GPT** | • Top-k router with expert FFNs<br>• LayerNorm<br>• GELU experts | Mixture-of-Experts feed-forward layers for higher capacity at similar compute. |
>>>>>>> 2f43700 (add models and bench's)

### Benchmark Results

The following table presents the performance metrics for each architecture. The goal is to quantify the impact of architectural changes on both model quality and computational efficiency.

*All benchmarks are run on a single NVIDIA A100 GPU with `bfloat16` precision.*

| Model | Parameters | HellaSwag (acc_norm) | Training Speed (tokens/sec) | Inference Speed (tok/s) | VRAM Usage (Train, GB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Vanilla GPT` | 124M | `TBD` | `TBD` | `TBD` | `TBD` |
| `Modern GPT` | 123M | `TBD` | `TBD` | `TBD` | `TBD` |
| `GQA GPT` | 123M | `TBD` | `TBD` | `TBD` | `TBD` |
| `Linformer GPT` | 120M | `TBD` | `TBD` | `TBD` | `TBD` |
| `State-Space LM` | 120M | `TBD` | `TBD` | `TBD` | `TBD` |
| `MoE GPT` | 124M(+ experts) | `TBD` | `TBD` | `TBD` | `TBD` |


### Running the Benchmarks

You can reproduce the quality benchmarks yourself using the `lm-evaluation-harness`.

**1. Quality Benchmarks (e.g., HellaSwag):**
```bash
# Make sure your project is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run evaluation for the modern GPT model
lm_eval --model modern_gpt \
        --model_args checkpoint_path=checkpoints/modern/best_checkpoint.pth \
        --tasks hellaswag \
        --device cuda:0 \
        --batch_size 8 \
        --include_path harness_adapter.py
```

**2. Performance Benchmarks (Speed & Memory):**
Performance metrics like training/inference speed and VRAM usage are measured directly within the `train.py` script's logging output and a dedicated inference benchmarking script (to be added).

---

## Roadmap

-   [ ] Integrate with `lm-evaluation-harness` for standardized benchmarking.
-   [ ] Implement KV Caching to accelerate inference.
-   [ ] Add more architectures (e.g., Llama 2, MoE, DeepSeek).
-   [ ] Add support for model quantization (GPTQ, GGUF).

## Acknowledgements

A huge thank you to Andrej Karpathy for his work on [nanoGPT](https://github.com/karpathy/nanogpt), which served as an excellent starting point and a major source of inspiration for this project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
