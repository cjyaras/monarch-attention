# Monarch Attention for DiT

Testing Monarch Attention on class-conditional image generation using the [DiT-XL/2-256](https://huggingface.co/facebook/DiT-XL-2-256) model on ImageNet. No training is required — the pretrained diffusion model is used directly.

## FLOP Benchmarking

`benchmark_flops.py` measures the floating-point operations for a single generation step across attention mechanisms.

```bash
python -m experiments.dit.benchmark_flops \
    --attention_type monarch \
    --monarch_block_size 16 \
    --monarch_num_steps 3 \
    --num_inference_steps 1
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--attention_type` | `softmax` | `softmax`, `monarch`, `linformer`, `performer`, `nystromformer`, `cosformer` |
| `--monarch_block_size` | `16` | Block size for Monarch Attention |
| `--monarch_num_steps` | `3` | Number of steps for Monarch Attention |
| `--rank` | `64` | Rank for low-rank attention methods |
| `--replace_all_layers` | `False` | Replace all layers (default: first half only) |
| `--num_inference_steps` | `1` | Number of diffusion steps |

## FID Sampling

`sample_fid.py` generates 50K images for FID evaluation, saving batches as `.npz` files.

```bash
python -m experiments.dit.sample_fid \
    --attention_type monarch \
    --monarch_block_size 16 \
    --monarch_num_steps 3 \
    --num_samples 50000 \
    --num_inference_steps 32 \
    --batch_size 32 \
    --save_dir ./experiments/dit/generations
```

Key arguments are the same as `benchmark_flops.py`, plus:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_samples` | `50000` | Total images to generate |
| `--batch_size` | `32` | Generation batch size |
| `--num_inference_steps` | `32` | Number of diffusion steps |
| `--save_dir` | (see script) | Output directory for `.npz` files |

## Visualization

`visualize_examples.py` generates a grid of sample images for visual comparison.

```bash
python -m experiments.dit.visualize_examples \
    --attention_type monarch \
    --monarch_block_size 16 \
    --monarch_num_steps 3 \
    --num_samples 36 \
    --num_inference_steps 32 \
    --save_dir ./experiments/dit/generations
```

Output is saved as a single PNG (e.g. `monarch_second_half_layers.png`) in `--save_dir`.
