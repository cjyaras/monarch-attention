# MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention

Code for [MonarchAttention](https://arxiv.org/abs/2505.18698) (NeurIPS 2025).

## Speed

Forward pass of MonarchAttention's Triton kernels vs softmax attention: fp16, batch size 1,
12 heads, head dimension 64, one step (`num_steps=1`) and block size a power of two near √N.

NVIDIA H100 80 GB, vs [FlashAttention-4](https://github.com/Dao-AILab/flash-attention) and
PyTorch's `scaled_dot_product_attention` (cuDNN):

| Sequence length N | FlashAttention-4 | PyTorch SDPA | MonarchAttention | Speedup vs FA4 |
|---:|---:|---:|---:|---:|
| 1,024 | 0.020 ms | 0.019 ms | 0.015 ms | 1.3x |
| 2,048 | 0.032 ms | 0.045 ms | 0.021 ms | 1.5x |
| 4,096 | 0.104 ms | 0.123 ms | 0.031 ms | 3.3x |
| 8,192 | 0.390 ms | 0.447 ms | 0.057 ms | 6.9x |
| 16,384 | 1.586 ms | 1.841 ms | 0.105 ms | 15x |
| 32,768 | 6.248 ms | 7.350 ms | 0.218 ms | 29x |

NVIDIA A100 80 GB, vs PyTorch's `scaled_dot_product_attention` (FlashAttention-2):

| Sequence length N | FlashAttention-2 | MonarchAttention | Speedup |
|---:|---:|---:|---:|
| 1,024 | 0.034 ms | 0.022 ms | 1.6x |
| 2,048 | 0.091 ms | 0.030 ms | 3.1x |
| 4,096 | 0.325 ms | 0.051 ms | 6.4x |
| 8,192 | 1.243 ms | 0.102 ms | 12x |
| 16,384 | 4.590 ms | 0.190 ms | 24x |
| 32,768 | 17.590 ms | 0.417 ms | 42x |

These are GPU kernel times. For short sequences, Python launch overhead dominates in eager
mode (MonarchAttention launches two Triton kernels per call), so run the model under CUDA
graphs, e.g. `torch.compile(model, mode="reduce-overhead")`. Reproduce with
[`perfbench/`](perfbench) (`--sweep seq_len --max-seq-len 32768`).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```
uv sync --extra experiments
```

This creates `.venv` with Python 3.14 and the pinned versions in `uv.lock`. Drop `--extra experiments` to install only the core `ma` package (just `torch`).

To use `ma` from another project: `pip install git+https://github.com/cjyaras/monarch-attention`.

## Usage

`MonarchAttention` is a drop-in replacement for softmax attention:

```python
import torch
from ma import MonarchAttention, PadType

attn = MonarchAttention(block_size=16, num_steps=2, pad_type=PadType.pre, impl="triton")
q, k, v = (
    torch.randn(2, 12, 1024, 64, device="cuda", dtype=torch.float16) for _ in range(3)
)
mask = torch.ones(2, 1024, dtype=torch.bool, device="cuda")  # optional, True = keep
out = attn(q, k, v, mask)  # (2, 12, 1024, 64)
```

Use `impl="torch"` for the reference implementation, which also runs on CPU.

Options for the Triton kernels:

- `max_workspace` (default 256 MiB): the most bytes of intermediate buffers per call. They
  take ~2x the output's memory, so larger inputs are processed a group of batch elements,
  heads or positions at a time (a few percent slower). `None` removes the limit.
- `fp8=True`: store the intermediate buffers in FP8 with per-row scales, halving their memory
  with no measurable accuracy change on the ViT and RoBERTa benchmarks. Needs FP8 support
  (e.g. an H100); up to 1.17x faster at 1K-16K tokens, a few percent slower at 32K.

Run the tests (CUDA-only tests are skipped without a GPU):

```
uv run pytest
```

Lint and format:

```
uvx ruff check .
uvx ruff format .
```

Run the ViT benchmark:

```
uv run python -m experiments.vit.benchmark
```

See the READMEs in [`experiments/`](experiments) for the other benchmarks, and
[`perfbench/`](perfbench) for runtime comparisons against FlashAttention-2.

## License

Academic research and education use only; see [`LICENSE`](LICENSE).

## Citation

```bibtex
@inproceedings{yaras2025monarchattention,
  title     = {MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention},
  author    = {Yaras, Can and Xu, Alec and Abillama, Pierre and Lee, Changwoo and Balzano, Laura},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  pages     = {3445--3471},
  year      = {2025}
}
```
