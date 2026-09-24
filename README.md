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
| 4,096 | 0.103 ms | 0.117 ms | 0.031 ms | 3.4x |
| 8,192 | 0.394 ms | 0.471 ms | 0.056 ms | 7.0x |
| 16,384 | 1.532 ms | 1.813 ms | 0.105 ms | 15x |

NVIDIA A100 80 GB, vs PyTorch's `scaled_dot_product_attention` (FlashAttention-2):

| Sequence length N | FlashAttention-2 | MonarchAttention | Speedup |
|---:|---:|---:|---:|
| 1,024 | 0.034 ms | 0.022 ms | 1.5x |
| 2,048 | 0.090 ms | 0.029 ms | 3.1x |
| 4,096 | 0.323 ms | 0.050 ms | 6.4x |
| 8,192 | 1.239 ms | 0.100 ms | 12x |
| 16,384 | 4.576 ms | 0.189 ms | 24x |

These are GPU kernel times. For short sequences, Python launch overhead dominates in eager
mode (MonarchAttention launches two Triton kernels per call), so run the model under CUDA
graphs, e.g. `torch.compile(model, mode="reduce-overhead")`. Reproduce with
[`perfbench/`](perfbench).

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
