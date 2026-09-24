# MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention

Code for [MonarchAttention](https://arxiv.org/abs/2505.18698) (NeurIPS 2025).

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
