# MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention
<p align="center">
  <img width="60%" src="flash_monarch.jpg">
</p>

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repo root:

```
uv sync --extra experiments
```

This creates `.venv` with Python 3.12 and the pinned versions in `uv.lock`. Drop `--extra experiments` to install only the core `ma` package (`torch`, `einops`).

To use `ma` from another project: `pip install git+https://github.com/cjyaras/monarch-attention`.

## Usage

Run the tests (GPU tests are skipped without CUDA):

```
uv run pytest ma/tests
```

Run the ViT benchmark:

```
uv run python -m experiments.vit.benchmark
```

See the READMEs in [`experiments/`](experiments) for the other benchmarks.

