# Runtime benchmarks

Compares the MonarchAttention Triton kernels with softmax attention
(`F.scaled_dot_product_attention`, i.e. FlashAttention-2) on a CUDA GPU. From the repo root:

```
uv run python -m perfbench.bench --sweep seq_len   # measure, write perfbench/results/runtime.csv
uv run python -m perfbench.plot                    # figures next to the CSV
```

- `--sweep seq_len`: sequence lengths 1024 to 16384 at batch size `--batch` (1), producing
  the paper figure `normalized_attention_runtime.pdf` and absolute runtimes.
- `--sweep single` (default): one configuration, set with `--batch` and `--seq-len`.

Also configurable: `--heads` (12), `--head-dim` (64), `--num-steps` (1) and `--dtype`
(`fp16`, `bf16`). Monarch uses block size √N. Before timing, each configuration checks the
kernel against the torch reference implementation. The CSV records runtime (ms) and peak
GPU memory (MB), with the GPU and host: absolute runtimes can differ by 2x between nodes with
the same GPU model, so compare methods within one run.

## Profiling

`--profile nsys` / `--profile ncu` run each kernel for a profiler instead of timing it:

```
nsys profile -o monarch uv run python -m perfbench.bench --profile nsys
ncu --set full -o monarch uv run python -m perfbench.bench --profile ncu
```
