"""Benchmark MonarchAttention's Triton kernels against softmax attention: PyTorch's
scaled_dot_product_attention (SDPA), and FlashAttention-4 if it is installed.

Writes one CSV row per (sweep, batch size, sequence length, method) with the
runtime and peak GPU memory; perfbench/plot.py turns the CSV into figures.

    python -m perfbench.bench --sweep seq_len        # the sweep in the paper figure
    python -m perfbench.bench --sweep single         # one configuration, e.g. to profile
"""

import argparse
import csv
import os
import socket

import torch
import torch.nn.functional as F
from triton.testing import do_bench

from ma.ma_torch import monarch_attention_torch
from ma.ma_triton import monarch_attention_triton

try:
    from flash_attn.cute import flash_attn_func as flash_attn_4
except ImportError:  # optional: pip install flash-attn-4 (Hopper and Blackwell GPUs)
    flash_attn_4 = None

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def block_size_for(seq_len: int) -> int:
    """Power of two near sqrt(seq_len), which the kernels' tiles divide evenly."""
    return 2 ** (seq_len.bit_length() - 1 >> 1)


def measure(fn, profile: str) -> tuple[float, float]:
    """Runtime in ms and peak memory in MB of `fn()`."""
    if profile == "ncu":  # the profiler replays kernels itself
        fn()
        return float("nan"), float("nan")
    if profile == "nsys":
        for _ in range(10):  # warmup
            fn()
        torch.cuda.synchronize()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        return float("nan"), float("nan")

    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak_mb = (torch.cuda.max_memory_allocated() - base) / 2**20
    return do_bench(fn), peak_mb


def benchmark(batch: int, seq_len: int, args) -> list[dict]:
    dtype = DTYPES[args.dtype]
    q, k, v = (
        torch.randn(
            batch, args.heads, seq_len, args.head_dim, dtype=dtype, device="cuda"
        )
        for _ in range(3)
    )
    block_size = block_size_for(seq_len)

    def monarch(q=q, k=k, v=v):
        return monarch_attention_triton(
            q, k, v, None, args.num_steps, block_size, False
        )

    def softmax():
        return F.scaled_dot_product_attention(q, k, v)

    # Check the kernel against the reference implementation on one sequence
    expected = monarch_attention_torch(
        q[:1], k[:1], v[:1], None, args.num_steps, block_size, False
    )
    torch.testing.assert_close(
        monarch(q[:1], k[:1], v[:1]), expected, atol=2e-2, rtol=2e-2
    )

    methods = [("monarch-attention", monarch), ("softmax", softmax)]
    if flash_attn_4 is not None:
        # FlashAttention-4 takes (batch, seq_len, heads, head_dim)
        q4, k4, v4 = (t.transpose(1, 2).contiguous() for t in (q, k, v))

        def fa4():
            return flash_attn_4(q4, k4, v4)[0]  # (output, log-sum-exp)

        torch.testing.assert_close(
            fa4().transpose(1, 2), softmax(), atol=2e-2, rtol=2e-2
        )
        methods.append(("flash-attention-4", fa4))

    rows = []
    for method, fn in methods:
        ms, peak_mb = measure(fn, args.profile)
        rows.append(
            {
                "gpu": torch.cuda.get_device_name(),
                "host": socket.gethostname(),
                "batch": batch,
                "seq_len": seq_len,
                "method": method,
                "ms": ms,
                "peak_mb": peak_mb,
            }
        )
        print(
            f"batch={batch:5d} seq_len={seq_len:6d} {method:18s} {ms:8.3f} ms {peak_mb:9.1f} MB"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", choices=["single", "seq_len"], default="single")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="sequence length for --sweep single",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16384,
        help="longest sequence length for --sweep seq_len (the paper figure: 16384)",
    )
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    parser.add_argument(
        "--profile",
        choices=["none", "nsys", "ncu"],
        default="none",
        help="run each kernel for a profiler instead of timing it",
    )
    parser.add_argument("--out", default="perfbench/results/runtime.csv")
    args = parser.parse_args()

    configs = [("single", args.batch, args.seq_len)]
    if args.sweep == "seq_len":
        seq_lens = [n for n in (2**i for i in range(10, 21)) if n <= args.max_seq_len]
        configs = [("seq_len", args.batch, n) for n in seq_lens]

    rows = []
    for sweep, batch, seq_len in configs:
        rows += [{"sweep": sweep, **row} for row in benchmark(batch, seq_len, args)]
        torch.cuda.empty_cache()  # release the previous configuration's inputs

    if args.profile == "none":
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
