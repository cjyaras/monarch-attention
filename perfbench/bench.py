import argparse
import logging
from enum import Enum
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from flash_attention import flash_attention
from flash_monarch_attention import (  # flash_monarch_attention,
    flash_monarch_attention_reference,
)
from flash_monarch_attention_v2 import (
    flash_monarch_attention_v2 as flash_monarch_attention,
)

# from fused_flash_monarch_attention import fused_flash_monarch_attention
from fused_flash_monarch_attention_v2 import (
    fused_flash_monarch_attention as fused_flash_monarch_attention,
)
from triton.testing import do_bench

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans Mono",
        "font.size": 14,
    }
)


def hex_color_to_tuple(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


colors = {
    "monarch-attention": hex_color_to_tuple("#E69F00"),
    "softmax": hex_color_to_tuple("#009E73"),
}


def run_and_plot_attention_sweeps(num_heads, seq_len, d, T):
    run_mode = RunMode.NORMAL
    batch_sizes = [2**i for i in range(3, 14)]
    seq_lens = [2**i for i in range(10, 15)]

    normalized_batch = {"monarch-attention": [], "softmax": []}
    normalized_seq = {"monarch-attention": [], "softmax": []}

    for batch in batch_sizes:
        q = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        k = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        attn_mask = None
        b = int(sqrt(seq_len))

        try:
            t1 = benchmark(run_mode, fused_flash_monarch_attention, q, k, v, b, T)
        except:
            t1 = float("nan")

        try:
            t2 = benchmark(run_mode, flash_attention, q, k, v, attn_mask)
        except:
            t2 = float("nan")

        try:
            t3 = benchmark(run_mode, F.scaled_dot_product_attention, q, k, v)
        except:
            t3 = float("nan")

        if all(np.isnan(t) for t in [t1, t2, t3]):
            normalized_batch["monarch-attention"].append(np.nan)
            normalized_batch["softmax"].append(np.nan)
        else:
            best_softmax = min(t for t in [t2, t3] if not np.isnan(t))
            max_t = max(t1, best_softmax)
            normalized_batch["monarch-attention"].append(t1 / max_t)
            normalized_batch["softmax"].append(best_softmax / max_t)

    batch = 1

    for seq_len in seq_lens:
        q = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        k = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        attn_mask = None
        b = int(sqrt(seq_len))
        pre_pad = False

        try:
            t1 = benchmark(
                run_mode, flash_monarch_attention, q, k, v, attn_mask, T, b, pre_pad
            )
        except:
            t1 = float("nan")

        try:
            t2 = benchmark(run_mode, flash_attention, q, k, v, attn_mask)
        except:
            t2 = float("nan")

        try:
            t3 = benchmark(run_mode, F.scaled_dot_product_attention, q, k, v)
        except:
            t3 = float("nan")

        if all(np.isnan(t) for t in [t1, t2, t3]):
            normalized_seq["monarch-attention"].append(np.nan)
            normalized_seq["softmax"].append(np.nan)
        else:
            best_softmax = min(t for t in [t2, t3] if not np.isnan(t))
            max_t = max(t1, best_softmax)
            normalized_seq["monarch-attention"].append(t1 / max_t)
            normalized_seq["softmax"].append(best_softmax / max_t)

    fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
    ax2, ax1 = axes

    width = 0.35
    x1 = np.arange(len(batch_sizes))
    x2 = np.arange(len(seq_lens))

    bar1 = ax1.bar(
        x1 - width / 2,
        normalized_batch["monarch-attention"],
        width,
        label="monarch-attention",
        color=colors["monarch-attention"],
        edgecolor="black",
        linewidth=1.5,
    )
    bar2 = ax1.bar(
        x1 + width / 2,
        normalized_batch["softmax"],
        width,
        label="softmax",
        color=colors["softmax"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_title("Normalized Runtime vs. Batch Size ($N = 256$)")
    ax1.set_xlabel("Batch Size ($E$)")
    ax1.set_ylabel("Normalized Runtime")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(batch_sizes)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

    bar3 = ax2.bar(
        x2 - width / 2,
        normalized_seq["monarch-attention"],
        width,
        label="monarch-attention",
        color=colors["monarch-attention"],
        edgecolor="black",
        linewidth=1.5,
    )
    bar4 = ax2.bar(
        x2 + width / 2,
        normalized_seq["softmax"],
        width,
        label="softmax",
        color=colors["softmax"],
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.set_title("Normalized Runtime vs. Sequence Length ($E = 1$)")
    ax2.set_xlabel("Sequence Length ($N$)")
    ax2.set_ylabel("Normalized Runtime")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(seq_lens)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)

    fig.subplots_adjust(right=0.75)
    fig.legend(
        handles=[bar1, bar2],
        labels=["monarch-attention", "flash-attention-2"],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    plt.tight_layout()
    fig.savefig("normalized_attention_runtime_side_by_side.pdf", bbox_inches="tight")
    plt.show()


class RunMode(Enum):
    NSYS = 0
    NCU = 1
    NORMAL = 2


def benchmark(run_mode, func, *args):
    if run_mode == RunMode.NORMAL:
        time = do_bench(lambda: func(*args))
        print(f"{func.__name__} time: {time}")
        return time
    elif run_mode == RunMode.NCU:
        o = func(*args)
    elif run_mode == RunMode.NSYS:
        num_iters = 20
        num_warmup_iters = 10
        for i in range(num_iters):
            if i == num_warmup_iters:
                torch.cuda.cudart().cudaProfilerStart()
            if i >= num_warmup_iters:
                torch.cuda.nvtx.range_push(f"{func.__name__} Iteration {i}")
            func(*args)
            if i >= num_warmup_iters:
                torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()


def sweep_batch(num_heads, seq_len, d, T):
    run_mode = RunMode.NORMAL
    batch_sizes = [2**i for i in range(3, 14)]

    flash_monarch_times = []
    flash_attn_times = []

    for batch in batch_sizes:
        print(f"\nBatch size: {batch}")
        q = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        k = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        attn_mask = None
        b = int(sqrt(seq_len))

        try:
            t1 = benchmark(run_mode, fused_flash_monarch_attention, q, k, v, b, T)
        except Exception as e:
            print(f"flash_monarch_attention failed: {e}")
            t1 = float("nan")

        try:
            t2 = benchmark(run_mode, flash_attention, q, k, v, attn_mask)
        except Exception as e:
            print(f"flash_attention failed: {e}")
            t2 = float("nan")

        try:
            t3 = benchmark(run_mode, F.scaled_dot_product_attention, q, k, v)
        except Exception as e:
            print(f"scaled_dot_product_attention failed: {e}")
            t3 = float("nan")

        best_flash_time = np.nanmin([t2, t3])
        flash_monarch_times.append(t1)
        flash_attn_times.append(best_flash_time)

    plt.figure(figsize=(10, 6))
    plt.plot(
        batch_sizes, flash_monarch_times, label="flash_monarch_attention", marker="o"
    )
    plt.plot(batch_sizes, flash_attn_times, label="best_flash_attention", marker="s")
    plt.xscale("log", base=2)
    plt.xlabel("Batch Size (log scale)")
    plt.ylabel("Runtime (ms)")
    plt.title("Attention Runtime vs. Batch Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"attention_runtime_vs_batchsize_seq_{seq_len}.pdf")

    normalized_monarch = []
    normalized_flash = []

    for t1, t_flash in zip(flash_monarch_times, flash_attn_times):
        values = [t1, t_flash]
        if any(np.isnan(values)):
            normalized_monarch.append(np.nan)
            normalized_flash.append(np.nan)
        else:
            max_val = max(values)
            normalized_monarch.append(t1 / max_val)
            normalized_flash.append(t_flash / max_val)

    x = np.arange(len(batch_sizes))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, normalized_monarch, width, label="MonarchAttention")
    plt.bar(x + width / 2, normalized_flash, width, label="Softmax Attention")

    plt.xlabel("Batch Size")
    plt.ylabel("Normalized Runtime (0 = fastest, 1 = slowest)")
    plt.title("Normalized Attention Runtime by Batch Size")
    plt.xticks(ticks=x, labels=batch_sizes, rotation=45)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"normalized_attention_runtime_vs_batchsize_seq_{seq_len}.pdf")


def sweep_seq_len(batch, num_heads, d, T):
    run_mode = RunMode.NORMAL
    seq_lens = [2**i for i in range(10, 15)]

    flash_monarch_times = []
    flash_attn_times = []

    for seq_len in seq_lens:
        print(f"\nSequence length: {seq_len}")
        q = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        k = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        v = torch.randn(
            batch, num_heads, seq_len, d, dtype=torch.float16, device="cuda"
        )
        attn_mask = None
        pre_pad = False
        b = int(sqrt(seq_len))

        try:
            t1 = benchmark(
                run_mode, flash_monarch_attention, q, k, v, attn_mask, T, b, pre_pad
            )
        except Exception as e:
            print(f"flash_monarch_attention failed: {e}")
            t1 = float("nan")

        try:
            t2 = benchmark(run_mode, flash_attention, q, k, v, attn_mask)
        except Exception as e:
            print(f"flash_attention failed: {e}")
            t2 = float("nan")

        try:
            t3 = benchmark(run_mode, F.scaled_dot_product_attention, q, k, v)
        except Exception as e:
            print(f"scaled_dot_product_attention failed: {e}")
            t3 = float("nan")

        best_flash_time = np.nanmin([t2, t3])
        flash_monarch_times.append(t1)
        flash_attn_times.append(best_flash_time)

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, flash_monarch_times, label="flash_monarch_attention", marker="o")
    plt.plot(seq_lens, flash_attn_times, label="best_flash_attention", marker="s")
    plt.xscale("log", base=2)
    plt.xlabel("Sequence Length (log scale)")
    plt.ylabel("Runtime (ms)")
    plt.title("Attention Runtime vs. Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"attention_runtime_vs_sequence_length_batch_{batch}.pdf")

    normalized_monarch = []
    normalized_flash = []

    for t1, t_flash in zip(flash_monarch_times, flash_attn_times):
        values = [t1, t_flash]
        if any(np.isnan(values)):
            normalized_monarch.append(np.nan)
            normalized_flash.append(np.nan)
        else:
            max_val = max(values)
            normalized_monarch.append(t1 / max_val)
            normalized_flash.append(t_flash / max_val)

    x = np.arange(len(seq_lens))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, normalized_monarch, width, label="MonarchAttention")
    plt.bar(x + width / 2, normalized_flash, width, label="Softmax Attention")
    plt.xlabel("Sequence Length")
    plt.ylabel("Normalized Runtime (0 = fastest, 1 = slowest)")
    plt.title("Normalized Attention Runtime by Sequence Length")
    plt.xticks(ticks=x, labels=seq_lens, rotation=45)
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(f"normalized_attention_runtime_vs_sequence_length_batch_{batch}.pdf")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_mode",
        type=lambda mode: RunMode[mode.upper()],
        choices=list(RunMode),
        default=RunMode.NORMAL,
        help="Choose the run mode: NSYS, NCU, or NORMAL",
    )
    parser.add_argument(
        "--sweep_batch", action="store_true", help="Run the script to sweep batch size"
    )
    parser.add_argument(
        "--sweep_seq_len",
        action="store_true",
        help="Run the script to sweep sequence length",
    )
    parser.add_argument(
        "--sweep_all",
        action="store_true",
        help="Run the script to sweep sequence length and batch size",
    )

    args = parser.parse_args()
    run_mode = args.run_mode

    batch = 1
    seq_len = 256
    num_heads = 12
    d = 64
    T = 1

    if args.sweep_batch:
        sweep_batch(num_heads, seq_len, d, T)
    elif args.sweep_seq_len:
        sweep_seq_len(batch, num_heads, d, T)
    elif args.sweep_all:
        run_and_plot_attention_sweeps(num_heads, seq_len, d, T)
    else:
        q = torch.randn(batch, num_heads, seq_len, d, dtype=torch.float16).cuda()
        k = torch.randn(batch, num_heads, seq_len, d, dtype=torch.float16).cuda()
        v = torch.randn(batch, num_heads, seq_len, d, dtype=torch.float16).cuda()
        attn_mask = None
        pre_pad = False
        b = int(sqrt(seq_len))

        try:
            benchmark(
                run_mode, flash_monarch_attention, q, k, v, attn_mask, T, b, pre_pad
            )
        except Exception as e:
            print(f"flash_monarch_attention failed: {e}")

        # try:
        #     benchmark(run_mode, fused_flash_monarch_attention, q, k, v, b, T)
        # except Exception as e:
        #     print(f"fused_flash_monarch_attention failed: {e}")

        # try:
        #     benchmark(run_mode, flash_attention, q, k, v, attn_mask)
        # except Exception as e:
        #     print(f"flash_attention failed: {e}")

        try:
            benchmark(run_mode, F.scaled_dot_product_attention, q, k, v)
        except Exception as e:
            print(f"scaled_dot_product_attention failed: {e}")
