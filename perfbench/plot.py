"""Plot the sequence length sweep measured by perfbench/bench.py.

python -m perfbench.plot [--csv perfbench/results/runtime.csv]
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans Mono", "font.size": 14})

COLORS = {"monarch-attention": "#E69F00", "softmax": "#009E73"}
LABELS = {"monarch-attention": "monarch-attention", "softmax": "flash-attention-2"}
WIDTH = 0.35


def load(path: str):
    """Sequence lengths, runtimes per method, and the batch size of the sweep."""
    times = defaultdict(dict)
    batch = None
    for row in csv.DictReader(open(path)):
        if row["sweep"] == "seq_len":
            times[row["method"]][int(row["seq_len"])] = float(row["ms"])
            batch = row["batch"]
    seq_lens = sorted(times["monarch-attention"])
    return (
        seq_lens,
        {m: np.array([t[n] for n in seq_lens]) for m, t in times.items()},
        batch,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default="perfbench/results/runtime.csv")
    args = parser.parse_args()
    out_dir = os.path.dirname(args.csv)
    seq_lens, times, batch = load(args.csv)

    # Paper figure: runtimes divided by the slower method at each sequence length
    slowest = np.maximum(times["monarch-attention"], times["softmax"])
    x = np.arange(len(seq_lens))
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, method in [(-WIDTH / 2, "monarch-attention"), (WIDTH / 2, "softmax")]:
        ax.bar(
            x + offset,
            times[method] / slowest,
            WIDTH,
            label=LABELS[method],
            color=COLORS[method],
            edgecolor="black",
            linewidth=1.5,
        )
    ax.set_title(f"Normalized Runtime vs. Sequence Length ($E = {batch}$)")
    ax.set_xlabel("Sequence Length ($N$)")
    ax.set_ylabel("Normalized Runtime")
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lens)
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(
        os.path.join(out_dir, "normalized_attention_runtime.pdf"), bbox_inches="tight"
    )

    # Absolute runtimes
    plt.figure(figsize=(10, 6))
    plt.plot(
        seq_lens, times["monarch-attention"], label="monarch_attention", marker="o"
    )
    plt.plot(seq_lens, times["softmax"], label="flash_attention_2", marker="s")
    plt.xscale("log", base=2)
    plt.xlabel("Sequence Length (log scale)")
    plt.ylabel("Runtime (ms)")
    plt.title("Attention Runtime vs. Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"attention_runtime_vs_sequence_length_batch_{batch}.pdf")
    )
    print(f"Wrote figures to {out_dir}/")


if __name__ == "__main__":
    main()
