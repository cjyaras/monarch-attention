import hashlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans Mono",
        "font.size": 12,
    }
)

ordering = {
    "monarch-attention": 0,
    "performer": 1,
    "cosformer": 2,
    "linear-attention": 3,
    "nystromformer": 4,
    "softmax": 5,
}

# assign color for each method
colors = {
    "monarch-attention": "lime",
    "performer": "olive",
    "cosformer": "indianred",
    "linear-attention": "cyan",
    "nystromformer": "purple",
    "softmax": "red",
}


def get_color_from_string(s: str) -> tuple[float, float, float]:
    """Generates a deterministic color based on a string hash."""
    hash_object = hashlib.md5(s.encode())
    hash_digest = hash_object.hexdigest()
    hash_digest = hash_digest[::-1]
    # Use the first 6 hex digits for color (RRGGBB)
    hex_color = f"#{hash_digest[:6]}"
    # Convert hex to RGB tuple (0-1 range)
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore


def plot_results(
    ax, results: list[dict], metric_name: str = "accuracy", title: str = ""
):
    """Plots a metric vs. FLOPs, optionally with a broken y-axis."""

    plotted_types = {}  # To store handles for the legend
    min_quality = 0
    max_quality = 100
    quality_range = max_quality - min_quality
    padding = quality_range * 0.05  # Add 5% padding

    ax.set_ylim(min_quality - padding, max_quality + padding)
    ax.set_xlabel("Total Attention FLOPs")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Sort results by attention type alphabetical
    results = sorted(results, key=lambda x: ordering[x["attention_type"]])

    for result in results:
        flops = result["result"]["total_attention_bmm_flops"]
        attention_type = result["attention_type"]
        quality = result["result"][metric_name]
        # color = get_color_from_string(attention_type)
        color = colors[attention_type]
        label = attention_type if attention_type not in plotted_types else ""

        # Plot on all relevant axes
        scatter = ax.scatter(
            flops,
            quality,
            label=label,
            color=color,
            marker="o",
            s=150,
            edgecolor="black",
            linewidth=1.5,
        )
        if label:  # Only store the handle once
            plotted_types[attention_type] = scatter

    return plotted_types
