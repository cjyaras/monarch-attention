import json
import os

import matplotlib.pyplot as plt

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


def hex_color_to_tuple(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))  # type: ignore


colors = {
    "monarch-attention": hex_color_to_tuple("#E69F00"),
    "performer": hex_color_to_tuple("#CC79A7"),
    "cosformer": hex_color_to_tuple("#56B4E9"),
    "linear-attention": hex_color_to_tuple("#D55E00"),
    "nystromformer": hex_color_to_tuple("#7AA4BD"),
    "softmax": hex_color_to_tuple("#009E73"),
}


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


def main():
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))

    # ViT plot
    SAVE_DIR = "experiments/vit/results"
    results = []
    for result_string in os.listdir(SAVE_DIR):
        with open(os.path.join(SAVE_DIR, result_string), "r") as f:
            result = json.load(f)
            results.append(result)

    plotted_types = plot_results(
        axes[0], results, metric_name="top-5 accuracy", title="ViT ImageNet"
    )

    # RoBERTa plot

    SAVE_DIR = "experiments/roberta/results"
    results = []
    for result_string in os.listdir(SAVE_DIR):
        with open(os.path.join(SAVE_DIR, result_string), "r") as f:
            result = json.load(f)
            results.append(result)

    plotted_types = plot_results(
        axes[1], results, metric_name="f1", title="RoBERTa SQuAD"
    )

    fig.subplots_adjust(right=0.78)
    fig.legend(
        handles=plotted_types.values(),
        labels=plotted_types.keys(),  # Use keys for labels with fig.legend
        loc="center left",  # Anchor point on the legend box
        bbox_to_anchor=(0.8, 0.5),
        # fontsize="small",
    )  # Position the anchor point (x, y) relative to figure (1=right edge, 0.5=center vertically)

    fig.savefig("figures/vit_roberta_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
