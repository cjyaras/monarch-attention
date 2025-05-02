import hashlib

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans Mono",
        "font.size": 12,
    }
)


def get_color_from_string(s: str) -> tuple[float, float, float]:
    """Generates a deterministic color based on a string hash."""
    hash_object = hashlib.md5(s.encode())
    hash_digest = hash_object.hexdigest()
    # Use the first 6 hex digits for color (RRGGBB)
    hex_color = f"#{hash_digest[:6]}"
    # Convert hex to RGB tuple (0-1 range)
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore


def plot_results(results: list[dict], metric_name: str = "accuracy", title: str = ""):
    """
    Plots a metric vs. FLOPs for different attention types.

    Args:
        results: A list of dictionaries, each containing 'attention_type',
                 'total_attention_bmm_flops', and the metric specified by metric_name.
        metric_name: The name of the metric to plot on the y-axis.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plotted_types = {}  # To store handles for the legend

    for result in results:
        flops = result["result"]["total_attention_bmm_flops"]
        attention_type = result["attention_type"]
        quality = result["result"][metric_name]
        color = get_color_from_string(attention_type)

        # Plot scatter point
        scatter = ax.scatter(
            flops,
            quality,
            label=attention_type if attention_type not in plotted_types else "",
            color=color,
            marker="o",
            s=150,  # Large size
            edgecolor="black",
            linewidth=1.5,  # Thick border
        )
        if attention_type not in plotted_types:
            plotted_types[attention_type] = scatter

    ax.set_xlabel("Total Attention FLOPs")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(title)
    ax.legend(handles=plotted_types.values())
    ax.grid(True, linestyle="--", alpha=0.6)

    return fig
