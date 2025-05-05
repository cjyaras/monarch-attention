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


def get_color_from_string(s: str) -> tuple[float, float, float]:
    """Generates a deterministic color based on a string hash."""
    hash_object = hashlib.md5(s.encode())
    hash_digest = hash_object.hexdigest()
    # Use the first 6 hex digits for color (RRGGBB)
    hex_color = f"#{hash_digest[:6]}"
    # Convert hex to RGB tuple (0-1 range)
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore


def plot_results(
    results: list[dict],
    metric_name: str = "accuracy",
    title: str = "",
    y_break_limits: (
        Tuple[float, float] | None
    ) = None,  # (lower_bound, upper_bound) for the break
    y_break_vspace: float = 0.05,  # Controls the visual vertical space between broken axes
):
    """Plots a metric vs. FLOPs, optionally with a broken y-axis."""

    plotted_types = {}  # To store handles for the legend
    all_quality = [r["result"][metric_name] for r in results]
    min_quality = 0
    max_quality = 100
    # min_quality = min(all_quality) if all_quality else 0
    # max_quality = max(all_quality) if all_quality else 1
    quality_range = max_quality - min_quality
    padding = quality_range * 0.05  # Add 5% padding

    # --- Plotting Logic ---
    break_needed = False
    if y_break_limits is not None and len(y_break_limits) == 2:
        y_break_lower, y_break_upper = y_break_limits
        # Check if break is valid and actually within the data range (excluding padding)
        if (
            y_break_lower < y_break_upper
            and min_quality < y_break_lower
            and y_break_upper < max_quality
        ):
            break_needed = True

    if not break_needed:
        # --- Standard Plot (No Break) ---
        fig = plt.figure()  # Create figure here
        ax = fig.add_subplot(111)
        axes_list = [ax]
        ax.set_ylim(min_quality - padding, max_quality + padding)
        ax.set_xlabel("Total Attention FLOPs")
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6)

    else:
        # --- Broken Axis Plot ---
        y_break_lower, y_break_upper = y_break_limits  # type: ignore[misc] # Already checked validity
        # Calculate height ratios based on the visible data ranges
        # Add a small epsilon to prevent zero ratios if data points are exactly at the break limits
        eps = 1e-9
        height_top = max(eps, max_quality - y_break_upper)
        height_bottom = max(eps, y_break_lower - min_quality)
        gs_kw = dict(height_ratios=[height_top, height_bottom])
        # Create figure and subplots here
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, gridspec_kw=gs_kw)
        fig.subplots_adjust(hspace=y_break_vspace)  # Adjust space between axes
        axes_list = [ax_top, ax_bottom]

        # Set the y-limits for the two axes, creating the break
        ax_top.set_ylim(y_break_upper, 100.0)
        ax_bottom.set_ylim(0.0, y_break_lower)

        # Hide spines and ticks connecting the break
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.xaxis.tick_top()  # Moves x-ticks to top, but we'll hide them
        ax_top.tick_params(
            labeltop=False, bottom=False
        )  # Hide top x-axis labels and ticks
        ax_bottom.xaxis.tick_bottom()

        # Add break marks (diagonal lines)
        d = 0.015  # Size of diagonal lines in axes coordinates
        # Calculate the ratio of heights to correctly scale the diagonal lines' vertical extent
        # Need to get actual figure positions after layout adjustments
        fig.canvas.draw()  # Ensure layout is computed
        pos_bottom = ax_bottom.get_position()
        pos_top = ax_top.get_position()
        # Add check for zero height to prevent division by zero
        height_ratio_bottom_to_top = (
            pos_bottom.height / pos_top.height if pos_top.height > 1e-6 else 1.0
        )

        kwargs_top = dict(
            transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1
        )
        ax_top.plot(
            (-d, +d),
            (-d * height_ratio_bottom_to_top, +d * height_ratio_bottom_to_top),
            **kwargs_top,
        )  # bottom-left diagonal
        ax_top.plot(
            (1 - d, 1 + d),
            (-d * height_ratio_bottom_to_top, +d * height_ratio_bottom_to_top),
            **kwargs_top,
        )  # bottom-right diagonal

        kwargs_bottom = dict(
            transform=ax_bottom.transAxes, color="k", clip_on=False, linewidth=1
        )
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs_bottom)  # top-left diagonal
        ax_bottom.plot(
            (1 - d, 1 + d), (1 - d, 1 + d), **kwargs_bottom
        )  # top-right diagonal

        # Set labels and title
        ax_bottom.set_xlabel("Total Attention FLOPs")
        fig.text(
            0.04, 0.5, metric_name.capitalize(), va="center", rotation="vertical"
        )  # Common Y label
        ax_top.set_title(title)
        ax_top.grid(True, linestyle="--", alpha=0.6)
        ax_bottom.grid(True, linestyle="--", alpha=0.6)

    # Sort results by attention type alphabetical
    results = sorted(results, key=lambda x: x["attention_type"])

    softmax_flops = 0

    for result in results:
        flops = result["result"]["total_attention_bmm_flops"]
        attention_type = result["attention_type"]
        quality = result["result"][metric_name]
        color = get_color_from_string(attention_type)
        label = attention_type if attention_type not in plotted_types else ""

        if attention_type == "softmax":
            softmax_flops = flops

        # Plot on all relevant axes
        for ax in axes_list:
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

    # Add legend (use handles from the last axis they were plotted on)
    if plotted_types:  # Only add legend if there are items to plot
        if not break_needed:
            # Standard plot: Legend on the single axis
            axes_list[0].legend(
                handles=plotted_types.values(), loc="best", fontsize="small"
            )
        else:
            # Broken axis plot: Place legend outside the axes, to the right

            # Adjust subplot parameters to make space for the legend on the right
            # You might need to tweak the 'right' value (e.g., 0.8, 0.75) depending on label lengths
            fig.subplots_adjust(right=0.78)

            fig.legend(
                handles=plotted_types.values(),
                labels=plotted_types.keys(),  # Use keys for labels with fig.legend
                loc="center left",  # Anchor point on the legend box
                bbox_to_anchor=(0.8, 0.5),
                fontsize="small",
            )  # Position the anchor point (x, y) relative to figure (1=right edge, 0.5=center vertically)

    # ax.set_xlim(0, softmax_flops * 1.1)
    return fig
