import matplotlib.lines as mlines  # For custom legend handles
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans Mono",
        "font.size": 12,
    }
)


def hex_color_to_tuple(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))  # type: ignore


# Original colors map
# colors = {
#     "monarch-attention": hex_color_to_tuple("#332288"),
#     "performer": hex_color_to_tuple("#44AA99"),  # Not used in BART data
#     "cosformer": hex_color_to_tuple("#88CCEE"),  # Not used in BART data
#     "linear-attention": hex_color_to_tuple("#DDCC77"),  # Not used in BART data
#     "nystromformer": hex_color_to_tuple("#AA4499"),
#     "softmax": hex_color_to_tuple("#882255"),
# }

colors = {
    "monarch-attention": hex_color_to_tuple("#E69F00"),
    "performer": hex_color_to_tuple("#CC79A7"),
    "cosformer": hex_color_to_tuple("#56B4E9"),
    "linear-attention": hex_color_to_tuple("#D55E00"),
    "nystromformer": hex_color_to_tuple("#0072B2"),
    "softmax": hex_color_to_tuple("#009E73"),
}

# Filter colors for those used in BART plot and define order for legend
bart_colors = {
    "softmax": colors["softmax"],
    "nystromformer": colors["nystromformer"],
    "monarch-attention": colors["monarch-attention"],
}
# attention_type_order = ["softmax", "nystromformer", "monarch-attention"]
attention_type_order = ["monarch-attention", "nystromformer", "softmax"]


def main():

    softmax_flops = [9.66e9, 38.7e9, 155.0e9, 619.0e9]
    nystrom_flops = [1.93e9, 4.41e9, 10.6e9, 35.0e9]
    monarch_flops = [1.96e9, 3.93e9, 10.9e9, 31.4e9]

    softmax_rouge_1 = [30.08, 33.32, 34.30, 34.76]
    nystrom_rouge_1 = [29.54, 32.13, 32.42, 32.86]
    monarch_rouge_1 = [29.78, 32.56, 33.52, 33.94]

    softmax_rouge_2 = [5.43, 6.65, 7.29, 7.62]
    nystrom_rouge_2 = [5.35, 6.30, 6.79, 7.05]
    monarch_rouge_2 = [5.38, 6.41, 6.94, 7.28]

    softmax_rouge_l = [15.04, 15.93, 16.31, 16.58]
    nystrom_rouge_l = [14.95, 15.56, 15.60, 15.83]
    monarch_rouge_l = [14.98, 15.76, 16.15, 16.42]

    plot_data_config = {
        "softmax": {
            "flops": softmax_flops,
            "rouge_1": softmax_rouge_1,
            "rouge_2": softmax_rouge_2,
            "rouge_l": softmax_rouge_l,
            "color": bart_colors["softmax"],
            "label": "softmax",
        },
        "nystromformer": {
            "flops": nystrom_flops,
            "rouge_1": nystrom_rouge_1,
            "rouge_2": nystrom_rouge_2,
            "rouge_l": nystrom_rouge_l,
            "color": bart_colors["nystromformer"],
            "label": "nystromformer",
        },
        "monarch-attention": {
            "flops": monarch_flops,
            "rouge_1": monarch_rouge_1,
            "rouge_2": monarch_rouge_2,
            "rouge_l": monarch_rouge_l,
            "color": bart_colors["monarch-attention"],
            "label": "monarch-attention",
        },
    }

    # Markers for different configurations/settings (e.g., sequence lengths)
    markers_list = ["P", "p", "D", "s"]  # Assuming 4 points per list
    # marker_legend_labels = [f"Setting {i+1}" for i in range(len(markers_list))]
    seq_len_labels = ["1024", "2048", "4096", "8192"]
    marker_legend_labels = [f"{sl}" for sl in seq_len_labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # Increased figsize for legend

    # fig, axes = plt.subplots(1, 3, figsize=(23, 5))  # Increased figsize for legend

    rouge_metrics_map = [
        ("rouge_1", "ROUGE-1"),
        # ("rouge_2", "ROUGE-2"),
        ("rouge_l", "ROUGE-L"),
    ]

    for i, (metric_key, metric_name) in enumerate(rouge_metrics_map):
        ax = axes[i]
        ax.set_xscale("log")
        ax.set_xlabel("Total Attention FLOPs")
        ax.set_ylabel(metric_name)
        # ax.set_title(f"{metric_name} vs FLOPs")
        ax.grid(True, linestyle="--", alpha=0.6)

        for (
            att_type_key
        ) in attention_type_order:  # Iterate in defined order for consistency
            data = plot_data_config[att_type_key]
            if len(data["flops"]) != len(markers_list):
                raise ValueError(
                    f"Mismatch between number of data points for {att_type_key} ({len(data['flops'])}) and number of defined markers ({len(markers_list)})"
                )
            for point_idx in range(len(data["flops"])):
                ax.scatter(
                    data["flops"][point_idx],
                    data[metric_key][point_idx],
                    marker=markers_list[point_idx],
                    color=data["color"],
                    s=150,
                    edgecolor="black",
                    linewidth=1.5,
                )

    # Create legend handles
    color_handles = []
    for att_type_key in attention_type_order:
        data = plot_data_config[att_type_key]
        color_handles.append(
            mlines.Line2D(
                [],
                [],
                color=data["color"],
                marker="o",
                linestyle="None",  # Use a consistent marker for color legend part
                markersize=10,
                label=data["label"],
                markeredgecolor="black",
                markeredgewidth=1.5,
            )
        )

    marker_handles = []
    for point_idx in range(len(markers_list)):
        marker_handles.append(
            mlines.Line2D(
                [],
                [],
                color="grey",
                marker=markers_list[point_idx],
                linestyle="None",  # Use a neutral color
                markersize=10,
                label=marker_legend_labels[point_idx],
                markeredgecolor="black",
                markeredgewidth=1.5,
            )
        )

    fig.subplots_adjust(right=0.78)  # Adjust space for the legend
    # Create the first legend for attention methods (colors)
    legend1 = fig.legend(
        handles=color_handles,
        loc="upper left",  # Position anchor point of the legend box
        bbox_to_anchor=(0.8, 0.7),  # Position legend box relative to figure
        # title="Attention Method",
        # fontsize="small",
    )
    # Add the first legend manually to the figure, so the second one doesn't overwrite it
    fig.add_artist(
        legend1
    )  # Not strictly necessary if calling fig.legend multiple times

    # Create the second legend for sequence lengths (markers)
    legend2 = fig.legend(
        handles=marker_handles,
        loc="lower left",  # Position anchor point of the legend box
        bbox_to_anchor=(0.825, 0.2),  # Position legend box relative to figure
        title="Sequence Length",
        # fontsize="small",
    )
    # fig.legend(
    #     handles=color_handles + marker_handles,
    #     loc="center left",
    #     bbox_to_anchor=(0.77, 0.5),
    #     # title="Legend",
    # )

    fig.savefig("figures/bart_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
