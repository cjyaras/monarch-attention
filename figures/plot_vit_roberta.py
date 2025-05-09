import json
import os

import matplotlib.pyplot as plt

from common.plotting import plot_results


def main():
    fig, axes = plt.subplots(ncols=2, figsize=(14, 5))

    # ViT plot
    SAVE_DIR = "vit/results"
    results = []
    for result_string in os.listdir(SAVE_DIR):
        with open(os.path.join(SAVE_DIR, result_string), "r") as f:
            result = json.load(f)
            results.append(result)

    plotted_types = plot_results(
        axes[0], results, metric_name="top-5 accuracy", title="ViT ImageNet"
    )

    # RoBERTa plot

    SAVE_DIR = "roberta/results"
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
