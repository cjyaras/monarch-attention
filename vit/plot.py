import json
import os

from common.plotting import plot_results

SAVE_DIR = "vit/results"


def main():
    results = []
    for result_string in os.listdir(SAVE_DIR):
        with open(os.path.join(SAVE_DIR, result_string), "r") as f:
            result = json.load(f)
            results.append(result)

    fig = plot_results(results, metric_name="top-5 accuracy", title="ViT ImageNet")
    fig.savefig("figures/vit_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
