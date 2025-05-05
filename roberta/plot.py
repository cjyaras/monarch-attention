import json
import os

from common.plotting import plot_results

SAVE_DIR = "roberta/results"


def main():
    results = []
    for result_string in os.listdir(SAVE_DIR):
        with open(os.path.join(SAVE_DIR, result_string), "r") as f:
            result = json.load(f)
            results.append(result)

    fig = plot_results(
        results,
        metric_name="f1",
        title="RoBERTa SQuAD",
        # y_break_limits=(25, 75),  # Break the axis between 0.3 and 0.7
        y_break_vspace=0.5,  # Set vertical space between subplots
    )
    fig.savefig("figures/roberta_results.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
