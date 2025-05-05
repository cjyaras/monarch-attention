import numpy as np
from datasets import Features, Sequence, Value
from evaluate import Metric, MetricInfo


def any_match(predictions, references):
    predictions = np.array(predictions)
    references = np.array(references)
    return np.any(predictions == references[:, None], axis=-1)


class TopKAccuracy(Metric):

    def _info(self):
        return MetricInfo(
            description="",
            citation="",
            inputs_description="",
            features=Features(
                {
                    "predictions": Sequence(Value("int32")),
                    "references": Value("int32"),
                }
            ),
            reference_urls=[],
        )

    def _compute(self, predictions, references):
        return {
            "top-5 accuracy": float(100 * any_match(predictions, references).mean())
        }
