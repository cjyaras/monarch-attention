from typing import Dict, Optional

import torch
from common.data import dataset_from_iterable
from config import CustomViTConfig, get_config
from data import get_dataset
from evaluate import ImageClassificationEvaluator
from metric import TopKAccuracy
from pipeline import get_pipeline


class TopKImageClassificationEvaluator(ImageClassificationEvaluator):

    def __init__(self, top_k: int):
        super().__init__(task="custom-image-classification", default_metric_name="")
        self.top_k = top_k
        self.PIPELINE_KWARGS["top_k"] = top_k

    def predictions_processor(self, predictions, label_mapping):
        pred_label = [[x["label"] for x in pred] for pred in predictions]
        pred_label = [
            [label_mapping[x] for x in pred] if label_mapping is not None else pred
            for pred in pred_label
        ]

        return {"predictions": pred_label}


@torch.no_grad()
def evaluate(
    config: CustomViTConfig, num_samples: Optional[int] = None, top_k: int = 5
) -> Dict[str, float]:
    pipe = get_pipeline(config)
    dataset = dataset_from_iterable(get_dataset(num_samples=num_samples))
    evaluator = TopKImageClassificationEvaluator(top_k=top_k)
    result = evaluator.compute(
        pipe,
        dataset,
        metric=TopKAccuracy(),
        label_mapping=pipe.model.config.label2id,  # type: ignore
    )
    assert isinstance(result, Dict)
    return result


@torch.no_grad()
def main():
    config = get_config()
    print(evaluate(config, num_samples=256))


if __name__ == "__main__":
    main()
