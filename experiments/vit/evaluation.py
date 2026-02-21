from typing import Dict, Optional

from evaluate import ImageClassificationEvaluator

from common.logging import Logger
from experiments.vit.config import CustomViTConfig
from experiments.vit.data import get_dataset
from experiments.vit.metric import TopKAccuracy
from experiments.vit.pipeline import CustomImageClassificationPipeline, get_pipeline


class CustomImageClassificationEvaluator(ImageClassificationEvaluator):

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


class Evaluator:

    def __init__(
        self, num_samples: Optional[int], top_k: int, batch_size: int, save_dir: str
    ):
        self.batch_size = batch_size
        self.dataset = get_dataset(num_samples=num_samples)
        self.metric = TopKAccuracy()
        self.evaluator = CustomImageClassificationEvaluator(top_k=top_k)
        self.logger = Logger(save_dir)

    def benchmark_flops(self, pipe: CustomImageClassificationPipeline):
        from torchtnt.utils.flops import FlopTensorDispatchMode

        with FlopTensorDispatchMode(pipe.model) as ftdm:
            self.evaluator.compute(
                model_or_pipeline=pipe,
                data=self.dataset.take(self.batch_size),
                metric=self.metric,
                label_mapping=pipe.model.config.label2id,  # type: ignore
            )
            return (
                sum(
                    [
                        ftdm.flop_counts[f"vit.encoder.layer.{i}.attention"][
                            "bmm.default"
                        ]
                        for i in range(pipe.model.config.num_hidden_layers)
                    ]
                )
                // self.batch_size
            )

    def evaluate(self, config: CustomViTConfig) -> Dict[str, float]:
        pipe = get_pipeline(config, batch_size=self.batch_size)

        # Benchmark FLOPs (and warmup for compilation)
        flop_count = self.benchmark_flops(pipe)
        result = self.evaluator.compute(
            model_or_pipeline=pipe,
            data=self.dataset,
            metric=self.metric,
            label_mapping=pipe.model.config.label2id,  # type: ignore
        )
        assert isinstance(result, Dict)
        result["total_attention_bmm_flops"] = flop_count
        return result

    def evaluate_and_save(self, config: CustomViTConfig) -> str:
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name
