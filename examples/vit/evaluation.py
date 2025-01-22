from typing import Dict

import torch
from common.data import dataset_from_iterable
from common.logging import Logger
from evaluate import ImageClassificationEvaluator
from vit.config import CustomViTConfig
from vit.data import get_dataset
from vit.metric import TopKAccuracy
from vit.pipeline import CustomImageClassificationPipeline, get_pipeline


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

    def __init__(self, num_samples: int, top_k: int, batch_size: int, save_dir: str):
        self.num_samples = num_samples
        self.top_k = top_k
        self.batch_size = batch_size
        self.dataset = dataset_from_iterable(get_dataset(num_samples=num_samples))
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
            return sum(
                [
                    ftdm.flop_counts[f"vit.encoder.layer.{i}.attention"]["bmm.default"]
                    for i in range(pipe.model.config.num_hidden_layers)
                ]
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


# def create_evaluate_fn(
#     num_samples: Optional[int] = None,
#     batch_size: int = 1,
#     top_k: int = 5,
# ):
#     from torchtnt.utils.flops import FlopTensorDispatchMode

#     dataset = dataset_from_iterable(get_dataset(num_samples=num_samples))
#     evaluator = CustomImageClassificationEvaluator(top_k=top_k)
#     metric = TopKAccuracy()

#     @torch.no_grad()
#     def evaluate_fn(config: CustomViTConfig) -> Dict[str, float]:
#         pipe = get_pipeline(config, batch_size=batch_size)

#         with FlopTensorDispatchMode(pipe.model) as ftdm:
#             result = evaluator.compute(
#                 model_or_pipeline=pipe,
#                 data=dataset,
#                 metric=metric,
#                 label_mapping=pipe.model.config.label2id,  # type: ignore
#             )
#             assert isinstance(result, Dict)
#             result["total_attention_bmm_flops"] = sum(
#                 [
#                     ftdm.flop_counts[f"vit.encoder.layer.{i}.attention"]["bmm.default"]
#                     for i in range(config.num_hidden_layers)
#                 ]
#             )
#         return result

#     return evaluate_fn


# class Evaluator:

#     def __init__(self, num_samples: int, top_k: int, batch_size: int, save_dir: str):
#         self.num_samples = num_samples
#         self.top_k = top_k
#         self.batch_size = batch_size

#         self.evaluate_fn = create_evaluate_fn(
#             num_samples=num_samples, batch_size=batch_size, top_k=top_k
#         )
#         self.logger = Logger(save_dir)

#     def evaluate(self, config: CustomViTConfig) -> Dict[str, float]:
#         return self.evaluate_fn(config)

#     def evaluate_and_save(self, config: CustomViTConfig) -> str:
#         result = self.evaluate(config)
#         file_name = self.logger.save(config, result)
#         return file_name
