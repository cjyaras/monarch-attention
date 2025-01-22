from typing import Dict

import torch
from common.logging import Logger
from evaluate import QuestionAnsweringEvaluator
from roberta.config import CustomRobertaConfig
from roberta.data import get_dataset
from roberta.pipeline import CustomQuestionAnsweringPipeline, get_pipeline


class CustomQuestionAnsweringEvaluator(QuestionAnsweringEvaluator):

    def __init__(self):
        super().__init__(task="custom-question-answering")
        self.PIPELINE_KWARGS["padding"] = "max_length"


class Evaluator:

    def __init__(self, num_samples: int, batch_size: int, save_dir: str):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.dataset = get_dataset(num_samples=num_samples)
        self.evaluator = CustomQuestionAnsweringEvaluator()
        self.logger = Logger(save_dir)

    def benchmark_flops(self, pipe: CustomQuestionAnsweringPipeline):
        from torchtnt.utils.flops import FlopTensorDispatchMode

        with FlopTensorDispatchMode(pipe.model) as ftdm:
            self.evaluator.compute(
                model_or_pipeline=pipe,
                data=self.dataset.take(self.batch_size),
                metric="squad_v2",
                squad_v2_format=True,
            )
            return sum(
                [
                    ftdm.flop_counts[f"roberta.encoder.layer.{i}.attention"][
                        "bmm.default"
                    ]
                    for i in range(pipe.model.config.num_hidden_layers)
                ]
            )

    def evaluate(self, config: CustomRobertaConfig) -> Dict[str, float]:
        pipe = get_pipeline(config, batch_size=self.batch_size)

        # Benchmark FLOPs (and warmup for compilation)
        flop_count = self.benchmark_flops(pipe)
        result = self.evaluator.compute(
            model_or_pipeline=pipe,
            data=self.dataset,
            metric="squad_v2",
            squad_v2_format=True,
        )
        assert isinstance(result, Dict)
        result["total_attention_bmm_flops"] = flop_count
        return result

    def evaluate_and_save(self, config: CustomRobertaConfig) -> str:
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name
