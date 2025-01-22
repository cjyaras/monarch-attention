from typing import Dict, Optional

import torch
from common.logging import Logger
from evaluate import QuestionAnsweringEvaluator
from roberta.config import CustomRobertaConfig
from roberta.data import get_dataset
from roberta.pipeline import get_pipeline


class CustomQuestionAnsweringEvaluator(QuestionAnsweringEvaluator):

    def __init__(self):
        super().__init__(task="custom-question-answering")
        self.PIPELINE_KWARGS["padding"] = "max_length"


def create_evaluate_fn(num_samples: Optional[int] = None, batch_size: int = 1):
    from torchtnt.utils.flops import FlopTensorDispatchMode

    dataset = get_dataset(num_samples=num_samples)
    evaluator = CustomQuestionAnsweringEvaluator()

    @torch.no_grad()
    def evaluate_fn(config: CustomRobertaConfig) -> Dict[str, float]:
        pipe = get_pipeline(config, batch_size=batch_size)

        with FlopTensorDispatchMode(pipe.model) as ftdm:
            result = evaluator.compute(
                model_or_pipeline=pipe,
                data=dataset,
                metric="squad_v2",
                squad_v2_format=True,
            )
            assert isinstance(result, Dict)
            result["total_attention_bmm_flops"] = sum(
                [
                    ftdm.flop_counts[f"roberta.encoder.layer.{i}.attention"][
                        "bmm.default"
                    ]
                    for i in range(config.num_hidden_layers)
                ]
            )
        return result

    return evaluate_fn


class Evaluator:

    def __init__(self, num_samples: int, batch_size: int, save_dir: str):
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.evaluate_fn = create_evaluate_fn(
            num_samples=num_samples, batch_size=batch_size
        )
        self.logger = Logger(save_dir)

    def evaluate(self, config: CustomRobertaConfig) -> Dict[str, float]:
        return self.evaluate_fn(config)

    def evaluate_and_save(self, config: CustomRobertaConfig) -> str:
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name
