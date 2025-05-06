from typing import Dict

from evaluate import SummarizationEvaluator

from transformers import SummarizationPipeline

from common.logging import Logger
from bart.config import CustomBartConfig
from bart.data import get_dataset
from bart.pipeline import get_pipeline, CustomSummarizationPipeline


class CustomSummarizationEvaluator(SummarizationEvaluator):

    def __init__(self, max_length):
        super().__init__(task="custom-summarization")
        self.PIPELINE_KWARGS['max_length'] = max_length


class Evaluator:

    def __init__(self, num_samples: int, batch_size: int, save_dir: str, max_length: int):
        self.batch_size = batch_size
        self.dataset = get_dataset(num_samples=num_samples)
        self.evaluator = CustomSummarizationEvaluator(max_length)
        self.logger = Logger(save_dir)
        self.max_length = max_length

    def benchmark_flops(self, pipe: CustomSummarizationPipeline):
        from torchtnt.utils.flops import FlopTensorDispatchMode

        with FlopTensorDispatchMode(pipe.model) as ftdm:
            self.evaluator.compute(
                model_or_pipeline=pipe,
                data=self.dataset.take(1),
                metric="rouge", 
                input_column="chapter",
                label_column="summary_text",
            )
            return sum(
                [
                    ftdm.flop_counts[f"model.encoder.layers.{i}.self_attn"][
                        "bmm.default"
                    ]
                    for i in range(pipe.model.config.num_hidden_layers)
                ]
            )

    def evaluate(self, config: CustomBartConfig) -> Dict[str, float]:
        pipe = get_pipeline(config, batch_size=self.batch_size, max_length=self.max_length)

        #print(pipe("Generative Pre-trained Transformer models, known as GPT or OPT, set themselves apart through breakthrough performance across complex language modelling tasks, but also by their extremely high computational and storage costs. Specifically, due to their massive size, even inference for large, highly-accurate GPT models may require multiple performant GPUs, which limits the usability of such models. While there is emerging work on relieving this pressure via model compression, the applicability and performance of existing compression techniques is limited by the scale and complexity of GPT models. In this paper, we address this challenge,", truncation=True))
        #exit()

        # Benchmark FLOPs (and warmup for compilation)
        flop_count = self.benchmark_flops(pipe)
        result = self.evaluator.compute(
            model_or_pipeline=pipe,
            data=self.dataset,#.take(self.batch_size),
            metric="rouge",  
            input_column="chapter",
            label_column="summary_text",
        )

        assert isinstance(result, Dict)
        result["total_attention_bmm_flops"] = flop_count
        return result


    def evaluate_and_save(self, config: CustomBartConfig) -> str:
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name
