from typing import Dict

from evaluate import SummarizationEvaluator

from experiments.common.logging import Logger
from experiments.bart.config import CustomBartConfig
from experiments.bart.data import get_dataset
from experiments.bart.pipeline import get_pipeline, CustomSummarizationPipeline


class CustomSummarizationEvaluator(SummarizationEvaluator):

    def __init__(self, max_length):
        super().__init__(task="custom-summarization")
        self.PIPELINE_KWARGS['max_new_tokens'] = 512


class Evaluator:

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        save_dir: str,
        max_length: int,
        model_checkpoint_path: str = "./bart/finetuned/output/",
    ):
        self.batch_size = batch_size
        self.dataset = get_dataset(num_samples=num_samples)
        self.evaluator = CustomSummarizationEvaluator(max_length)
        self.logger = Logger(save_dir)
        self.max_length = max_length
        self.model_checkpoint_path = model_checkpoint_path

    def benchmark_flops(self, pipe: CustomSummarizationPipeline):
        from torchtnt.utils.flops import FlopTensorDispatchMode

        with FlopTensorDispatchMode(pipe.model.model.encoder.layers[0].self_attn) as ftdm:
            self.evaluator.compute(
                model_or_pipeline=pipe,
                data=self.dataset.select(range(1)),
                metric="rouge", 
                input_column="chapter",
                label_column="summary_text",
            )
            flops = ftdm.flop_counts[""]["bmm.default"]
            return flops

    def evaluate(
        self,
        config: CustomBartConfig,
    ) -> Dict[str, float]:
        pipe = get_pipeline(
            config,
            batch_size=self.batch_size,
            max_length=self.max_length,
            model_checkpoint_path=self.model_checkpoint_path,
        )

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


    def evaluate_and_save(self, config: CustomBartConfig):
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name, result
