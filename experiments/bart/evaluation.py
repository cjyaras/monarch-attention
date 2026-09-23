import copy
from typing import Dict

import evaluate
import torch

from experiments.common.logging import Logger
from experiments.common.utils import attention_bmm_flops, batches
from experiments.bart.config import CustomBartConfig
from experiments.bart.data import get_dataset
from experiments.bart.model import CustomBartForConditionalGeneration, get_model
from experiments.bart.processor import get_processor


class Evaluator:

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        save_dir: str,
        max_length: int,
        model_checkpoint_path: str = "experiments/bart/finetuned/output/",
        max_new_tokens: int = 512,
    ):
        self.batch_size = batch_size
        self.dataset = get_dataset(num_samples=num_samples)
        self.tokenizer = get_processor(max_length)
        self.metric = evaluate.load("rouge")
        self.logger = Logger(save_dir)
        self.model_checkpoint_path = model_checkpoint_path
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def summarize(self, model: CustomBartForConditionalGeneration, texts: list[str]) -> list[str]:
        # Same settings the Hugging Face summarization pipeline used: the model's
        # generation config, with pipeline defaults for anything it leaves unset
        generation_config = copy.deepcopy(model.generation_config)
        generation_config.update(max_new_tokens=256, num_beams=4, defaults_only=True)
        generation_config.update(max_new_tokens=self.max_new_tokens)

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        output_ids = model.generate(**inputs.to(model.device), generation_config=generation_config)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def evaluate(self, config: CustomBartConfig) -> Dict[str, float]:
        model = get_model(config, model_checkpoint_path=self.model_checkpoint_path)
        flops = attention_bmm_flops(
            model,
            [f"model.encoder.layers.{i}.self_attn" for i in range(config.encoder_layers)],
            lambda: self.summarize(model, self.dataset[:1]["chapter"]),
        )

        predictions = []
        for examples in batches(self.dataset, self.batch_size):
            predictions += self.summarize(model, examples["chapter"])
        result = self.metric.compute(
            predictions=predictions, references=self.dataset["summary_text"]
        )
        assert result is not None
        result["total_attention_bmm_flops"] = flops
        return result

    def evaluate_and_save(self, config: CustomBartConfig):
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name, result
