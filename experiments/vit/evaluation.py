from typing import Dict, Optional

import torch
from transformers.image_utils import load_image

from experiments.common.logging import Logger
from experiments.common.utils import attention_bmm_flops, batches
from experiments.vit.config import CustomViTConfig
from experiments.vit.data import get_dataset
from experiments.vit.model import CustomViTForImageClassification, get_model
from experiments.vit.processor import get_processor


class Evaluator:

    def __init__(
        self,
        num_samples: Optional[int],
        top_k: int,
        batch_size: int,
        save_dir: str,
        split: str = "validation",
    ):
        self.top_k = top_k
        self.batch_size = batch_size
        self.dataset = get_dataset(num_samples=num_samples, split=split)
        self.processor = get_processor()
        self.logger = Logger(save_dir)

    @torch.no_grad()
    def predict(self, model: CustomViTForImageClassification, examples) -> torch.Tensor:
        """Top-k predicted class ids for each image."""
        images = [load_image(image) for image in examples["image"]]
        inputs = self.processor(images=images, return_tensors="pt").to(model.device)
        return model(**inputs).logits.topk(self.top_k).indices.cpu()

    def evaluate(self, config: CustomViTConfig) -> Dict[str, float]:
        model = get_model(config)
        flops = attention_bmm_flops(
            model,
            [f"vit.layers.{i}.attention" for i in range(config.num_hidden_layers)],
            lambda: self.predict(model, self.dataset[: self.batch_size]),
        )

        correct = []
        for examples in batches(self.dataset, self.batch_size):
            predictions = self.predict(model, examples)
            labels = torch.tensor(examples["label"])
            correct.append((predictions == labels[:, None]).any(dim=-1))
        accuracy = 100 * torch.cat(correct).double().mean().item()

        return {
            f"top-{self.top_k} accuracy": accuracy,
            "total_attention_bmm_flops": flops // self.batch_size,
        }

    def evaluate_and_save(self, config: CustomViTConfig) -> str:
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name
