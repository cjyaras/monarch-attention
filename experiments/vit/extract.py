from typing import Optional, Tuple

import torch

from experiments.common.utils import batches, capture_attention_inputs, stack_captured
from experiments.vit.config import CustomViTConfig
from experiments.vit.evaluation import Evaluator
from experiments.vit.model import get_model

Tensor = torch.Tensor


@torch.no_grad()
def extract_query_key(
    config: CustomViTConfig,
    num_samples: Optional[int] = None,
    batch_size: int = 1,
    split: str = "validation",
) -> Tuple[Tensor, Tensor]:
    evaluator = Evaluator(
        num_samples, top_k=1, batch_size=batch_size, save_dir="", split=split
    )
    model = get_model(config)
    attn_modules = [layer.attention.attn_module for layer in model.vit.layers]

    with capture_attention_inputs(attn_modules) as records:
        for examples in batches(evaluator.dataset, batch_size):
            evaluator.predict(model, examples)

    return stack_captured(records, "query"), stack_captured(records, "key")
