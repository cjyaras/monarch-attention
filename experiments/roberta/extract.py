from typing import Optional, Tuple

import torch

from experiments.common.utils import capture_attention_inputs, stack_captured
from experiments.roberta.config import CustomRobertaConfig
from experiments.roberta.evaluation import Evaluator
from experiments.roberta.model import get_model

Tensor = torch.Tensor


@torch.no_grad()
def extract_query_key_mask(
    config: CustomRobertaConfig,
    num_samples: Optional[int] = None,
    batch_size: int = 1,
    split: str = "validation",
) -> Tuple[Tensor, Tensor, Tensor]:
    evaluator = Evaluator(num_samples, batch_size=batch_size, save_dir="", split=split)
    model = get_model(config)
    attn_modules = [
        layer.attention.self.attn_module for layer in model.roberta.encoder.layer
    ]

    with capture_attention_inputs(attn_modules) as records:
        evaluator.predict(model, evaluator.dataset[:])

    # The padding mask is the same in every layer
    attention_mask = torch.cat(records[0]["attention_mask"])
    return (
        stack_captured(records, "query"),
        stack_captured(records, "key"),
        attention_mask,
    )
