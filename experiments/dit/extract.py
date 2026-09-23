from typing import Tuple

import torch

from experiments.common.attention import AttentionType
from experiments.common.utils import capture_attention_inputs, stack_captured
from experiments.dit.pipeline import get_pipeline

Tensor = torch.Tensor


@torch.no_grad()
def extract_query_key(
    attn_type: AttentionType,
    words: list,
    seed: int = 33,
    num_inference_steps: int = 1,
) -> Tuple[Tensor, Tensor]:

    pipe = get_pipeline(attn_type)

    attn_modules = [block.attn1.processor.attn_module for block in pipe.transformer.transformer_blocks]
    class_ids = pipe.get_label_ids(words)

    with capture_attention_inputs(attn_modules) as records:
        pipe(class_labels=class_ids, num_inference_steps=num_inference_steps, generator=torch.manual_seed(seed))

    return stack_captured(records, "query"), stack_captured(records, "key")
