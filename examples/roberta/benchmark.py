from time import time
from typing import Optional

import torch
from common.utils import benchmark_flops, benchmark_time, get_device, move
from roberta.data import GlueTaskName, glue_dataloader
from roberta.models import AttentionType, CustomRobertaConfig, get_config, get_model
from torchtnt.utils.flops import FlopTensorDispatchMode
from tqdm import tqdm
from transformers.utils import logging

# @torch.no_grad()
# def evaluate_accuracy(
#     config: CustomViTConfig,
#     batch_size: int = 1,
#     top_k: int = 5,
#     num_samples: Optional[int] = None,
# ):
#     device = get_device()
#     dataloader = imagenet_dataloader(batch_size=batch_size)
#     model = get_model(config, device)

#     total = 0
#     total_correct = 0

#     # TODO: Clean this up a bit (make a metric class, move to common utils?)

#     for batch in tqdm(dataloader):
#         inputs = move(batch, device)
#         labels = inputs.pop("labels")
#         outputs = model(**inputs).logits
#         num_correct = top_k_accuracy_score(
#             labels.cpu().numpy(),
#             outputs.cpu().numpy(),
#             k=top_k,
#             labels=list(range(1000)),
#             normalize=False,
#         )
#         total = total + labels.size(0)
#         total_correct = total_correct + num_correct

#         if num_samples is not None and total >= num_samples:
#             break

#     print(f"{config.attention_type} top-{top_k} accuracy: {total_correct / total:0.3f}")


@torch.no_grad()
def evaluate_runtime_and_flops(task_name: GlueTaskName, config: CustomRobertaConfig):
    device = get_device()
    example = next(iter(glue_dataloader(task_name)))
    inputs = move(example, device)
    model = get_model(config, device)

    # Warm up
    model(**inputs)

    # Timing
    total_ms = benchmark_time(model, inputs)

    # Flops
    flops_counts = benchmark_flops(model, inputs, "roberta.encoder.layer.0.attention")

    print(f"{config.attention_type} attention time: {total_ms:.2f}ms")
    print(f"{config.attention_type} attention flops: {flops_counts:0.2e}")


def main():
    pass

    # num_samples = 100
    # batch_size = 4

    # # Softmax
    # config = get_config()
    # config.attention_type = AttentionType.softmax
    # config.enable_flash_attention = False
    # evaluate_accuracy(config, batch_size=batch_size, num_samples=num_samples)
    # evaluate_runtime_and_flops(config)

    # # Sparsemax
    # config = get_config()
    # config.attention_type = AttentionType.sparsemax
    # config.scale_attention_temperature = True
    # evaluate_accuracy(config, batch_size=batch_size, num_samples=num_samples)

    # # Monarch
    # config = get_config()
    # config.attention_type = AttentionType.monarch
    # config.scale_attention_temperature = True
    # config.efficient_attention_num_steps = 3
    # config.efficient_attention_step_size = 2.5
    # config.efficient_attention_block_size = 14
    # evaluate_accuracy(config, batch_size=batch_size, num_samples=num_samples)
    # evaluate_runtime_and_flops(config)

    # TODO: Add baselines here


if __name__ == "__main__":
    main()
