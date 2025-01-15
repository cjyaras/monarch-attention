from typing import Optional

import torch
from common.utils import benchmark_flops, benchmark_time, get_device, move
from roberta.data import GlueTaskName, glue_dataloader
from roberta.metrics import GLUE_METRIC_DICT, GlueMetric
from roberta.models import AttentionType, CustomRobertaConfig, get_config, get_model
from tqdm import tqdm
from transformers.utils import logging

# logging.set_verbosity_error()


@torch.no_grad()
def evaluate_accuracy(
    task_name: GlueTaskName,
    config: CustomRobertaConfig,
    batch_size: int = 1,
    num_samples: Optional[int] = None,
):

    device = get_device()
    dataloader = glue_dataloader(
        task_name, batch_size=batch_size, num_samples=num_samples, streaming=True
    )
    model = get_model(task_name, config, device)
    metric = GlueMetric(task_name)
    metric_name = GLUE_METRIC_DICT[task_name]

    for batch in tqdm(dataloader):
        inputs = move(batch, device)
        labels = inputs.pop("labels")
        outputs = model(**inputs).logits
        print(outputs)
        metric.add_batch(logits=outputs, labels=labels)

    metric_value = metric.compute()[metric_name]
    print(f"{config.attention_type} {task_name} {metric_name}: {metric_value:0.3f}")


@torch.no_grad()
def evaluate_runtime_and_flops(config: CustomRobertaConfig):

    device = get_device()
    example = next(
        iter(glue_dataloader(GlueTaskName.cola, batch_size=1, num_samples=1))
    )
    inputs = move(example, device)
    model = get_model(GlueTaskName.cola, config, device)

    # Warm up
    model(**inputs)

    # Timing
    total_ms = benchmark_time(model, inputs)

    # Flops
    flops_counts = benchmark_flops(model, inputs, "roberta.encoder.layer.0.attention")

    print(f"{config.attention_type} attention time: {total_ms:.2f}ms")
    print(f"{config.attention_type} attention flops: {flops_counts:0.2e}")


def main():

    num_samples = 8
    batch_size = 8
    task_name = GlueTaskName.cola

    # Sparsemax
    config = get_config()
    config.attention_type = AttentionType.sparsemax
    evaluate_accuracy(task_name, config, batch_size=batch_size, num_samples=num_samples)
    # evaluate_runtime_and_flops(config)

    # Monarch
    config = get_config()
    config.attention_type = AttentionType.low_rank
    config.efficient_attention_num_steps = 50
    config.efficient_attention_step_size = 1.0
    config.efficient_attention_rank = 4
    config.efficient_attention_block_size = 4
    evaluate_accuracy(task_name, config, batch_size=batch_size, num_samples=num_samples)
    # evaluate_runtime_and_flops(config)

    # TODO: Add baselines here


if __name__ == "__main__":
    main()
