from time import time
from typing import Optional

import torch
from common.utils import benchmark_flops, benchmark_time, get_device, move
from torchtnt.utils.flops import FlopTensorDispatchMode
from tqdm import tqdm
from transformers.utils import logging
from vit.data import imagenet_dataloader
from vit.metrics import TopKAccuracyMetric
from vit.models import AttentionType, CustomViTConfig, get_config, get_model

logging.set_verbosity_error()


@torch.no_grad()
def evaluate_accuracy(
    config: CustomViTConfig,
    batch_size: int = 1,
    top_k: int = 5,
    num_samples: Optional[int] = None,
):

    device = get_device()
    dataloader = imagenet_dataloader(batch_size=batch_size)
    model = get_model(config, device)
    metric = TopKAccuracyMetric()

    total = 0

    for batch in tqdm(dataloader):
        inputs = move(batch, device)
        labels = inputs.pop("labels")
        outputs = model(**inputs).logits
        metric.add_batch(logits=outputs, labels=labels)
        total = total + labels.size(0)
        if num_samples is not None and total >= num_samples:
            break

    accuracy = metric.compute()["accuracy"]
    print(f"{config.attention_type} top-{top_k} accuracy: {accuracy:0.3f}")


@torch.no_grad()
def evaluate_runtime_and_flops(config: CustomViTConfig):

    device = get_device()
    example = next(iter(imagenet_dataloader()))
    inputs = move(example, device)
    model = get_model(config, device)

    # Warm up
    model(**inputs)

    # Timing
    total_ms = benchmark_time(model, inputs)

    # Flops
    flops_counts = benchmark_flops(model, inputs, "vit.encoder.layer.0.attention")

    print(f"{config.attention_type} attention time: {total_ms:.2f}ms")
    print(f"{config.attention_type} attention flops: {flops_counts:0.2e}")


def main():

    num_samples = 100
    batch_size = 4

    # Softmax
    config = get_config()
    config.attention_type = AttentionType.softmax
    config.enable_flash_attention = False
    evaluate_accuracy(config, batch_size=batch_size, num_samples=num_samples)
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
