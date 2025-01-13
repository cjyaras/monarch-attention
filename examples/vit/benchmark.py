from time import time
from typing import Optional

import torch
from common.utils import get_device, move
from sklearn.metrics import top_k_accuracy_score
from torchtnt.utils.flops import FlopTensorDispatchMode
from tqdm import tqdm
from transformers.utils import logging
from vit.data import imagenet_dataloader
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

    total = 0
    total_correct = 0

    for batch in tqdm(dataloader):
        images, labels = move(batch["image"], device), move(batch["label"], device)
        outputs = model(**images).logits
        num_correct = top_k_accuracy_score(
            labels.cpu().numpy(),
            outputs.cpu().numpy(),
            k=top_k,
            labels=list(range(1000)),
            normalize=False,
        )
        total = total + labels.size(0)
        total_correct = total_correct + num_correct

        if num_samples is not None and total >= num_samples:
            break

    print(f"{config.attention_type} top-{top_k} accuracy: {total_correct / total:0.3f}")


@torch.no_grad()
def evaluate_runtime_and_flops(config: CustomViTConfig):
    device = get_device()
    inputs = move(next(iter(imagenet_dataloader())), device)["image"]
    model = get_model(config, device)

    # Warm up
    model(**inputs)

    # Timing
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # type: ignore
        model(**inputs)
        end_event.record()  # type: ignore
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)
    else:
        start = time()
        model(**inputs)
        total_ms = (time() - start) * 1000

    # Flops
    with FlopTensorDispatchMode(model) as ftdm:
        model(**inputs)
        flops_counts = ftdm.flop_counts["vit.encoder.layer.0.attention"]["bmm.default"]

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
    evaluate_runtime_and_flops(config)

    # Sparsemax
    config = get_config()
    config.attention_type = AttentionType.sparsemax
    config.scale_attention_temperature = True
    evaluate_accuracy(config, batch_size=batch_size, num_samples=num_samples)

    # Monarch
    config = get_config()
    config.attention_type = AttentionType.monarch
    config.scale_attention_temperature = True
    config.efficient_attention_num_steps = 3
    config.efficient_attention_step_size = 2.5
    config.efficient_attention_block_size = 14
    evaluate_accuracy(config, batch_size=batch_size, num_samples=num_samples)
    evaluate_runtime_and_flops(config)

    # TODO: Add baselines here


if __name__ == "__main__":
    main()
