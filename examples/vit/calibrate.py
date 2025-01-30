import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from common.baselines import Softmax, Sparsemax
from tqdm.auto import tqdm
from vit.config import AttentionType, get_config
from vit.data import get_processed_dataset
from vit.extract import extract_query_key
from vit.model import get_model

from sobalib.layers import PadType, SobaMonarch

Tensor = torch.Tensor


def calibrate_sparsemax_layerwise(
    learning_rate: float,
    num_steps: int,
    num_samples: Optional[int] = None,
) -> Dict[str, Tensor]:
    config = get_config()
    all_query, all_key = extract_query_key(config, num_samples=num_samples)

    query_per_layer = torch.unbind(all_query.transpose(1, 0))
    key_per_layer = torch.unbind(all_key.transpose(1, 0))

    softmax = Softmax()
    log_attention_scale = {}

    for i in range(config.num_hidden_layers):
        query = query_per_layer[i]
        key = key_per_layer[i]
        sparsemax = Sparsemax(config.num_attention_heads)
        optimizer = torch.optim.Adam(sparsemax.parameters(), lr=learning_rate)
        loss_vals = []
        for _ in tqdm(range(num_steps)):
            optimizer.zero_grad()
            softmax_out = softmax.get_matrix(query, key)
            sparsemax_out = sparsemax.get_matrix(query, key)
            loss = torch.nn.functional.mse_loss(sparsemax_out, softmax_out)
            loss.backward()
            optimizer.step()
            loss_vals.append(loss.item())
        log_attention_scale[
            f"vit.encoder.layer.{i}.attention.attention.attn_module.log_attention_scale"
        ] = sparsemax.log_attention_scale.detach()
        plt.plot(loss_vals)
        plt.show()

    return log_attention_scale


def calibrate_sparsemax_logits(
    learning_rate: float,
    num_steps: int,
    init_log_attention_scale: Optional[Dict[str, Tensor]] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Tensor]:

    # Load softmax model
    config = get_config()
    softmax_model = get_model(config)

    # Load sparsemax model
    config = get_config()
    config.attention_type = AttentionType.sparsemax
    config.scale_attention_temperature = True
    sparsemax_model = get_model(config)
    if init_log_attention_scale is not None:
        sparsemax_model.load_state_dict(init_log_attention_scale, strict=False)

    # Freeze all parameters (except attention_scale)
    for k, v in sparsemax_model.named_parameters():
        v.requires_grad = "log_attention_scale" in k

    trainable_params = filter(lambda p: p.requires_grad, sparsemax_model.parameters())

    # Optimizer
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Load dataset
    dataset = get_processed_dataset(num_samples=num_samples)

    loss_fn = lambda x, y: torch.nn.functional.kl_div(
        x.log_softmax(-1), y.log_softmax(-1), reduction="batchmean", log_target=True
    )

    loss_vals = []
    num_samples = len(dataset)

    for _ in tqdm(range(num_steps)):
        batch_idx = (
            np.random.choice(np.arange(num_samples), batch_size, replace=False)
            if batch_size is not None
            else np.arange(num_samples)
        )
        inputs = dataset[batch_idx]
        optimizer.zero_grad()
        with torch.no_grad():
            softmax_logits = softmax_model(**inputs).logits
        sparsemax_logits = sparsemax_model(**inputs).logits
        loss = loss_fn(sparsemax_logits, softmax_logits)
        loss.backward()
        optimizer.step()
        loss_vals.append(loss.item())
        print(loss.item())

    plt.plot(loss_vals)
    plt.show()

    log_attention_scale = {
        k: v
        for k, v in sparsemax_model.state_dict().items()
        if "log_attention_scale" in k
    }
    return log_attention_scale


def calibrate_soba_layerwise(
    learning_rate: float,
    num_steps: int,
    init_log_attention_scale: Optional[Dict[str, Tensor]] = None,
    num_samples: Optional[int] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    config = get_config()
    all_query, all_key = extract_query_key(config, num_samples=num_samples)
    query_per_layer = torch.unbind(all_query.transpose(1, 0))
    key_per_layer = torch.unbind(all_key.transpose(1, 0))

    softmax = Softmax()

    log_attention_scale = {}
    log_step_size = {}

    for i in range(config.num_hidden_layers):
        query = query_per_layer[i]
        key = key_per_layer[i]

        soba = SobaMonarch(
            block_size=14,
            num_steps=3,
            num_heads=config.num_attention_heads,
            pad_type=PadType.pre,
        )

        if init_log_attention_scale is not None:
            soba.load_state_dict(
                {
                    "log_attention_scale": init_log_attention_scale[
                        f"vit.encoder.layer.{i}.attention.attention.attn_module.log_attention_scale"
                    ]
                },
                strict=False,
            )

        optimizer = torch.optim.Adam(soba.parameters(), lr=learning_rate)
        loss_vals = []

        for _ in tqdm(range(num_steps)):
            optimizer.zero_grad()
            softmax_out = softmax.get_matrix(query, key)
            soba_out = soba.get_matrix(query, key)
            loss = torch.nn.functional.mse_loss(soba_out, softmax_out)
            loss.backward()
            optimizer.step()
            loss_vals.append(loss.item())

        loss_vals = torch.tensor(loss_vals)

        log_attention_scale[
            f"vit.encoder.layer.{i}.attention.attention.attn_module.log_attention_scale"
        ] = soba.log_attention_scale.detach()

        log_step_size[
            f"vit.encoder.layer.{i}.attention.attention.attn_module.log_step_size"
        ] = soba.log_step_size.detach()

        plt.plot(loss_vals)
        plt.show()

    return log_attention_scale, log_step_size


def calibrate_soba_logits(
    learning_rate: float,
    num_steps: int,
    init_log_attention_scale: Optional[Dict[str, Tensor]] = None,
    init_log_step_size: Optional[Dict[str, Tensor]] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:

    # Load softmax model
    config = get_config()
    softmax_model = get_model(config)

    # Load soba model
    config = get_config()
    config.attention_type = AttentionType.soba_monarch
    config.num_steps = 3
    config.block_size = 14
    soba_model = get_model(config)

    if init_log_attention_scale is not None:
        soba_model.load_state_dict(init_log_attention_scale, strict=False)

    if init_log_step_size is not None:
        soba_model.load_state_dict(init_log_step_size, strict=False)

    # Freeze all parameters (except attention_temperature and step_size)
    for k, v in soba_model.named_parameters():
        v.requires_grad = "log_attention_scale" in k or "log_step_size" in k

    trainable_params = filter(lambda p: p.requires_grad, soba_model.parameters())

    # Optimizer
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Load dataset
    dataset = get_processed_dataset(num_samples=num_samples)

    loss_fn = lambda x, y: torch.nn.functional.kl_div(
        x.log_softmax(-1), y.log_softmax(-1), reduction="batchmean", log_target=True
    )

    loss_vals = []
    num_samples = len(dataset)

    for _ in tqdm(range(num_steps)):
        batch_idx = (
            np.random.choice(np.arange(num_samples), batch_size, replace=False)
            if batch_size is not None
            else np.arange(num_samples)
        )
        inputs = dataset[batch_idx]
        optimizer.zero_grad()
        with torch.no_grad():
            softmax_logits = softmax_model(**inputs).logits
        soba_logits = soba_model(**inputs).logits
        loss = loss_fn(soba_logits, softmax_logits)
        loss.backward()
        optimizer.step()
        loss_vals.append(loss.item())
        print(loss.item())

    plt.plot(loss_vals)
    plt.show()

    log_attention_temperature = soba_model.log_attention_scale.detach()
    log_step_size = soba_model.log_step_size.detach()
    return log_attention_temperature, log_step_size


def calibrate_sparsemax():
    if not os.path.exists("vit/sparsemax_layerwise_log_attention_scale.pt"):
        log_attention_scale = calibrate_sparsemax_layerwise(
            learning_rate=5e-2, num_steps=200, num_samples=4
        )
        torch.save(
            log_attention_scale, "vit/sparsemax_layerwise_log_attention_scale.pt"
        )


def calibrate_soba():
    assert os.path.exists("vit/sparsemax_layerwise_log_attention_scale.pt")

    log_attention_scale = torch.load(
        "vit/sparsemax_layerwise_log_attention_scale.pt", weights_only=True
    )

    if not os.path.exists(
        "vit/soba_layerwise_log_attention_scale.pt"
    ) or not os.path.exists("vit/soba_layerwise_log_step_size.pt"):
        log_attention_scale, log_step_size = calibrate_soba_layerwise(
            learning_rate=5e-2,
            num_steps=300,
            init_log_attention_scale=log_attention_scale,
            num_samples=4,
        )
        torch.save(log_attention_scale, "vit/soba_layerwise_log_attention_scale.pt")
        torch.save(log_step_size, "vit/soba_layerwise_log_step_size.pt")
    else:
        log_attention_scale = torch.load(
            "vit/soba_layerwise_log_attention_scale.pt", weights_only=True
        )
        log_step_size = torch.load(
            "vit/soba_layerwise_log_step_size.pt", weights_only=True
        )

    log_attention_scale, log_step_size = calibrate_soba_logits(
        learning_rate=1e-2,
        num_steps=1000,
        init_log_attention_scale=log_attention_scale,
        init_log_step_size=log_step_size,
        num_samples=16,
        batch_size=4,
    )
    torch.save(log_attention_scale, "vit/soba_logits_log_attention_scale.pt")
    torch.save(log_step_size, "vit/soba_logits_log_step_size.pt")


def main():
    calibrate_sparsemax()
    calibrate_soba()


if __name__ == "__main__":
    main()
