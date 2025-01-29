import os
from typing import Dict, Optional

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
):
    """First sparsemax calibration step.
    Tune attention temperature per layer based on mean-squared error of attention maps.
    """
    config = get_config()
    all_query, all_key = extract_query_key(config, num_samples=num_samples)

    query_per_layer = torch.unbind(all_query.transpose(1, 0))
    key_per_layer = torch.unbind(all_key.transpose(1, 0))

    softmax = Softmax()
    sparsemax = Sparsemax()

    attention_temperature = {}

    for i in range(config.num_hidden_layers):
        query = query_per_layer[i]
        key = key_per_layer[i]

        attention_temperature_per_layer = torch.zeros(
            (config.num_attention_heads,), device=query.device
        )
        attention_temperature_per_layer.requires_grad = True

        optimizer = torch.optim.Adam(
            [attention_temperature_per_layer], lr=learning_rate
        )
        loss_vals = []

        for _ in tqdm(range(num_steps)):
            optimizer.zero_grad()
            softmax_out = softmax.get_matrix(query, key)
            sparsemax_out = sparsemax.get_matrix(
                query * torch.exp(attention_temperature_per_layer[..., None, None]), key
            )
            loss = torch.nn.functional.mse_loss(sparsemax_out, softmax_out)
            loss.backward()
            optimizer.step()

            loss_vals.append(loss.item())

        loss_vals = torch.tensor(loss_vals)

        attention_temperature[
            f"vit.encoder.layer.{i}.attention.attention.attention_temperature"
        ] = attention_temperature_per_layer.detach()

        plt.plot(loss_vals)
        plt.show()

    return attention_temperature


def calibrate_sparsemax_logits(
    learning_rate: float,
    num_steps: int,
    init_attention_temperature: Optional[Dict[str, Tensor]] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """Second calibration step.
    Tune attention temperature across layers based on kl divergence between output probabilities.
    """

    # Load softmax model
    config = get_config()
    softmax_model = get_model(config)

    # Load sparsemax model
    config = get_config()
    config.attention_type = AttentionType.sparsemax
    config.scale_attention_temperature = True
    sparsemax_model = get_model(config)
    if init_attention_temperature is not None:
        sparsemax_model.load_state_dict(init_attention_temperature, strict=False)

    # Freeze all parameters (except attention_temperature)
    for k, v in sparsemax_model.named_parameters():
        v.requires_grad = "attention_temperature" in k

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
        idx = (
            np.random.choice(np.arange(num_samples), batch_size, replace=False)
            if batch_size is not None
            else np.arange(num_samples)
        )
        inputs = dataset[idx]

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

    attention_temperature = {
        k: v
        for k, v in sparsemax_model.state_dict().items()
        if "attention_temperature" in k
    }
    return attention_temperature


def calibrate_soba_layerwise(
    learning_rate: float,
    num_steps: int,
    init_attention_temperature: Optional[Dict[str, Tensor]] = None,
    num_samples: Optional[int] = None,
):
    """First soba calibration step.
    Tune attention temperature and step size per layer based on mean-squared error of attention maps.
    """
    config = get_config()
    all_query, all_key = extract_query_key(config, num_samples=num_samples)

    query_per_layer = torch.unbind(all_query.transpose(1, 0))
    key_per_layer = torch.unbind(all_key.transpose(1, 0))

    softmax = Softmax()

    attention_temperature = {}
    step_size = {}

    for i in range(config.num_hidden_layers):
        query = query_per_layer[i]
        key = key_per_layer[i]

        attention_temperature_per_layer = (
            torch.zeros((config.num_attention_heads,), device=query.device)
            if init_attention_temperature is None
            else init_attention_temperature[
                f"vit.encoder.layer.{i}.attention.attention.attention_temperature"
            ]
        )
        attention_temperature_per_layer.requires_grad = True

        soba = SobaMonarch(
            block_size=14,
            num_steps=3,
            init_step_size=2.5,
            num_heads=12,
            pad_type=PadType.pre,
        )

        optimizer = torch.optim.Adam(
            [attention_temperature_per_layer] + list(soba.parameters()),
            lr=learning_rate,
        )
        loss_vals = []

        for _ in tqdm(range(num_steps)):
            optimizer.zero_grad()
            softmax_out = softmax.get_matrix(query, key)
            soba_out = soba.get_matrix(
                query * torch.exp(attention_temperature_per_layer[..., None, None]), key
            )
            loss = torch.nn.functional.mse_loss(soba_out, softmax_out)
            loss.backward()
            optimizer.step()

            loss_vals.append(loss.item())

        loss_vals = torch.tensor(loss_vals)

        attention_temperature[
            f"vit.encoder.layer.{i}.attention.attention.attention_temperature"
        ] = attention_temperature_per_layer.detach()

        step_size[
            f"vit.encoder.layer.{i}.attention.attention.attn_module.step_size"
        ] = next(soba.parameters()).detach()

        plt.plot(loss_vals)
        plt.show()

    return attention_temperature, step_size


def calibrate_soba_logits(
    learning_rate: float,
    num_steps: int,
    init_attention_temperature: Optional[Dict[str, Tensor]] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """Soba calibration.
    Tune attention temperature and step size across layers based on kl divergence between output probabilities.
    """

    # Load softmax model
    config = get_config()
    softmax_model = get_model(config)

    # Load soba model
    config = get_config()
    config.attention_type = AttentionType.soba_monarch
    config.num_steps = 3
    config.init_step_size = 2.5
    config.block_size = 14
    config.scale_attention_temperature = True
    soba_model = get_model(config)

    if init_attention_temperature is not None:
        soba_model.load_state_dict(init_attention_temperature, strict=False)

    # Freeze all parameters (except attention_temperature and step_size)
    for k, v in soba_model.named_parameters():
        v.requires_grad = "attention_temperature" in k or "step_size" in k

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
        idx = (
            np.random.choice(np.arange(num_samples), batch_size, replace=False)
            if batch_size is not None
            else np.arange(num_samples)
        )
        inputs = dataset[idx]

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

    attention_temperature = {
        k: v
        for k, v in soba_logits.state_dict().items()
        if "attention_temperature" in k
    }
    step_size = {k: v for k, v in soba_logits.state_dict().items() if "step_size" in k}
    return attention_temperature, step_size


def calibrate_sparsemax():

    # First step
    if not os.path.exists("vit/attention_temperature.pt"):
        attention_temperature = calibrate_sparsemax_layerwise(
            learning_rate=5e-2, num_steps=200, num_samples=4
        )
        torch.save(attention_temperature, "vit/attention_temperature.pt")
    else:
        attention_temperature = torch.load(
            "vit/attention_temperature.pt", weights_only=True
        )

    # Second step
    if not os.path.exists("vit/attention_temperature_2.pt"):
        attention_temperature = calibrate_sparsemax_logits(
            learning_rate=1e-2,
            num_steps=1000,
            init_attention_temperature=attention_temperature,
            num_samples=256,
            batch_size=4,
        )
        torch.save(attention_temperature, "vit/attention_temperature_2.pt")


def calibrate_soba():

    init_attention_temperature = torch.load(
        "vit/attention_temperature.pt", weights_only=True
    )
    calibrate_soba_layerwise(
        learning_rate=5e-2,
        num_steps=200,
        num_samples=4,
        init_attention_temperature=init_attention_temperature,
    )

    # calibrate_soba_logits(
    #     learning_rate=1e-3,
    #     num_steps=200,
    #     init_attention_temperature=init_attention_temperature,
    #     num_samples=256,
    #     batch_size=4,
    # )


def main():
    calibrate_sparsemax()
    calibrate_soba()


if __name__ == "__main__":
    main()
