import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from common.baselines import Softmax, Sparsemax
from common.soba import PadType, SobaMonarch
from vit.config import AttentionType, get_config
from vit.data import get_processed_dataset
from vit.extract import extract_query_key
from vit.model import get_model

Tensor = torch.Tensor
Params = Dict[str, Tensor]


def get_attn_module_path(layer: int):
    return f"vit.encoder.layer.{layer}.attention.attention.attn_module"


def calibrate_sparsemax_layerwise(
    learning_rate: float,
    num_steps: int,
    num_samples: Optional[int] = None,
) -> Params:

    config = get_config()

    all_query, all_key = extract_query_key(config, num_samples=num_samples)
    query_per_layer = torch.unbind(all_query.transpose(1, 0))
    key_per_layer = torch.unbind(all_key.transpose(1, 0))

    softmax = Softmax()
    sparsemax_params = {}

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

        sparsemax_params[".".join([get_attn_module_path(i), "log_attention_scale"])] = (
            sparsemax.log_attention_scale.detach()
        )

        plt.plot(loss_vals)
        plt.show()

    return sparsemax_params


def calibrate_sparsemax_logits(
    learning_rate: float,
    num_steps: int,
    params_path: str,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Params:

    # Load softmax model
    config = get_config()
    softmax_model = get_model(config)

    # Load sparsemax model
    config = get_config()
    config.attn_module_save_path = params_path
    config.attention_type = AttentionType.sparsemax
    sparsemax_model = get_model(config)

    # Freeze all parameters (except ones in attn_module)
    for k, v in sparsemax_model.named_parameters():
        v.requires_grad = "attn_module" in k

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

    sparsemax_params = {
        k: v for k, v in sparsemax_model.state_dict().items() if "attn_module" in k
    }
    return sparsemax_params


def calibrate_soba_layerwise(
    learning_rate: float,
    num_steps: int,
    params_path: str,
    num_samples: Optional[int] = None,
) -> Params:

    config = get_config()

    all_query, all_key = extract_query_key(config, num_samples=num_samples)
    query_per_layer = torch.unbind(all_query.transpose(1, 0))
    key_per_layer = torch.unbind(all_key.transpose(1, 0))

    sparsemax_params = torch.load(params_path, weights_only=True)

    softmax = Softmax()
    soba_params = {}

    for i in range(config.num_hidden_layers):
        query = query_per_layer[i]
        key = key_per_layer[i]
        soba = SobaMonarch(
            block_size=14,
            num_steps=3,
            num_heads=config.num_attention_heads,
            pad_type=PadType.pre,
        )

        soba.load_state_dict(
            {
                k.split(".")[-1]: v
                for k, v in sparsemax_params.items()
                if str(i) in k and "attn_module" in k
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

        soba_params[".".join([get_attn_module_path(i), "log_attention_scale"])] = (
            soba.log_attention_scale.detach()
        )

        soba_params[".".join([get_attn_module_path(i), "log_step_size"])] = (
            soba.log_step_size.detach()
        )

        plt.plot(loss_vals)
        plt.show()

    return soba_params


def calibrate_soba_logits(
    learning_rate: float,
    num_steps: int,
    params_path: str,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Params:

    # Load softmax model
    config = get_config()
    softmax_model = get_model(config)

    # Load soba model
    config = get_config()
    config.attention_type = AttentionType.soba_monarch
    config.attn_module_save_path = params_path
    config.num_steps = 3
    config.block_size = 14
    soba_model = get_model(config)

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

    soba_params = {
        k: v for k, v in soba_model.state_dict().items() if "attn_module" in k
    }
    return soba_params


SPARSEMAX_PARAMS_PATH = "vit/sparsemax_params_layerwise.pt"
SOBA_PARAMS_LAYERWISE_PATH = "vit/soba_params_layerwise.pt"
SOBA_PARAMS_LOGITS_PATH = "vit/soba_params_logits.pt"


def calibrate_sparsemax():
    if not os.path.exists(SPARSEMAX_PARAMS_PATH):
        sparsemax_params = calibrate_sparsemax_layerwise(
            learning_rate=5e-2, num_steps=200, num_samples=4
        )
        torch.save(sparsemax_params, SPARSEMAX_PARAMS_PATH)


def calibrate_soba():
    assert os.path.exists(SPARSEMAX_PARAMS_PATH)

    if not os.path.exists(SOBA_PARAMS_LAYERWISE_PATH):
        soba_params = calibrate_soba_layerwise(
            learning_rate=5e-2,
            num_steps=300,
            params_path=SPARSEMAX_PARAMS_PATH,
            num_samples=4,
        )
        torch.save(soba_params, SOBA_PARAMS_LAYERWISE_PATH)

    if not os.path.exists(SOBA_PARAMS_LOGITS_PATH):

        soba_params = calibrate_soba_logits(
            learning_rate=1e-2,
            num_steps=1000,
            params_path=SOBA_PARAMS_LAYERWISE_PATH,
            num_samples=16,
            batch_size=4,
        )
        torch.save(soba_params, SOBA_PARAMS_LOGITS_PATH)


def main():
    calibrate_sparsemax()
    calibrate_soba()


if __name__ == "__main__":
    main()
