import os
from typing import Dict, Optional

import torch
from tqdm.auto import tqdm

from common.baselines import Softmax, Sparsemax
from common.soba import PadType, SobaMonarch, SobaMonarchV2
from common.utils import get_device, maybe_compile
from vit.config import get_config
from vit.extract import extract_query_key

Tensor = torch.Tensor
Params = Dict[str, Tensor]

kl_div = lambda x, y: torch.nn.functional.kl_div(
    x.log_softmax(-1), y.log_softmax(-1), reduction="batchmean", log_target=True
)


def get_attn_module_path(layer: int):
    return f"vit.encoder.layer.{layer}.attention.attention.attn_module"


def calibrate_sparsemax(
    learning_rate: float,
    num_steps: int,
    num_samples: Optional[int] = None,
) -> Params:

    device = get_device()
    config = get_config()

    all_query, all_key = extract_query_key(
        config, num_samples=num_samples, split="train"
    )
    query_per_layer = torch.unbind(all_query.transpose(1, 0))
    key_per_layer = torch.unbind(all_key.transpose(1, 0))

    softmax = Softmax().to(device)
    sparsemax_params = {}

    for i in (pbar := tqdm(range(config.num_hidden_layers))):
        pbar.set_description(f"Layer {i}")
        query = query_per_layer[i]
        key = key_per_layer[i]
        sparsemax = Sparsemax(config.num_attention_heads).to(device)
        optimizer = torch.optim.Adam(sparsemax.parameters(), lr=learning_rate)

        for _ in (pbar2 := tqdm(range(num_steps), leave=True)):
            optimizer.zero_grad()
            softmax_out = softmax.get_matrix(query, key)
            sparsemax_out = sparsemax.get_matrix(query, key)
            loss = kl_div(sparsemax_out, softmax_out)
            loss.backward()
            optimizer.step()
            pbar2.set_description(f"Loss: {loss.item():0.2e}")

        sparsemax_params[".".join([get_attn_module_path(i), "attention_scale"])] = (
            sparsemax.attention_scale.detach()
        )

    return sparsemax_params


def calibrate_soba(
    learning_rate: float,
    num_steps: int,
    params_path: str,
    num_samples: Optional[int] = None,
) -> Params:

    device = get_device()
    config = get_config()

    all_query, all_key = extract_query_key(
        config, num_samples=num_samples, split="train"
    )
    query_per_layer = torch.unbind(all_query.transpose(1, 0))
    key_per_layer = torch.unbind(all_key.transpose(1, 0))

    sparsemax_params = torch.load(params_path, weights_only=True)

    softmax = Softmax().to(device)
    soba_params = {}

    for i in (pbar := tqdm(range(config.num_hidden_layers))):
        pbar.set_description(f"Layer {i}")

        query = query_per_layer[i]
        key = key_per_layer[i]
        soba = SobaMonarchV2(
            block_size=14,
            num_steps=3,
            num_heads=config.num_attention_heads,
            pad_type=PadType.pre,
        ).to(device)
        maybe_compile(soba)

        soba.load_state_dict(
            {
                k.split(".")[-1]: v
                for k, v in sparsemax_params.items()
                if str(i) in k and "attn_module" in k
            },
            strict=False,
        )

        for k, v in soba.named_parameters():
            v.requires_grad = True

        trainable_params = filter(lambda p: p.requires_grad, soba.parameters())

        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

        for _ in (pbar2 := tqdm(range(num_steps), leave=True)):
            optimizer.zero_grad()
            softmax_out = softmax.get_matrix(query, key)
            soba_out = soba.get_matrix(query, key)
            loss = kl_div(soba_out, softmax_out)
            loss.backward()
            optimizer.step()
            pbar2.set_description(f"Loss: {loss.item():0.2e}")

        soba_params[".".join([get_attn_module_path(i), "attention_scale"])] = (
            soba.attention_scale.detach()
        )

        # soba_params[".".join([get_attn_module_path(i), "step_size"])] = (
        #     soba.step_size.detach()
        # )

    return soba_params


SPARSEMAX_PARAMS_PATH = "vit/sparsemax_params.pt"
SOBA_PARAMS_PATH = "vit/soba_params.pt"

torch.autograd.set_detect_anomaly(True)


def main():
    if not os.path.exists(SPARSEMAX_PARAMS_PATH):
        sparsemax_params = calibrate_sparsemax(
            learning_rate=0.1, num_steps=50, num_samples=8
        )
        torch.save(sparsemax_params, SPARSEMAX_PARAMS_PATH)

    if not os.path.exists(SOBA_PARAMS_PATH):
        soba_params = calibrate_soba(
            learning_rate=0.1,
            num_steps=50,
            params_path=SPARSEMAX_PARAMS_PATH,
            num_samples=8,
        )
        torch.save(soba_params, SOBA_PARAMS_PATH)


if __name__ == "__main__":
    main()
