import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import torch
from common.baselines import Softmax, Sparsemax
from tqdm.auto import tqdm
from vit.config import AttentionType, get_config
from vit.extract import extract_query_key
from vit.model import get_model
from vit.preprocessed_data import get_preprocessed_dataset

Tensor = torch.Tensor


def calibrate_layerwise(
    step_size: float, num_steps: int, num_samples: Optional[int] = None
):
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
            config.num_attention_heads, device=query.device
        )
        attention_temperature_per_layer.requires_grad = True

        optimizer = torch.optim.SGD([attention_temperature_per_layer], lr=step_size)
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


def calibrate_logits(
    step_size: float,
    num_steps: int,
    init_attention_temperature: Optional[Dict[str, Tensor]] = None,
    num_samples: Optional[int] = None,
):

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

    # Optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, sparsemax_model.parameters()), lr=step_size
    )

    # Load dataset
    dataset = get_preprocessed_dataset(
        num_samples=2 * num_samples if num_samples else None
    )
    train_inputs, test_inputs = (
        dataset[: len(dataset) // 2],
        dataset[len(dataset) // 2 :],
    )

    # Extract softmax logits
    with torch.no_grad():
        softmax_train_logits = softmax_model(**train_inputs).logits
        softmax_test_logits = softmax_model(**test_inputs).logits

    train_loss_vals = []
    test_loss_vals = []

    # loss_fn = torch.nn.functional.mse_loss
    loss_fn = torch.nn.functional.cross_entropy

    for _ in tqdm(range(num_steps)):
        optimizer.zero_grad()
        sparsemax_train_logits = sparsemax_model(**train_inputs).logits
        sparsemax_test_logits = sparsemax_model(**test_inputs).logits
        train_loss = loss_fn(sparsemax_train_logits, softmax_train_logits)
        test_loss = loss_fn(sparsemax_test_logits, softmax_test_logits)
        train_loss.backward()
        optimizer.step()

        train_loss_vals.append(train_loss.item())
        test_loss_vals.append(test_loss.item())

    plt.plot(train_loss_vals, label="train")
    plt.plot(test_loss_vals, label="test")
    plt.legend()
    plt.show()


if not os.path.exists("vit/attention_temperature.pt"):
    attention_temperature = calibrate_layerwise(
        step_size=5e3, num_steps=200, num_samples=4
    )
    torch.save(attention_temperature, "vit/attention_temperature.pt")
else:
    attention_temperature = torch.load(
        "vit/attention_temperature.pt", weights_only=True
    )

calibrate_logits(
    step_size=0.5,
    num_steps=100,
    num_samples=16,
    init_attention_temperature=attention_temperature,
)

# @torch.no_grad()
# def calibrate_sparsemax_temperature(
#     query_list: List[Tensor], key_list: List[Tensor], attention_temperature_vals: Tensor
# ) -> Tensor:
#     num_layers, num_heads, seq_len, dim_per_head = query_list[0].shape
#     search_size = len(attention_temperature_vals)
#     differences = torch.zeros(
#         search_size,
#         num_layers * num_heads,
#         device=query_list[0].device,
#     )

#     assert search_size % BATCH_SIZE == 0

#     softmax = Softmax()
#     sparsemax = Sparsemax()

#     for i in tqdm(range(len(query_list))):
#         query = query_list[i].reshape(1, num_layers * num_heads, seq_len, dim_per_head)
#         key = key_list[i].reshape(1, num_layers * num_heads, seq_len, dim_per_head)
#         softmax_attn_probs = softmax.get_matrix(query, key)

#         for j in range(0, search_size, BATCH_SIZE):
#             sparsemax_attn_probs = sparsemax.get_matrix(
#                 query
#                 / attention_temperature_vals[j : j + BATCH_SIZE, None, None, None],
#                 key
#                 / torch.ones_like(
#                     attention_temperature_vals[j : j + BATCH_SIZE, None, None, None]
#                 ),
#             )
#             attn_weights_diff = torch.flatten(
#                 softmax_attn_probs - sparsemax_attn_probs, start_dim=-2
#             )
#             differences[j : j + BATCH_SIZE] += torch.linalg.norm(
#                 attn_weights_diff, ord=ORD, dim=-1
#             )

#     optimal_temperature_idx = differences.min(dim=0)[1]
#     optimal_temperature = attention_temperature_vals[
#         optimal_temperature_idx.reshape(num_layers, num_heads)
#     ]
#     return optimal_temperature


# def calibrate_sparsemax_temperature_step_1():
#     pass


# def calibrate_sparsemax_temperature():

#     # First, load softmax model
#     config = get_config()
#     config.attention_type = "softmax"
