# from math import inf, sqrt
# from typing import List

# import torch
# from common.utils import get_device
# from config import get_config
# from data import get_dataset
# from entmax import sparsemax
# from extract import extract_query_key
# from torch.nn.functional import softmax
# from tqdm.auto import tqdm

# Tensor = torch.Tensor

# NUM_SAMPLES = 128
# SEARCH_RANGE = (1.0, 20.0)
# SEARCH_STEPS = 10

# # TODO: Deal with padding

# @torch.no_grad()
# def calibrate_sparsemax_temperature(
#     query_list: List[Tensor], key_list: List[Tensor], attention_temperature_vals: Tensor
# ) -> Tensor:
#     """
#     query_list: List of [num_layers, num_heads, seq_len, dim_per_head]
#     key_list: List of [num_layers, num_heads, seq_len, dim_per_head]
#     attention_temperature_vals: [num_temperatures]

#     returns: [num_layers, num_heads]
#     """
#     num_layers, num_heads, seq_len, dim_per_head = query_list[0].shape
#     differences = torch.zeros(
#         num_layers,
#         num_heads,
#         len(attention_temperature_vals),
#         device=query_list[0].device,
#     )

#     for i in tqdm(range(len(query_list))):
#         query = query_list[i]
#         key = key_list[i]
#         attn_weights = query @ key.transpose(-1, -2) / sqrt(dim_per_head)
#         softmax_attn_weights = softmax(attn_weights, dim=-1)[..., None, :, :]
#         sparsemax_attn_weights = sparsemax(
#             attn_weights[..., None, :, :] / attention_temperature_vals[:, None, None]
#         )
#         attn_weights_diff = torch.flatten(
#             softmax_attn_weights - sparsemax_attn_weights, start_dim=-2
#         )
#         differences += torch.linalg.norm(attn_weights_diff, ord=inf, dim=-1)

#     optimal_temperature_idx = differences.min(dim=-1)[1]
#     optimal_temperature = attention_temperature_vals[optimal_temperature_idx]
#     return optimal_temperature


# @torch.no_grad()
# def main():
#     config = get_config()
#     device = get_device()
#     dataset = get_dataset(num_samples=NUM_SAMPLES)

#     all_query, all_key = extract_query_key(config, dataset)

#     optimal_temperature = calibrate_sparsemax_temperature(
#         all_query,
#         all_key,
#         torch.linspace(*SEARCH_RANGE, SEARCH_STEPS).to(device),
#     )

#     torch.save(
#         {
#             f"vit.encoder.layer.{i}.attention.attention.attention_temperature": optimal_temperature[
#                 i
#             ]
#             for i in range(len(optimal_temperature))
#         },
#         "vit/sparsemax_temperature_2.pt",
#     )


# if __name__ == "__main__":
#     main()
