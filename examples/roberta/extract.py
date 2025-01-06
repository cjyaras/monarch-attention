from math import sqrt
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from entmax import sparsemax
from matplotlib import pyplot as plt
from roberta.models import CustomRobertaConfig, CustomRobertaForMaskedLM
from transformers import RobertaTokenizer, pipeline
from transformers.utils import logging

from sobalib.layers import BlockDiagLowRankMHA, LowRankMHA, MonarchMHA

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("/Users/cjyaras/Desktop/soba/examples/roberta/text.txt", "r") as f:
    text = f.read()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
base_config = CustomRobertaConfig.from_pretrained("mtreviso/sparsemax-roberta")

config = CustomRobertaConfig.from_dict(base_config.to_dict())
config.attention_type = "sparsemax"
model = CustomRobertaForMaskedLM.from_pretrained(
    "mtreviso/sparsemax-roberta", config=config
)
model = model.to(device)  # type: ignore


def register_qk_hook(
    model: CustomRobertaForMaskedLM,
    all_layer_intermediates: List[Dict[str, torch.Tensor]],
):
    layers = model.roberta.encoder.layer

    for layer_idx in range(len(layers)):
        attn_layer = layers[layer_idx].attention.self

        def query_hook(_layer_idx):
            def hook(module, input, output):
                all_layer_intermediates[_layer_idx].update(
                    {"query": attn_layer.transpose_for_scores(output)}
                )

            return hook

        def key_hook(_layer_idx):
            def hook(module, input, output):
                all_layer_intermediates[_layer_idx].update(
                    {"key": attn_layer.transpose_for_scores(output)}
                )

            return hook

        attn_layer.query.register_forward_hook(query_hook(layer_idx))
        attn_layer.key.register_forward_hook(key_hook(layer_idx))


all_layer_intermediates = [{} for _ in range(config.num_hidden_layers)]
register_qk_hook(model, all_layer_intermediates)

with torch.no_grad():
    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
    unmasker(text)

query = torch.stack(
    [layer_intermediates["query"] for layer_intermediates in all_layer_intermediates],
    dim=0,
).transpose(1, 0)

key = torch.stack(
    [layer_intermediates["key"] for layer_intermediates in all_layer_intermediates],
    dim=0,
).transpose(1, 0)


with torch.no_grad():
    # layer, head = 0, 0
    layer, head = 5, 5

    # query = query[0, layer, head][1:][None, None]
    # key = key[0, layer, head][1:][None, None]

    query = query[0, layer, head][None, None]
    key = key[0, layer, head][None, None]

    attn_scores = (query @ key.transpose(-1, -2) / sqrt(query.shape[-1]))[0, 0]
    # efficient_attn = MonarchMHA(16, 100, 1e4, "pre")
    efficient_attn = BlockDiagLowRankMHA(16, 8, 10, 5e4, "pre")
    # efficient_attn = LowRankMHA(16, 100, 8e4)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(sparsemax(attn_scores))  # type: ignore
    ax[1].imshow(efficient_attn.get_matrix(query, key)[0, 0])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()  # type: ignore
    plt.show()
