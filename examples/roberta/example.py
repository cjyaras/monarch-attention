import torch
from roberta.models import CustomRobertaConfig, CustomRobertaForMaskedLM
from torchtnt.utils.flops import FlopTensorDispatchMode
from transformers import RobertaTokenizer, pipeline
from transformers.utils import logging

logging.set_verbosity_error()

use_sparsemax = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("/Users/cjyaras/Desktop/soba/examples/roberta/text.txt", "r") as f:
    text = f.read()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
base_config = CustomRobertaConfig.from_pretrained(
    "mtreviso/sparsemax-roberta" if use_sparsemax else "roberta-base"
)

config = CustomRobertaConfig.from_dict(base_config.to_dict())
config.attention_type = "sparsemax" if use_sparsemax else "softmax"
model = CustomRobertaForMaskedLM.from_pretrained(
    "mtreviso/sparsemax-roberta" if use_sparsemax else "roberta-base", config=config
)
model = model.to(device)  # type: ignore

with FlopTensorDispatchMode(model) as ftdm:

    with torch.no_grad():
        unmasker = pipeline(
            "fill-mask", model=model, tokenizer=tokenizer, device=device
        )
        results = unmasker(text)
        [result.pop("sequence") for result in results]  # type: ignore
        print("Sparsemax:")
        [print(k) for k in results]  # type: ignore

    sparsemax_flops = ftdm.flop_counts["roberta.encoder.layer.0.attention"][
        "bmm.default"
    ]


config = CustomRobertaConfig.from_dict(base_config.to_dict())
config.attention_type = "monarch"
# config.attention_type = "monarch-block-diagonal"
config.efficient_attention_num_steps = 6
config.efficient_attention_step_size = 1.0
config.efficient_attention_block_size = 16
config.efficient_attention_pad_type = "pre"
model = CustomRobertaForMaskedLM.from_pretrained(
    "mtreviso/sparsemax-roberta", config=config
)
model = model.to(device)  # type: ignore

with FlopTensorDispatchMode(model) as ftdm:

    with torch.no_grad():
        unmasker = pipeline(
            "fill-mask", model=model, tokenizer=tokenizer, device=device
        )
        results = unmasker(text)
        [result.pop("sequence") for result in results]  # type: ignore
        print("Monarch:")
        [print(k) for k in results]  # type: ignore

    monarch_flops = ftdm.flop_counts["roberta.encoder.layer.0.attention"]["bmm.default"]

print(
    f"Monarch to Sparsemax FLOP ratio: {monarch_flops:0.2e}/{sparsemax_flops:0.2e} ({monarch_flops / sparsemax_flops * 100:0.2f}%)"
)
