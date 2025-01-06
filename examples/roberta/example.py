import torch
from roberta.models import CustomRobertaConfig, CustomRobertaForMaskedLM
from transformers import RobertaTokenizer, pipeline
from transformers.utils import logging

logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("/Users/cjyaras/Desktop/soba/examples/roberta/text_2.txt", "r") as f:
    text = f.read()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
base_config = CustomRobertaConfig.from_pretrained("mtreviso/sparsemax-roberta")

config = CustomRobertaConfig.from_dict(base_config.to_dict())
config.attention_type = "sparsemax"
model = CustomRobertaForMaskedLM.from_pretrained(
    "mtreviso/sparsemax-roberta", config=config
)
model = model.to(device)  # type: ignore

with torch.no_grad():
    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
    results = unmasker(text)
    [result.pop("sequence") for result in results]  # type: ignore
    print("Sparsemax:")
    [print(k) for k in results]  # type: ignore

config = CustomRobertaConfig.from_dict(base_config.to_dict())
# config.attention_type = "monarch"
config.attention_type = "block-diag-low-rank"
config.efficient_attention_num_steps = 100
config.efficient_attention_step_size = 1e5
config.efficient_attention_block_size = 16
config.efficient_attention_rank = 64
config.efficient_attention_pad_type = "pre"
model = CustomRobertaForMaskedLM.from_pretrained(
    "mtreviso/sparsemax-roberta", config=config
)
model = model.to(device)  # type: ignore

with torch.no_grad():
    unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=device)
    results = unmasker(text)
    [result.pop("sequence") for result in results]  # type: ignore
    print("Monarch:")
    [print(k) for k in results]  # type: ignore
