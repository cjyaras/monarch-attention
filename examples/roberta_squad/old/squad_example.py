import torch
from common.utils import get_device
from data import squad_dataloader
from models import AttentionType, PadType, get_config, get_model
from transformers.utils import logging

logging.set_verbosity_error()


@torch.no_grad()
def main():

    device = get_device()
    inputs = next(iter(squad_dataloader(batch_size=16)))

    # Softmax
    config = get_config()
    config.attention_type = AttentionType.softmax
    model = get_model(config, device)

    outputs = model(**inputs)
    print(outputs.start_logits.argmax(-1), outputs.end_logits.argmax(-1))

    # Sparsemax
    config = get_config()
    config.attention_type = AttentionType.sparsemax
    config.scale_attention_temperature = True
    model = get_model(config, device)

    outputs = model(**inputs)
    print(outputs.start_logits.argmax(-1), outputs.end_logits.argmax(-1))

    # Efficient
    config.attention_type = AttentionType.hybrid
    config.scale_attention_temperature = True
    config.efficient_attention_num_steps = 4
    config.efficient_attention_step_size = 5.0
    config.efficient_attention_block_size = 16
    config.efficient_attention_pad_type = PadType.post
    config.hybrid_attention_layers = [8, 9, 10, 11]
    model = get_model(config, device)

    outputs = model(**inputs)
    print(outputs.start_logits.argmax(-1), outputs.end_logits.argmax(-1))


if __name__ == "__main__":
    main()
