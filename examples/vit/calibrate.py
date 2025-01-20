from itertools import islice
from math import ceil

import torch
from common.utils import calibrate_sparsemax_temperature, get_device, move
from data import imagenet_dataloader
from models import get_config, get_model
from utils import extract_qk

NUM_SAMPLES = 128
BATCH_SIZE = 4
SEARCH_RANGE = (1.0, 20.0)
SEARCH_STEPS = 100


@torch.no_grad()
def main():
    config = get_config()
    device = get_device()
    model = get_model(config, device)

    all_query = []
    all_key = []

    dataloader = imagenet_dataloader()

    for inputs in islice(dataloader, ceil(NUM_SAMPLES / BATCH_SIZE)):
        inputs = move(inputs, device)
        query, key = extract_qk(model, inputs)
        all_query.extend(list(torch.unbind(query)))
        all_key.extend(list(torch.unbind(key)))

    optimal_temperature = calibrate_sparsemax_temperature(
        all_query,
        all_key,
        torch.linspace(*SEARCH_RANGE, SEARCH_STEPS, dtype=torch.float).to(device),
    )
    torch.save(
        {
            f"vit.encoder.layer.{i}.attention.attention.attention_temperature": optimal_temperature[
                i
            ]
            for i in range(len(optimal_temperature))
        },
        "vit/sparsemax_temperature.pt",
    )


if __name__ == "__main__":
    main()
