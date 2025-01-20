import torch
from common.utils import calibrate_sparsemax_temperature, get_device, move
from data import squad_dataloader
from models import get_config, get_model
from utils import extract_qk


@torch.no_grad()
def main():
    device = get_device()
    inputs = move(next(iter(squad_dataloader(batch_size=4))), device)
    config = get_config()
    model = get_model(config, device)

    query, key = extract_qk(model, inputs)
    optimal_temperature = calibrate_sparsemax_temperature(
        list(torch.unbind(query)),
        list(torch.unbind(key)),
        torch.arange(1, 51, dtype=torch.float).to(device),
    )
    torch.save(
        {
            f"roberta.encoder.layer.{i}.attention.self.attention_temperature": optimal_temperature[
                i
            ]
            for i in range(len(optimal_temperature))
        },
        "roberta/sparsemax_temperature.pt",
    )


if __name__ == "__main__":
    main()
