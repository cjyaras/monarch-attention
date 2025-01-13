import torch
from common.utils import get_device, move
from vit.data import imagenet_dataloader
from vit.models import get_config, get_model
from vit.utils import calibrate_sparsemax_temperature, extract_qk


@torch.no_grad()
def main():
    device = get_device()
    inputs = move(next(iter(imagenet_dataloader())), device)
    config = get_config()
    model = get_model(config, device)

    query, key = extract_qk(model, inputs)
    optimal_temperature = calibrate_sparsemax_temperature(
        list(torch.unbind(query)),
        list(torch.unbind(key)),
        torch.linspace(1, 50, 50).to(device),
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
