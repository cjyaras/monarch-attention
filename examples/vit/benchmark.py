import torch
from common.utils import get_device, move
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from vit.data import load_imagenet_dataloader
from vit.models import CustomViTConfig, CustomViTForImageClassification


@torch.no_grad()
def evaluate(config, top_k=5):
    device = get_device()
    dataloader = load_imagenet_dataloader(batch_size=4)

    model = CustomViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", config=config
    )
    if config.attention_type == "monarch":
        model.load_state_dict(
            torch.load("vit/sparsemax_temperature.pt", weights_only=True), strict=False
        )
    model = model.to(device)  # type: ignore

    total = 0
    total_correct = 0

    for batch in tqdm(dataloader):
        images, labels = move(batch["image"], device), move(batch["label"], device)
        outputs = model(**images).logits
        num_correct = top_k_accuracy_score(
            labels.cpu().numpy(),
            outputs.cpu().numpy(),
            k=top_k,
            labels=list(range(1000)),
            normalize=False,
        )
        total = total + labels.size(0)
        total_correct = total_correct + num_correct

    print(f"{config.attention_type} Top-{top_k} accuracy: {total_correct / total}")


def main():

    # Softmax
    config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")
    assert isinstance(config, CustomViTConfig)
    config.attention_type = "softmax"
    evaluate(config)

    # Monarch
    config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")
    assert isinstance(config, CustomViTConfig)
    config.attention_type = "monarch"
    config.scale_attention_temperature = True
    config.efficient_attention_num_steps = 3
    config.efficient_attention_step_size = 2.5
    config.efficient_attention_block_size = 14
    evaluate(config)


if __name__ == "__main__":
    main()
