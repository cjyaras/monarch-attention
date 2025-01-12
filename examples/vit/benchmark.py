import torch
from common.utils import get_device, move
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm
from vit.data import load_imagenet_dataloader
from vit.models import CustomViTConfig, CustomViTForImageClassification


@torch.no_grad()
def main():
    device = get_device()
    dataloader = load_imagenet_dataloader(batch_size=4)

    config = CustomViTConfig.from_pretrained("google/vit-base-patch16-224")
    config.attention_type = "softmax"
    model = CustomViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", config=config
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
            k=5,
            labels=list(range(1000)),
            normalize=False,
        )
        total = total + labels.size(0)
        total_correct = total_correct + num_correct

    print(f"Top-5 accuracy: {total_correct / total}")


if __name__ == "__main__":
    main()
