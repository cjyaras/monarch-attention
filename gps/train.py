import torch

from gps.config import CustomGPSConfig
from gps.data import get_processed_dataset
from gps.model import get_model

Tensor = torch.Tensor

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-5
NUM_STEPS: int = 1000


@torch.no_grad()
def compute_accuracy(
    predictions: Tensor, labels: Tensor, mask: Tensor
) -> tuple[int, int]:
    predicted_classes = predictions[mask].argmax(dim=1)
    correct_predictions = (predicted_classes == labels[mask]).sum().item()
    total_samples = mask.sum().item()
    return correct_predictions, total_samples  # pyright: ignore[reportReturnType]


def main():
    config = CustomGPSConfig()
    model = get_model(config, pretrained=False)
    dataset = get_processed_dataset()

    node_features: Tensor = dataset.node_features
    edge_index: Tensor = dataset.edge_index
    labels: Tensor = dataset.labels
    train_mask: Tensor = dataset.train_mask
    val_mask: Tensor = dataset.val_mask

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        outputs = model(node_features, edge_index)
        loss: Tensor = loss_fn(outputs[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        train_correct, train_total = compute_accuracy(outputs, labels, train_mask)
        train_accuracy = train_correct / train_total
        print(
            f"Step: {step:02d} | Loss: {loss.item():.4f} | Train Accuracy: {train_accuracy:.4f} | ({train_correct}/{train_total})"
        )

    torch.save(model.state_dict(), "gps/gps_model.pth")


if __name__ == "__main__":
    main()
