import torch

from gps.config import CustomGPSConfig
from gps.data import get_processed_dataset
from gps.evaluation import compute_accuracy
from gps.model import get_model

Tensor = torch.Tensor

LEARNING_RATE: float = 5e-4
WEIGHT_DECAY: float = 1e-5
NUM_STEPS: int = 1500


def main():
    config = CustomGPSConfig()
    model = get_model(config, pretrained=False)
    dataset = get_processed_dataset()

    node_features: Tensor = dataset.node_features
    edge_index: Tensor = dataset.edge_index
    labels: Tensor = dataset.labels
    train_mask: Tensor = dataset.train_mask

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

        accuracy = compute_accuracy(model, dataset)
        print(f"Step: {step:02d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "gps/gps_model.pth")


if __name__ == "__main__":
    main()
