import torch

from gps_pubmed.config import CustomGPSConfig
from gps_pubmed.data import get_processed_data
from gps_pubmed.evaluation import compute_accuracy
from gps_pubmed.model import get_model

Tensor = torch.Tensor

LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-5
NUM_STEPS: int = 500


def main():
    config = CustomGPSConfig()
    model = get_model(config, pretrained=False)
    data = get_processed_data()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    for step in range(NUM_STEPS):
        model.train()
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index)
        loss: Tensor = loss_fn(outputs[data.train_mask], data.y[data.train_mask])  # type: ignore
        loss.backward()
        optimizer.step()

        model.eval()
        accuracy = compute_accuracy(model, data)
        print(f"Step: {step:02d} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "gps_pubmed/gps_model.pt")


if __name__ == "__main__":
    main()
