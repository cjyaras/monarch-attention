import torch
from tqdm import tqdm

from common.utils import get_device
from gps_lrgb.config import CustomGPSConfig
from gps_lrgb.data import get_processed_dataloader
from gps_lrgb.model import get_model

Tensor = torch.Tensor

LEARNING_RATE: float = 5e-4
NUM_EPOCHS: int = 500
BATCH_SIZE: int = 16


def main():
    device = get_device()
    config = CustomGPSConfig()
    train_dataloader = get_processed_dataloader(
        split="train", pe_dims=config.pe_dims, batch_size=BATCH_SIZE
    )
    val_dataloader = get_processed_dataloader(
        split="val", pe_dims=config.pe_dims, batch_size=BATCH_SIZE
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    model = get_model(config, pretrained=False)
    # print model size
    print(f"Model size: {sum(p.numel() for p in model.parameters())}")

    best_val_loss = float("inf")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()

        total_train_loss = 0
        train_steps = 0

        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(
                data.x, data.pe, data.edge_index, data.edge_attr, data.batch
            )
            loss: Tensor = loss_fn(outputs, data.y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_steps += 1

        # Validation
        model.eval()

        total_val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for data in val_dataloader:
                data = data.to(device)
                outputs = model(
                    data.x, data.pe, data.edge_index, data.edge_attr, data.batch
                )
                loss: Tensor = loss_fn(outputs, data.y)
                total_val_loss += loss.item()
                val_steps += 1

        val_loss = total_val_loss / val_steps

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                f"gps_lrgb/best_model.pt",
            )

        print(
            f"Epoch: {epoch:02d} | Train Loss: {total_train_loss / train_steps:.4f} | Val Loss: {val_loss:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
