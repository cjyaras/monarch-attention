import torch

from tokengt.config import AttentionType, CustomTokenGTConfig
from tokengt.data import get_processed_data
from tokengt.evaluation import compute_accuracy
from tokengt.model import PRETRAINED_PATH, get_model

Tensor = torch.Tensor

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-3
NUM_STEPS: int = 500
EVAL_FREQ: int = 1


def main():
    config = CustomTokenGTConfig()
    config.attention_type = AttentionType.nystromformer
    config.rank = 64
    config.num_steps = 1
    config.block_size = 300
    model = get_model(config, pretrained=False)
    data = get_processed_data(config.pos_emb_dims)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    for step in range(NUM_STEPS):
        model.train()
        optimizer.zero_grad()
        outputs = model(
            node_features=data.x,
            positional_encoding=data.pe,
            edge_index=data.edge_index,
        )
        loss: Tensor = loss_fn(outputs[data.train_mask], data.y[data.train_mask])  # type: ignore
        loss.backward()
        optimizer.step()

        if (step + 1) % EVAL_FREQ == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(
                    node_features=data.x,
                    positional_encoding=data.pe,
                    edge_index=data.edge_index,
                )
            accuracy = compute_accuracy(outputs[data.val_mask], data.y[data.val_mask])  # type: ignore
            print(
                f"Step: {step:02d} | Loss: {loss.item():.4f} | Val Accuracy: {accuracy:.4f}"
            )
        else:
            print(f"Step: {step:02d} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), PRETRAINED_PATH)


if __name__ == "__main__":
    main()
