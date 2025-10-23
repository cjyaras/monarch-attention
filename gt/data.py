import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddLaplacianEigenvectorPE


def get_processed_dataset(split: str, pos_enc_dims: int) -> LRGBDataset:
    transform = AddLaplacianEigenvectorPE(k=pos_enc_dims, attr_name="pos_enc")
    return LRGBDataset(
        "data/", name="PascalVOC-SP", split=split, pre_transform=transform
    )


def get_processed_dataloader(
    split: str, pos_enc_dims: int, batch_size: int
) -> DataLoader:
    dataset = get_processed_dataset(split, pos_enc_dims)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))


print(
    next(
        iter(get_processed_dataloader("train", pos_enc_dims=8, batch_size=2))
    ).edge_attr
)

# def get_processed_dataloaders() -> tuple[]:
#     pass


# if __name__ == "__main__":
#     data = get_processed_data()
#     print(len(data.x))
#     print(data.train_mask.sum().item())
#     print(data.val_mask.sum().item())
#     print(data.test_mask.sum().item())
