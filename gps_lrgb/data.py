from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import AddLaplacianEigenvectorPE


def get_processed_dataset(split: str, pe_dims: int) -> LRGBDataset:
    transform = AddLaplacianEigenvectorPE(k=pe_dims, attr_name="pe")
    return LRGBDataset(
        "data/", name="PascalVOC-SP", split=split, pre_transform=transform
    )


def get_processed_dataloader(split: str, pe_dims: int, batch_size: int) -> DataLoader:
    dataset = get_processed_dataset(split, pe_dims)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
