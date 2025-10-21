import torch
import torch_geometric.data
import torch_geometric.datasets
import torch_geometric.transforms

from common.utils import get_device


def get_processed_data() -> torch_geometric.data.Data:
    device = get_device()
    transform = torch_geometric.transforms.AddLaplacianEigenvectorPE(
        k=4, attr_name=None
    )
    dataset = torch_geometric.datasets.Planetoid(
        ".", name="PubMed", pre_transform=transform, split="full"
    )
    data = dataset[0]
    assert isinstance(data, torch_geometric.data.Data)
    return data.to(device)


if __name__ == "__main__":
    data = get_processed_data()
    print(len(data.x))
    print(data.train_mask.sum().item())
    print(data.val_mask.sum().item())
    print(data.test_mask.sum().item())
