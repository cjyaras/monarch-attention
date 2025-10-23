from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import AddLaplacianEigenvectorPE


def get_processed_data(pos_emb_dims: int) -> Data:
    transform = AddLaplacianEigenvectorPE(k=pos_emb_dims, attr_name="pe")
    # dataset = Planetoid("data/", name="Cora", transform=transform)
    # dataset = Planetoid("data/", name="Cora", transform=transform)
    # dataset = WikipediaNetwork("data/", name="chameleon", transform=transform)
    dataset = Planetoid("data/", name="PubMed", transform=transform, split="full")
    data = dataset[0]
    # data.train_mask = data.train_mask[:, 0]
    # data.val_mask = data.val_mask[:, 0]
    # data.test_mask = data.test_mask[:, 0]
    return data  # pyright: ignore[reportReturnType]


if __name__ == "__main__":
    data = get_processed_data(16)
    print(data)
