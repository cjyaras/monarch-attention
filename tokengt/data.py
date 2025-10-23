from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from tokengt.config import POS_EMB_DIMS


def get_processed_data() -> Data:
    transform = AddLaplacianEigenvectorPE(k=POS_EMB_DIMS, attr_name="pe")
    dataset = Planetoid("data/", name="PubMed", transform=transform, split="full")
    data = dataset[0]
    return data  # pyright: ignore[reportReturnType]


def get_input_output_dims(data: Data) -> tuple[int, int]:
    input_dim = data.x.size(1)  # type: ignore
    output_dim = int(data.y.max().item()) + 1  # type: ignore
    return input_dim, output_dim
