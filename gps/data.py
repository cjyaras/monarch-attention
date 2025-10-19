from dataclasses import dataclass

import torch
import torch_geometric.data
import torch_geometric.datasets
import torch_geometric.utils

from common.utils import get_device, move

Tensor = torch.Tensor


@dataclass
class ActorData:
    node_features: Tensor
    edge_index: Tensor
    labels: Tensor
    train_mask: Tensor
    val_mask: Tensor


def get_processed_dataset() -> ActorData:
    split_version: int = 0
    device = get_device()
    data = torch_geometric.datasets.Actor(".")[0]

    assert isinstance(data, torch_geometric.data.Data)
    assert isinstance(data.x, Tensor)
    assert isinstance(data.edge_index, Tensor)
    assert isinstance(data.y, Tensor)

    node_features = move(data.x, device)
    edge_index = move(torch_geometric.utils.to_undirected(data.edge_index), device)
    labels = move(data.y, device)
    train_mask = move(data.train_mask[:, split_version], device)
    val_mask = move(data.val_mask[:, split_version], device)

    return ActorData(
        node_features=node_features,
        edge_index=edge_index,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
    )
