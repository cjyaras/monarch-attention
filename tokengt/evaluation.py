import torch
import torch_geometric
from torch_geometric.data import Data

from common.logging import Logger
from tokengt.config import CustomTokenGTConfig
from tokengt.data import get_processed_data
from tokengt.model import TokenGTModel, get_model


@torch.no_grad()
def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    predicted_classes = predictions.argmax(dim=1)
    correct_predictions = (predicted_classes == labels).sum().item()
    total_samples = labels.size(0)
    return 100 * correct_predictions / total_samples
