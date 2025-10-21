import torch
import torch_geometric.data

from common.logging import Logger
from gps_pubmed.config import CustomGPSConfig
from gps_pubmed.data import get_processed_data
from gps_pubmed.model import GPSModel, get_model


@torch.no_grad()
def compute_accuracy(model: GPSModel, data: torch_geometric.data.Data) -> float:
    mask = data.val_mask
    predictions = model(data.x, data.edge_index)[mask]
    predicted_classes = predictions.argmax(dim=1)
    correct_predictions = (predicted_classes == data.y[mask]).sum().item()  # type: ignore
    total_samples = mask.sum().item()
    return 100 * correct_predictions / total_samples


class Evaluator:

    def __init__(self, save_dir: str):
        self.data = get_processed_data()
        self.logger = Logger(save_dir)

    def benchmark_flops(self, model: GPSModel, config: CustomGPSConfig) -> int:
        from torchtnt.utils.flops import FlopTensorDispatchMode

        with FlopTensorDispatchMode(model) as ftdm:
            model(self.data.x, self.data.edge_index)
            return sum(
                ftdm.flop_counts[f"backbone.{i}.attn"]["bmm.default"]
                for i in range(config.num_layers)
            )

    def evaluate(self, config: CustomGPSConfig) -> dict[str, float]:
        model = get_model(config)
        model.eval()
        flop_count = self.benchmark_flops(model, config)
        accuracy = compute_accuracy(model, self.data)
        result = {
            "accuracy": accuracy,
            "total_attention_bmm_flops": flop_count,
        }
        return result

    def evaluate_and_save(self, config: CustomGPSConfig) -> str:
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name
