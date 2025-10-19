import torch

from common.logging import Logger
from gps.config import CustomGPSConfig, get_config
from gps.data import ActorData, get_processed_dataset
from gps.model import GPSModel, get_model


@torch.no_grad()
def compute_accuracy(model: GPSModel, data: ActorData) -> float:
    mask = data.train_mask
    predictions = model(data.node_features, data.edge_index)[mask]
    predicted_classes = predictions.argmax(dim=1)
    correct_predictions = (predicted_classes == data.labels[mask]).sum().item()
    total_samples = mask.sum().item()
    return 100 * correct_predictions / total_samples


class Evaluator:

    def __init__(self, save_dir: str):
        self.data = get_processed_dataset()
        self.logger = Logger(save_dir)

    def benchmark_flops(self, model: GPSModel):
        from torchtnt.utils.flops import FlopTensorDispatchMode

        with FlopTensorDispatchMode(model) as ftdm:
            model(self.data.node_features, self.data.edge_index)
            return ftdm.flop_counts["backbone.0.attn"]["bmm.default"]

    def evaluate(self, config: CustomGPSConfig) -> dict[str, float]:
        model = get_model(config)
        flop_count = self.benchmark_flops(model)
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
