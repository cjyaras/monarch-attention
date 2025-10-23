import torch

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


class Evaluator:

    def __init__(self, save_dir: str):
        self.data = get_processed_data()
        self.logger = Logger(save_dir)

    def benchmark_flops(self, model: TokenGTModel, config: CustomTokenGTConfig) -> int:
        from torchtnt.utils.flops import FlopTensorDispatchMode

        with FlopTensorDispatchMode(model) as ftdm:
            model(
                node_features=self.data.x,
                positional_encoding=self.data.pe,
                edge_index=self.data.edge_index,
            )
            return sum(
                ftdm.flop_counts[f"encoder.layers.{i}.self_attn.attn_module"][
                    "bmm.default"
                ]
                for i in range(config.num_layers)
            )

    def evaluate(self, config: CustomTokenGTConfig) -> dict[str, float]:
        model = get_model(config, pretrained=True)
        model.eval()
        flop_count = self.benchmark_flops(model, config)
        with torch.no_grad():
            outputs = model(
                node_features=self.data.x,
                positional_encoding=self.data.pe,
                edge_index=self.data.edge_index,
            )[self.data.val_mask]
        accuracy = compute_accuracy(
            outputs,
            self.data.y[self.data.val_mask],  # type: ignore
        )
        result = {
            "accuracy": accuracy,
            "total_attention_bmm_flops": flop_count,
        }
        return result

    def evaluate_and_save(self, config: CustomTokenGTConfig) -> str:
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name
