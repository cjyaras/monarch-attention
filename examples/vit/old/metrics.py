# from typing import Dict

# import datasets
# import evaluate
# import numpy as np
# import torch

# Tensor = torch.Tensor


# def any_match(predictions, references):
#     predictions = np.array(predictions)
#     references = np.array(references)
#     return np.any(predictions == references[:, None], axis=-1)


# class TopKAccuracy(evaluate.Metric):

#     def _info(self):
#         return evaluate.MetricInfo(
#             description="",
#             citation="",
#             inputs_description="",
#             features=datasets.Features(
#                 {
#                     "predictions": datasets.Sequence(datasets.Value("int32")),
#                     "references": datasets.Value("int32"),
#                 }
#             ),
#             reference_urls=[],
#         )

#     def _compute(self, predictions, references):
#         return {"accuracy": float(any_match(predictions, references).mean())}


# class TopKAccuracyMetric:

#     def __init__(self, top_k: int = 5):
#         self.top_k = top_k
#         self.metric = TopKAccuracy()

#     def add_batch(self, logits: Tensor, labels: Tensor):
#         top_idx = torch.topk(input=logits, k=self.top_k, sorted=False).indices
#         self.metric.add_batch(predictions=top_idx, references=labels)

#     def compute(self) -> Dict:
#         result = self.metric.compute()
#         assert result is not None
#         return result
