# from typing import Dict

# import evaluate
# from data import GlueTaskName

# GLUE_METRIC_DICT = {
#     "cola": "matthews_correlation",
#     "sst2": "accuracy",
#     "stsb": "pearson",
#     "qqp": "accuracy",
#     "mrpc": "accuracy",
#     "mnli": "accuracy",
#     "qnli": "accuracy",
#     "rte": "accuracy",
# }


# class GlueMetric:

#     def __init__(self, task_name: GlueTaskName):
#         self.eval_metric = evaluate.load("glue", task_name)
#         self.is_regression = task_name == "stsb"

#     def add_batch(self, logits, labels):
#         predictions = logits[..., 0] if self.is_regression else logits.argmax(dim=-1)
#         self.eval_metric.add_batch(predictions=predictions, references=labels)

#     def compute(self) -> Dict:
#         result = self.eval_metric.compute()
#         assert result is not None
#         result = {k: round(v * 100, 2) for k, v in result.items()}
#         return result


# class SquadMetric:

#     def __init__(self):
#         self.eval_metric = evaluate.load("squad_v2")
