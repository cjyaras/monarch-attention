from datasets import load_dataset
from evaluate import evaluator
from transformers import pipeline

data = load_dataset("imagenet-1k", split="validation")


pipe = pipeline(
    task="image-classification", model="facebook/deit-small-distilled-patch16-224"
)

task_evaluator = evaluator("image-classification")
eval_results = task_evaluator.compute(
    model_or_pipeline=pipe,
    data=data,
    metric="accuracy",
    label_mapping=pipe.model.config.label2id,
)
