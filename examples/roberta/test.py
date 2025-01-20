from datasets import load_dataset

ds = load_dataset("beans")

from transformers import ViTFeatureExtractor

model_name_or_path = "google/vit-base-patch16-224-in21k"

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)


def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["labels"]
    return inputs


ds = ds.with_transform(transform)


# from transformers import ViTFeatureExtractor

# model_name_or_path = "google/vit-base-patch16-224-in21k"
# feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)


# def transform(example_batch):
#     print(example_batch)
#     # Take a list of PIL images and turn them to pixel values
#     # inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")

#     # Don't forget to include the labels!
#     inputs["labels"] = example_batch["labels"]
#     # return inputs


# prepared_ds = ds.with_transform(transform)

# import torch


def collate_fn(batch):
    print(batch)
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


import numpy as np
from evaluate import load

metric = load("accuracy")


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


from transformers import ViTForImageClassification

labels = ds["train"].features["labels"].names

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./vit-base-beans-demo-v5", remove_unused_columns=False
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    # data_collator=collate_fn,
    # compute_metrics=compute_metrics,
    # train_dataset=prepared_ds["train"],
    # eval_dataset=prepared_ds["validation"],
    # tokenizer=feature_extractor,
)


metrics = trainer.evaluate(ds["validation"])
print(metrics)
# trainer.log_metrics("eval", metrics)
# trainer.save_metrics("eval", metrics)
