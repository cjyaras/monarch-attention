from common.utils import IterableKeyDataset, get_device
from datasets import IterableDataset, load_dataset
from evaluate import evaluator

# from pipeline import register_pipeline
from transformers import ViTImageProcessorFast, pipeline

from examples.vit.model import get_config, get_model

config = get_config()
device = get_device()
model = get_model(config, device)


image_processor = ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")


iterable_dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
assert isinstance(iterable_dataset, IterableDataset)
pipe = pipeline(
    task="no-check-image-classification", model=model, image_processor=image_processor
)


result = pipe(IterableKeyDataset(iterable_dataset, "image"))
assert result is not None
for out in result:
    print(out)
    break


# pipe = pipeline(
#     task="image-classification", model="facebook/deit-small-distilled-patch16-224"
# )

# task_evaluator = evaluator("image-classification")
# eval_results = task_evaluator.compute(
#     model_or_pipeline=pipe,
#     data=data,
#     metric="accuracy",
#     label_mapping=pipe.model.config.label2id,
# )
