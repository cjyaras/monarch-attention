from transformers.pipelines import PIPELINE_REGISTRY, pipeline
from transformers.pipelines.image_classification import ImageClassificationPipeline
from transformers.utils.import_utils import requires_backends

from common.utils import get_device
from experiments.vit.config import CustomViTConfig
from experiments.vit.model import CustomViTForImageClassification, get_model
from experiments.vit.processor import get_processor


class CustomImageClassificationPipeline(ImageClassificationPipeline):

    def __init__(self, *args, **kwargs):
        super(ImageClassificationPipeline, self).__init__(*args, **kwargs)
        requires_backends(self, "vision")


PIPELINE_REGISTRY.register_pipeline(
    "custom-image-classification",
    pipeline_class=CustomImageClassificationPipeline,
    pt_model=CustomViTForImageClassification,
)


def get_pipeline(
    config: CustomViTConfig,
    batch_size: int = 1,
) -> CustomImageClassificationPipeline:
    pipe = pipeline(
        "custom-image-classification",
        model=get_model(config),
        device=get_device(),
        image_processor=get_processor(),
        batch_size=batch_size,
        torch_dtype="float16",
    )
    assert isinstance(pipe, CustomImageClassificationPipeline)
    return pipe
