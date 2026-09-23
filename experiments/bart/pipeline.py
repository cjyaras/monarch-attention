from transformers.pipelines import PIPELINE_REGISTRY, pipeline

from experiments.common.summarization_pipeline import SummarizationPipeline
from experiments.common.utils import get_device

from experiments.bart.config import CustomBartConfig
from experiments.bart.model import CustomBartForConditionalGeneration, get_model
from experiments.bart.processor import get_processor

from transformers import AutoConfig, AutoModelForSeq2SeqLM


AutoConfig.register("custom_bart", CustomBartConfig)
AutoModelForSeq2SeqLM.register(
    CustomBartConfig,
    CustomBartForConditionalGeneration,
)

class CustomSummarizationPipeline(SummarizationPipeline):
    return_name = "summary"

PIPELINE_REGISTRY.register_pipeline(
    "custom-summarization",
    pipeline_class=CustomSummarizationPipeline,
    pt_model=CustomBartForConditionalGeneration,
)


def get_pipeline(
    config: CustomBartConfig,
    batch_size: int = 1,
    max_length: int = 8192,
    model_checkpoint_path: str = "experiments/bart/finetuned/output/",
) -> CustomSummarizationPipeline:
    pipe = pipeline(
        "custom-summarization",
        model=get_model(
            config,
            model_checkpoint_path=model_checkpoint_path,
        ),
        device=get_device(),
        tokenizer=get_processor(max_length),
        batch_size=batch_size,
        pipeline_class=CustomSummarizationPipeline,
    )
    assert isinstance(pipe, CustomSummarizationPipeline)
    return pipe
