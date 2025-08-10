from typing import Optional, Union

from transformers.modelcard import ModelCard
from transformers.pipelines import PIPELINE_REGISTRY, pipeline
from transformers.pipelines.text2text_generation import (
    SummarizationPipeline,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from common.utils import get_device

from bart.config import CustomBartConfig
from bart.model import CustomBartForConditionalGeneration, get_model
from bart.processor import get_processor

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
    model_checkpoint_path: str = "./bart/finetuned/output/",
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
        torch_dtype="bfloat16",
        pipeline_class=CustomSummarizationPipeline,
    )
    assert isinstance(pipe, CustomSummarizationPipeline)
    return pipe
