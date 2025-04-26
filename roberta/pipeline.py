from typing import Optional, Union

from transformers import ModelCard, PreTrainedTokenizer, pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.question_answering import (
    QuestionAnsweringArgumentHandler,
    QuestionAnsweringPipeline,
)

from common.utils import get_device
from roberta.config import CustomRobertaConfig
from roberta.model import CustomRobertaForQuestionAnswering, get_model
from roberta.processor import get_processor


class CustomQuestionAnsweringPipeline(QuestionAnsweringPipeline):

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],  # type: ignore
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        **kwargs,
    ):
        super(QuestionAnsweringPipeline, self).__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            task=task,
            **kwargs,
        )
        self._args_parser = QuestionAnsweringArgumentHandler()


PIPELINE_REGISTRY.register_pipeline(
    "custom-question-answering",
    pipeline_class=CustomQuestionAnsweringPipeline,
    pt_model=CustomRobertaForQuestionAnswering,
)


def get_pipeline(
    config: CustomRobertaConfig,
    batch_size: int = 1,
) -> CustomQuestionAnsweringPipeline:
    pipe = pipeline(
        "custom-question-answering",
        model=get_model(config),
        device=get_device(),
        tokenizer=get_processor(),
        batch_size=batch_size,
        torch_dtype="float16",
    )
    assert isinstance(pipe, CustomQuestionAnsweringPipeline)
    return pipe
