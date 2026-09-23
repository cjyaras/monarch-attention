from transformers.modeling_utils import PreTrainedModel
from transformers.pipelines import PIPELINE_REGISTRY, pipeline
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from experiments.common.qa_pipeline import (
    QuestionAnsweringArgumentHandler,
    QuestionAnsweringPipeline,
)
from experiments.common.utils import get_device
from experiments.roberta.config import CustomRobertaConfig
from experiments.roberta.model import CustomRobertaForQuestionAnswering, get_model
from experiments.roberta.processor import get_processor


class CustomQuestionAnsweringPipeline(QuestionAnsweringPipeline):

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        task: str = "",
        **kwargs,
    ):
        # Skip QuestionAnsweringPipeline.__init__ to avoid its model type check
        super(QuestionAnsweringPipeline, self).__init__(
            model=model,
            tokenizer=tokenizer,
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
        dtype="float16",
    )
    assert isinstance(pipe, CustomQuestionAnsweringPipeline)
    return pipe
