# Adapted from transformers 4.53.3 `pipelines/text2text_generation.py`
# (Apache-2.0, Copyright The HuggingFace Team), which was removed in transformers 5.
# Changes: dropped TensorFlow branches and TranslationPipeline, PyTorch only
# (no `framework` attribute in v5), updated imports, and `_forward` folds
# generation arguments into the generation config.

import copy
import enum
import warnings
from typing import Any, Union

from transformers.generation import GenerationConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
from transformers.pipelines.base import Pipeline, build_pipeline_init_args
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import add_end_docstrings, logging

logger = logging.get_logger(__name__)


class ReturnType(enum.Enum):
    TENSORS = 0
    TEXT = 1


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class Text2TextGenerationPipeline(Pipeline):
    """
    Pipeline for text to text generation using seq2seq models.

    Unless the model you're using explicitly sets these generation parameters in its configuration files
    (`generation_config.json`), the following default values will be used:
    - max_new_tokens: 256
    - num_beams: 4

    Example:

    ```python
    >>> from transformers import pipeline

    >>> generator = pipeline(model="mrm8488/t5-base-finetuned-question-generation-ap")
    >>> generator(
    ...     "answer: Manuel context: Manuel has created RuPERTa-base with the support of HF-Transformers and Google"
    ... )
    [{'generated_text': 'question: Who created the RuPERTa-base?'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial). You can pass text
    generation parameters to this pipeline to control stopping criteria, decoding strategy, and more. Learn more about
    text generation parameters in [Text generation strategies](../generation_strategies) and [Text
    generation](text_generation).

    This Text2TextGenerationPipeline pipeline can currently be loaded from [`pipeline`] using the following task
    identifier: `"text2text-generation"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=text2text-generation). For a list of available
    parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Usage:

    ```python
    text2text_generator = pipeline("text2text-generation")
    text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
    ```"""

    _pipeline_calls_generate = True
    # Make sure the docstring is updated when the default generation config is changed (in all pipelines in this file)
    _default_generation_config = GenerationConfig(
        max_new_tokens=256,
        num_beams=4,
    )

    # Used in the return key of the pipeline.
    return_name = "generated"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_model_type(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES)

    def _sanitize_parameters(
        self,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        truncation=None,
        stop_sequence=None,
        **generate_kwargs,
    ):
        preprocess_params = {}
        if truncation is not None:
            preprocess_params["truncation"] = truncation

        forward_params = generate_kwargs

        postprocess_params = {}
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS if return_tensors else ReturnType.TEXT
        if return_type is not None:
            postprocess_params["return_type"] = return_type

        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        if self.assistant_model is not None:
            forward_params["assistant_model"] = self.assistant_model
        if self.assistant_tokenizer is not None:
            forward_params["tokenizer"] = self.tokenizer
            forward_params["assistant_tokenizer"] = self.assistant_tokenizer

        return preprocess_params, forward_params, postprocess_params

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        return True

    def _parse_and_tokenize(self, *args, truncation):
        prefix = self.prefix if self.prefix is not None else ""
        if isinstance(args[0], list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )
        inputs = self.tokenizer(*args, padding=padding, truncation=truncation, return_tensors="pt")
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs

    def __call__(self, *args: Union[str, list[str]], **kwargs: Any) -> list[dict[str, str]]:
        r"""
        Generate the output text(s) using text(s) given as inputs.

        Args:
            args (`str` or `list[str]`):
                Input text for the encoder.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            truncation (`TruncationStrategy`, *optional*, defaults to `TruncationStrategy.DO_NOT_TRUNCATE`):
                The truncation strategy for the tokenization within the pipeline. `TruncationStrategy.DO_NOT_TRUNCATE`
                (default) will never truncate, but it is sometimes desirable to truncate the input to fit the model's
                max_length instead of throwing an error down the line.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./text_generation)).

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:

            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
        """

        result = super().__call__(*args, **kwargs)
        if (
            isinstance(args[0], list)
            and all(isinstance(el, str) for el in args[0])
            and all(len(res) == 1 for res in result)
        ):
            return [res[0] for res in result]
        return result

    def preprocess(self, inputs, truncation=TruncationStrategy.DO_NOT_TRUNCATE, **kwargs):
        inputs = self._parse_and_tokenize(inputs, truncation=truncation, **kwargs)
        return inputs

    def _forward(self, model_inputs, **generate_kwargs):
        in_b, input_length = model_inputs["input_ids"].shape

        self.check_inputs(
            input_length,
            generate_kwargs.get("min_length", self.generation_config.min_length),
            generate_kwargs.get("max_length", self.generation_config.max_length),
        )

        # User-defined `generation_config` passed to the pipeline call take precedence.
        # Fold the remaining generation arguments into a copy of it, since transformers 5
        # deprecates passing both.
        generation_config = copy.deepcopy(
            generate_kwargs.pop("generation_config", self.generation_config)
        )
        model_kwargs = generation_config.update(**generate_kwargs)

        output_ids = self.model.generate(
            **model_inputs, generation_config=generation_config, **model_kwargs
        )
        out_b = output_ids.shape[0]
        output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        for output_ids in model_outputs["output_ids"][0]:
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                }
            records.append(record)
        return records


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class SummarizationPipeline(Text2TextGenerationPipeline):
    """
    Summarize news articles and other documents.

    This summarizing pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"summarization"`.

    The models that this pipeline can use are models that have been fine-tuned on a summarization task, which is
    currently, '*bart-large-cnn*', '*google-t5/t5-small*', '*google-t5/t5-base*', '*google-t5/t5-large*', '*google-t5/t5-3b*', '*google-t5/t5-11b*'. See the up-to-date
    list of available models on [huggingface.co/models](https://huggingface.co/models?filter=summarization). For a list
    of available parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Unless the model you're using explicitly sets these generation parameters in its configuration files
    (`generation_config.json`), the following default values will be used:
    - max_new_tokens: 256
    - num_beams: 4

    Usage:

    ```python
    # use bart in pytorch
    summarizer = pipeline("summarization")
    summarizer("An apple a day, keeps the doctor away", min_length=5, max_length=20)
    ```"""

    # Used in the return key of the pipeline.
    return_name = "summary"

    def __call__(self, *args, **kwargs):
        r"""
        Summarize the text(s) given as inputs.

        Args:
            documents (*str* or `list[str]`):
                One or several articles (or one list of articles) to summarize.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./text_generation)).

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:

            - **summary_text** (`str`, present when `return_text=True`) -- The summary of the corresponding input.
            - **summary_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
              ids of the summary.
        """
        return super().__call__(*args, **kwargs)

    def check_inputs(self, input_length: int, min_length: int, max_length: int) -> bool:
        """
        Checks whether there might be something wrong with given input with regard to the model.
        """
        if max_length < min_length:
            logger.warning(f"Your min_length={min_length} must be inferior than your max_length={max_length}.")

        if input_length < max_length:
            logger.warning(
                f"Your max_length is set to {max_length}, but your input_length is only {input_length}. Since this is "
                "a summarization task, where outputs shorter than the input are typically wanted, you might "
                f"consider decreasing max_length manually, e.g. summarizer('...', max_length={input_length // 2})"
            )
