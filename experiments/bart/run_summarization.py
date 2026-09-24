"""Fine-tune the custom BART model on BookSum with Seq2SeqTrainer.

Minimal version of Hugging Face's `examples/pytorch/summarization/run_summarization.py`,
reduced to what this repo uses. See experiments/bart/README.md for the training commands.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import nltk
import numpy as np
import transformers
from datasets import load_dataset
from filelock import FileLock
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from experiments.bart.config import CustomBartConfig
from experiments.bart.model import CustomBartForConditionalGeneration

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    model_name_or_path: str = "facebook/bart-base"
    dataset_name: str = "kmfoda/booksum"
    text_column: str = "chapter"
    summary_column: str = "summary_text"
    trust_remote_code: bool = False
    max_source_length: int = field(
        default=1024, metadata={"help": "Inputs are truncated to this many tokens."}
    )
    max_target_length: int = field(
        default=128, metadata={"help": "Summaries are truncated to this many tokens."}
    )
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    num_beams: int = field(
        default=1, metadata={"help": "Beams for generation during evaluation."}
    )
    ignore_pad_token_for_loss: bool = True
    attention_type: str = field(
        default="softmax",
        metadata={"help": "See experiments/common/attention.py:AttentionType."},
    )
    num_steps: Optional[int] = field(
        default=None, metadata={"help": "Monarch attention steps."}
    )
    block_size: Optional[int] = field(
        default=None, metadata={"help": "Monarch attention block size."}
    )
    attention_rank: Optional[int] = field(
        default=None, metadata={"help": "Rank for low-rank baselines."}
    )


def main():
    args, training_args = HfArgumentParser(
        (ScriptArguments, Seq2SeqTrainingArguments)
    ).parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError, OSError:
        with FileLock(".lock"):
            nltk.download("punkt_tab", quiet=True)

    set_seed(training_args.seed)
    raw_datasets = load_dataset(args.dataset_name)

    config = CustomBartConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        attention_type=args.attention_type,
        num_steps=args.num_steps,
        block_size=args.block_size,
        rank=args.attention_rank,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    model = CustomBartForConditionalGeneration.from_pretrained(
        args.model_name_or_path, config=config, trust_remote_code=args.trust_remote_code
    )
    if model.config.max_position_embeddings < args.max_source_length:
        # Linearly interpolate the pretrained encoder position embeddings to the longer input length.
        logger.warning(
            f"Increasing the model's number of position embedding vectors from"
            f" {model.config.max_position_embeddings} to {args.max_source_length}."
        )
        model.resize_position_embeddings(args.max_source_length)

    def preprocess(examples):
        pairs = [
            (t, s)
            for t, s in zip(examples[args.text_column], examples[args.summary_column])
            if t and s
        ]
        inputs, targets = [t for t, _ in pairs], [s for _, s in pairs]
        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, truncation=True
        )
        labels = tokenizer(
            text_target=targets, max_length=args.max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare(split, max_samples):
        ds = raw_datasets[split]
        if max_samples is not None:
            ds = ds.select(range(min(len(ds), max_samples)))
        with training_args.main_process_first(
            desc=f"{split} dataset map pre-processing"
        ):
            return ds.map(
                preprocess,
                batched=True,
                remove_columns=ds.column_names,
                desc=f"Tokenizing {split}",
            )

    train_dataset = (
        prepare("train", args.max_train_samples) if training_args.do_train else None
    )
    eval_dataset = (
        prepare("validation", args.max_eval_samples) if training_args.do_eval else None
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100
        if args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # rougeLsum expects a newline after each sentence.
        decoded_preds, decoded_labels = (
            [
                "\n".join(nltk.sent_tokenize(text.strip()))
                for text in tokenizer.batch_decode(ids, skip_special_tokens=True)
            ]
            for ids in (preds, labels)
        )
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        result["gen_len"] = np.mean(
            [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        )
        return result

    if training_args.generation_max_length is None:
        training_args.generation_max_length = args.max_target_length
    training_args.generation_num_beams = args.num_beams

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.predict_with_generate
        else None,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint
        if checkpoint is None and os.path.isdir(training_args.output_dir):
            checkpoint = get_last_checkpoint(training_args.output_dir)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
