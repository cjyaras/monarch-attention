from typing import Dict

import evaluate
import numpy as np
import torch

from experiments.common.logging import Logger
from experiments.common.utils import attention_bmm_flops
from experiments.roberta.config import CustomRobertaConfig
from experiments.roberta.data import get_dataset
from experiments.roberta.model import CustomRobertaForQuestionAnswering, get_model
from experiments.roberta.processor import get_processor

# Same defaults as the Hugging Face question-answering pipeline
MAX_SEQ_LEN = 384
DOC_STRIDE = 128
MAX_ANSWER_LEN = 15


def best_span(start: np.ndarray, end: np.ndarray, allowed: np.ndarray):
    """Most likely (start, end, score) answer span within one window, or None."""
    start = np.where(allowed == 0, -10000.0, start)
    end = np.where(allowed == 0, -10000.0, end)
    start = np.exp(start - start.max())
    start = start / start.sum()
    end = np.exp(end - end.max())
    end = end / end.sum()
    start[0] = end[0] = 0.0  # never answer with the CLS token

    # Score every span with start <= end < start + MAX_ANSWER_LEN
    candidates = np.tril(np.triu(np.outer(start, end)), MAX_ANSWER_LEN - 1)
    s, e = np.unravel_index(np.argmax(candidates), candidates.shape)
    if not (allowed[s] and allowed[e]):
        return None
    return s, e, candidates[s, e]


class Evaluator:

    def __init__(
        self, num_samples: int, batch_size: int, save_dir: str, split: str = "validation"
    ):
        self.batch_size = batch_size
        self.dataset = get_dataset(num_samples=num_samples, split=split)
        self.tokenizer = get_processor()
        self.metric = evaluate.load("squad")
        self.logger = Logger(save_dir)

    def windows(self, examples):
        """Split each question/context pair into overlapping windows of the context."""
        windows = []
        for i, (question, context) in enumerate(zip(examples["question"], examples["context"])):
            encoding = self.tokenizer(
                question,
                context,
                padding="max_length",
                truncation="only_second",
                max_length=MAX_SEQ_LEN,
                stride=DOC_STRIDE,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
            )
            for span in range(len(encoding["input_ids"])):
                input_ids = np.array(encoding["input_ids"][span])
                attention_mask = np.array(encoding["attention_mask"][span])
                # Answers must come from the context (or CLS, which best_span rules out)
                allowed = np.array([sid == 1 for sid in encoding.sequence_ids(span)])
                allowed = (allowed | (input_ids == self.tokenizer.cls_token_id)) & attention_mask
                windows.append((i, input_ids, attention_mask, allowed, encoding[span]))
        return windows

    @torch.no_grad()
    def predict(self, model: CustomRobertaForQuestionAnswering, examples) -> list[str]:
        windows = self.windows(examples)
        answers = [{} for _ in examples["question"]]  # answer text -> summed score

        for start in range(0, len(windows), self.batch_size):
            batch = windows[start : start + self.batch_size]
            outputs = model(
                input_ids=torch.tensor(np.stack([w[1] for w in batch])).to(model.device),
                attention_mask=torch.tensor(np.stack([w[2] for w in batch])).to(model.device),
            )
            start_logits = outputs.start_logits.float().cpu().numpy()
            end_logits = outputs.end_logits.float().cpu().numpy()

            for (i, _, _, allowed, encoding), s_logits, e_logits in zip(batch, start_logits, end_logits):
                span = best_span(s_logits, e_logits, allowed)
                if span is None:
                    continue
                s, e, score = span
                # Expand the answer to whole words of the context
                try:
                    char_start = encoding.word_to_chars(encoding.token_to_word(s), sequence_index=1)[0]
                    char_end = encoding.word_to_chars(encoding.token_to_word(e), sequence_index=1)[1]
                except Exception:
                    char_start, char_end = encoding.offsets[s][0], encoding.offsets[e][1]
                text = examples["context"][i][char_start:char_end]
                # The same answer found in several windows accumulates score
                key = next((a for a in answers[i] if a.lower() == text.lower()), text)
                answers[i][key] = answers[i].get(key, 0.0) + score.item()

        return [max(a, key=a.get) if a else "" for a in answers]

    def evaluate(self, config: CustomRobertaConfig) -> Dict[str, float]:
        model = get_model(config)
        flops = attention_bmm_flops(
            model,
            [f"roberta.encoder.layer.{i}.attention" for i in range(config.num_hidden_layers)],
            lambda: self.predict(model, self.dataset[: self.batch_size]),
        )

        predictions = self.predict(model, self.dataset[:])
        result = self.metric.compute(
            predictions=[
                {"id": id, "prediction_text": text}
                for id, text in zip(self.dataset["id"], predictions)
            ],
            references=[
                {"id": id, "answers": answers}
                for id, answers in zip(self.dataset["id"], self.dataset["answers"])
            ],
        )
        assert result is not None
        result["total_attention_bmm_flops"] = flops // self.batch_size
        return result

    def evaluate_and_save(self, config: CustomRobertaConfig) -> str:
        result = self.evaluate(config)
        file_name = self.logger.save(config, result)
        return file_name
