from transformers.models.bart import BartTokenizerFast

def get_processor(max_length: int = 8192):
    return BartTokenizerFast.from_pretrained(
        "facebook/bart-base",
        model_max_length=max_length,
        truncation=True,
    )
