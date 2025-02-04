from transformers import RobertaTokenizerFast


def get_processor():
    return RobertaTokenizerFast.from_pretrained(
        "csarron/roberta-base-squad-v1"
    )  # "deepset/roberta-base-squad2")
