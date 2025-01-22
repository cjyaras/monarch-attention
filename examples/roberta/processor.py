from transformers import RobertaTokenizerFast


def get_processor():
    return RobertaTokenizerFast.from_pretrained("deepset/roberta-base-squad2")
