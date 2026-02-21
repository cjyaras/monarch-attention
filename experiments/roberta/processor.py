from transformers.models.roberta import RobertaTokenizerFast


def get_processor():
    return RobertaTokenizerFast.from_pretrained("csarron/roberta-base-squad-v1")
    # return RobertaTokenizerFast.from_pretrained("deepset/roberta-base-squad2")
