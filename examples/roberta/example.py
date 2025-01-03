from roberta.models import CustomRobertaConfig, CustomRobertaForSequenceClassification

config = CustomRobertaConfig.from_pretrained("mtreviso/sparsemax-roberta")
model = CustomRobertaForSequenceClassification.from_pretrained(
    "mtreviso/sparsemax-roberta", config=config, revision="main/checkpoint-80000-cola"
)
print(model)
