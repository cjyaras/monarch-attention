from bert.sparse_roberta import get_custom_model
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Load the model
model = get_custom_model(
    "mtreviso/sparsemax-roberta",
    initial_alpha=2.0,
    use_triton_entmax=False,
    from_scratch=False,
)
