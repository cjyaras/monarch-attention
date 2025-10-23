import torch

from tokengt.config import AttentionType, get_config
from tokengt.data import get_input_output_dims, get_processed_data
from tokengt.evaluation import Evaluator

SAVE_DIR = "tokengt/results"


@torch.no_grad()
def main():

    evaluator = Evaluator(save_dir=SAVE_DIR)
    input_dims, output_dims = get_input_output_dims(get_processed_data())

    config = get_config(input_dims, output_dims)
    config.attention_type = AttentionType.softmax
    print(evaluator.evaluate(config))

    config = get_config(input_dims, output_dims)
    config.attention_type = AttentionType.monarch_attention
    config.num_steps = 1
    for block_size in [16, 32, 64]:
        config.block_size = block_size
        print(evaluator.evaluate(config))

    config = get_config(input_dims, output_dims)
    config.attention_type = AttentionType.nystromformer
    for rank in [64, 128]:
        config.rank = rank
        print(evaluator.evaluate(config))


if __name__ == "__main__":
    main()
