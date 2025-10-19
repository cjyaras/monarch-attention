import torch

from gps.config import AttentionType, get_config
from gps.evaluation import Evaluator

SAVE_DIR = "gps/results"


@torch.no_grad()
def main():

    evaluator = Evaluator(save_dir=SAVE_DIR)

    config = get_config()
    config.attention_type = AttentionType.softmax
    print(evaluator.evaluate(config))

    config = get_config()
    config.attention_type = AttentionType.monarch_attention
    config.num_steps = 1
    config.block_size = 90
    print(evaluator.evaluate(config))

    config = get_config()
    config.attention_type = AttentionType.nystromformer
    config.rank = 128
    print(evaluator.evaluate(config))


if __name__ == "__main__":
    main()
