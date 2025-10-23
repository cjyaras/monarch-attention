import torch

from gps.config import AttentionType, get_config
from gps.evaluation import Evaluator

SAVE_DIR = "gps/results"


@torch.no_grad()
def main():
    evaluator = Evaluator(save_dir=SAVE_DIR)

    config = get_config()
    config.attention_type = AttentionType.softmax
    evaluator.evaluate_and_save(config)

    config = get_config()
    config.attention_type = AttentionType.monarch_attention
    config.block_size = 128
    for num_steps in [1, 2, 3, 4]:
        config.num_steps = num_steps
        evaluator.evaluate_and_save(config)

    config = get_config()
    config.attention_type = AttentionType.nystromformer
    for rank in [128, 192, 256, 320]:
        config.rank = rank
        evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
