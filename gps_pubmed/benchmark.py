import torch

from gps_pubmed.config import AttentionType, get_config
from gps_pubmed.evaluation import Evaluator

SAVE_DIR = "gps_pubmed/results"


@torch.no_grad()
def main():

    evaluator = Evaluator(save_dir=SAVE_DIR)

    config = get_config()
    config.attention_type = AttentionType.softmax
    evaluator.evaluate_and_save(config)

    config = get_config()
    config.attention_type = AttentionType.monarch_attention
    config.num_steps = 1
    for block_size in [128, 256, 512]:
        config.block_size = block_size
        evaluator.evaluate_and_save(config)

    config = get_config()
    config.attention_type = AttentionType.nystromformer
    for rank in [256, 384, 448]:
        config.rank = rank
        evaluator.evaluate_and_save(config)

    # config = get_config()
    # config.attention_type = AttentionType.monarch_attention
    # config.block_size = 128
    # for num_steps in [4, 12, 22, 30]:
    #     config.num_steps = num_steps
    #     evaluator.evaluate_and_save(config)

    # config = get_config()
    # config.attention_type = AttentionType.nystromformer
    # for rank in [128, 256, 384, 448, 480]:
    #     config.rank = rank
    #     evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
