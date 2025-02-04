import torch

from roberta.config import AttentionType, get_config
from roberta.evaluation import Evaluator
from roberta.model import prepare_args

NUM_SAMPLES = 2048
BATCH_SIZE = 16
SAVE_DIR = "roberta/results"


@torch.no_grad()
def main():
    evaluator = Evaluator(
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR,
    )

    # Softmax
    config = get_config()
    config.attention_type = AttentionType.softmax
    config.enable_flash_attention = False
    print(config.attention_type, prepare_args(config))
    print(evaluator.evaluate(config))
    # evaluator.evaluate_and_save(config)

    # Sparsemax
    config = get_config()
    config.attention_type = AttentionType.sparsemax
    config.attn_module_save_path = "roberta/sparsemax_params.pt"
    print(config.attention_type, prepare_args(config))
    print(evaluator.evaluate(config))
    # evaluator.evaluate_and_save(config)

    # Monarch
    config = get_config()
    config.attention_type = AttentionType.soba_monarch
    config.attn_module_save_path = "roberta/soba_params.pt"
    config.num_steps = 3
    config.block_size = 14
    print(config.attention_type, prepare_args(config))
    print(evaluator.evaluate(config))
    # evaluator.evaluate_and_save(config)

    return


if __name__ == "__main__":
    main()
