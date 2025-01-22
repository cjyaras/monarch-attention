import torch
from roberta.config import AttentionType, get_config
from roberta.evaluation import Evaluator
from roberta.model import prepare_args

NUM_SAMPLES = 128
BATCH_SIZE = 4
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
    print(config.attention_type, prepare_args(config, 0))
    print(evaluator.evaluate(config))
    # evaluator.evaluate_and_save(config)

    return

    # Sparsemax
    config = get_config()
    config.attention_type = AttentionType.sparsemax
    config.scale_attention_temperature = True
    print(config.attention_type, prepare_args(config, 0))
    print(evaluator.evaluate(config))
    # evaluator.evaluate_and_save(config)

    return

    # Monarch
    config = get_config()
    config.attention_type = AttentionType.soba_monarch
    config.scale_attention_temperature = True
    config.num_steps = 3
    config.step_size = 2.5
    config.block_size = 14
    print(config.attention_type, prepare_args(config, 0))
    evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
