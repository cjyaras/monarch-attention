import torch

from vit.config import AttentionType, PadType, get_config
from vit.evaluation import Evaluator

NUM_SAMPLES = 1024
TOP_K = 5
BATCH_SIZE = 8
SAVE_DIR = "vit/results"


@torch.no_grad()
def main():
    evaluator = Evaluator(
        num_samples=NUM_SAMPLES,
        top_k=TOP_K,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR,
    )

    # Softmax
    config = get_config()
    config.attention_type = AttentionType.softmax
    config.enable_flash_attention = False
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Monarch
    for num_steps in [1, 2, 3]:
        for block_size in [14]:
            config = get_config()
            config.attention_type = AttentionType.monarch_attention
            config.pad_type = PadType.pre
            config.block_size = block_size
            config.num_steps = num_steps
            print(config.attention_type, num_steps, block_size)
            evaluator.evaluate_and_save(config)

    # Linformer
    # for rank in range(16, 96 + 1, 16):
    #     config = get_config()
    #     config.attention_type = AttentionType.linformer
    #     config.rank = rank
    #     print(config.attention_type, rank)
    #     evaluator.evaluate_and_save(config)

    # Performer
    for rank in range(16, 96 + 1, 16):
        config = get_config()
        config.attention_type = AttentionType.performer
        config.rank = rank
        print(config.attention_type, rank)
        evaluator.evaluate_and_save(config)

    # Nystromformer
    for rank in range(16, 40 + 1, 8):
        config = get_config()
        config.attention_type = AttentionType.nystromformer
        config.rank = rank
        print(config.attention_type, rank)
        evaluator.evaluate_and_save(config)

    # Cosformer
    config = get_config()
    config.attention_type = AttentionType.cosformer
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # LinearAttention
    config = get_config()
    config.attention_type = AttentionType.linear_attention
    print(config.attention_type)
    evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
