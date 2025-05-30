import torch

from roberta.config import AttentionType, PadType, get_config
from roberta.evaluation import Evaluator

NUM_SAMPLES = 1024
BATCH_SIZE = 8
SAVE_DIR = "roberta/results"


@torch.no_grad()
def main():
    evaluator = Evaluator(
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR,
    )

    efficient_attn_layers = [0, 1, 2, 3, 8, 9, 10, 11]

    def get_mixed_type(efficient_type, default_type):
        return {
            layer_num: (
                efficient_type if layer_num in efficient_attn_layers else default_type
            )
            for layer_num in range(12)
        }

    # Softmax
    config = get_config()
    config.attention_type = AttentionType.softmax
    config.enable_flash_attention = False
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Monarch
    for num_steps in [1]:
        for block_size in [24, 48, 96, 128]:
            config = get_config()
            config.attention_type = get_mixed_type(
                AttentionType.monarch_attention, AttentionType.softmax
            )
            config.block_size = block_size
            config.pad_type = PadType.post
            config.num_steps = num_steps
            print(config.attention_type[0], num_steps, block_size)
            evaluator.evaluate_and_save(config)

    # Linformer
    # for rank in range(32, 192 + 1, 32):
    #     config = get_config()
    #     config.attention_type = get_mixed_type(
    #         AttentionType.linformer, AttentionType.softmax
    #     )
    #     config.rank = rank
    #     print(config.attention_type[0], rank)
    #     evaluator.evaluate_and_save(config)

    # Performer
    for rank in range(32, 192 + 1, 32):
        config = get_config()
        config.attention_type = get_mixed_type(
            AttentionType.performer, AttentionType.softmax
        )
        config.rank = rank
        print(config.attention_type[0], rank)
        evaluator.evaluate_and_save(config)

    # Nystromformer
    for rank in range(16, 64 + 1, 16):
        config = get_config()
        config.attention_type = get_mixed_type(
            AttentionType.nystromformer, AttentionType.softmax
        )
        config.rank = rank
        print(config.attention_type[0], rank)
        evaluator.evaluate_and_save(config)

    # Cosformer
    config = get_config()
    config.attention_type = get_mixed_type(
        AttentionType.cosformer, AttentionType.softmax
    )
    print(config.attention_type[0])
    evaluator.evaluate_and_save(config)

    # LinearAttention
    config = get_config()
    config.attention_type = get_mixed_type(
        AttentionType.linear_attention, AttentionType.softmax
    )
    print(config.attention_type[0])
    evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
