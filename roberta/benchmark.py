import torch

from roberta.config import AttentionType, get_config
from roberta.evaluation import Evaluator

NUM_SAMPLES = 256
BATCH_SIZE = 4
SAVE_DIR = "roberta/results"


@torch.no_grad()
def main():
    evaluator = Evaluator(
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR,
    )

    efficient_attn_layers = [0, 1, 2, 3, 8, 9, 10, 11]
    # efficient_attn_layers = [1, 3, 5, 7, 9, 11]
    # efficient_attn_layers = [8, 9, 10, 11]
    # efficient_attn_layers = [6, 7, 8, 9, 10, 11]
    # efficient_attn_layers = range(12)

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
    print(evaluator.evaluate(config))
    # evaluator.evaluate_and_save(config)

    # Monarch
    config = get_config()
    config.attention_type = get_mixed_type(AttentionType.monarch, AttentionType.softmax)
    config.num_steps = 2
    config.block_size = 32
    print(config.attention_type)
    print(evaluator.evaluate(config))
    # evaluator.evaluate_and_save(config)

    return

    # Linformer
    config = get_config()
    config.attention_type = get_mixed_type(
        AttentionType.linformer, AttentionType.softmax
    )
    config.rank = 64
    config.share_kv = False
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Performer
    config = get_config()
    config.attention_type = get_mixed_type(
        AttentionType.performer, AttentionType.softmax
    )
    config.rank = 64
    config.estimator_type = "trig"
    config.ortho_features = False
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Nystromformer
    config = get_config()
    config.attention_type = get_mixed_type(
        AttentionType.nystromformer, AttentionType.softmax
    )
    config.rank = 64
    config.conv_kernel_size = None
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Cosformer
    config = get_config()
    config.attention_type = get_mixed_type(
        AttentionType.cosformer, AttentionType.softmax
    )
    print(config.attention_type)
    evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
