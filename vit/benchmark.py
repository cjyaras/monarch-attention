import torch

from vit.config import AttentionType, PadType, get_config
from vit.evaluation import Evaluator

NUM_SAMPLES = 256
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
    for num_steps in [1, 2]:
        for block_size in [14, 2]:
            config = get_config()
            config.attention_type = AttentionType.monarch
            config.pad_type = PadType.pre
            config.block_size = block_size
            config.num_steps = num_steps
            print(config.attention_type, num_steps, block_size)
            evaluator.evaluate_and_save(config)

    # Linformer
    for rank in [32, 64, 128]:
        config = get_config()
        config.attention_type = AttentionType.linformer
        config.rank = rank
        config.share_kv = False
        print(config.attention_type, rank)
        evaluator.evaluate_and_save(config)

    # Performer
    for rank in [16, 32, 48, 64]:
        config = get_config()
        config.attention_type = AttentionType.performer
        config.rank = rank
        config.estimator_type = "trig"
        config.ortho_features = False
        print(config.attention_type, rank)
        evaluator.evaluate_and_save(config)

    # Nystromformer
    for rank in [16, 32, 48]:
        config = get_config()
        config.attention_type = AttentionType.nystromformer
        config.rank = rank
        config.conv_kernel_size = None
        print(config.attention_type, rank)
        evaluator.evaluate_and_save(config)

    # Cosformer
    config = get_config()
    config.attention_type = AttentionType.cosformer
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # LinearAttention
    config = get_config()
    config.attention_type = AttentionType.linear
    print(config.attention_type)
    evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
