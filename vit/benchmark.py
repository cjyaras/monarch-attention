import torch

from vit.config import AttentionType, PadType, get_config
from vit.evaluation import Evaluator

NUM_SAMPLES = 512
TOP_K = 5
BATCH_SIZE = 16
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
    print(config)
    evaluator.evaluate_and_save(config)

    # Monarch
    for num_steps in [1, 2]:
        config = get_config()
        config.attention_type = AttentionType.monarch
        config.pad_type = PadType.pre
        config.block_size = 14
        config.num_steps = num_steps
        print(config)
        evaluator.evaluate_and_save(config)

    # # Linformer
    # config = get_config()
    # config.attention_type = AttentionType.linformer
    # config.rank = 64
    # config.share_kv = False
    # print(config.attention_type)
    # print(evaluator.evaluate(config))
    # # evaluator.evaluate_and_save(config)

    # # Performer
    # config = get_config()
    # config.attention_type = AttentionType.performer
    # config.rank = 64
    # config.estimator_type = "trig"
    # config.ortho_features = False
    # print(config.attention_type)
    # print(evaluator.evaluate(config))
    # # evaluator.evaluate_and_save(config)

    # Nystromformer
    for rank in [16, 32, 48, 64]:
        config = get_config()
        config.attention_type = AttentionType.nystromformer
        config.rank = rank
        config.conv_kernel_size = None
        print(config)
        evaluator.evaluate_and_save(config)

    # # TaylorLinearAttention
    # config = get_config()
    # config.attention_type = AttentionType.taylor
    # config.rank = 64  # This will be used as proj_dim
    # print(config.attention_type)
    # print(evaluator.evaluate(config))
    # # evaluator.evaluate_and_save(config)

    # # Cosformer
    # config = get_config()
    # config.attention_type = AttentionType.cosformer
    # print(config.attention_type)
    # print(evaluator.evaluate(config))
    # # evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
