import torch

from vit.config import AttentionType, get_config
from vit.evaluation import Evaluator

NUM_SAMPLES = 1024
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
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Sparsemax
    config = get_config()
    config.attention_type = AttentionType.sparsemax
    config.attn_module_save_path = "vit/sparsemax_params.pt"
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Monarch
    config = get_config()
    config.attention_type = AttentionType.soba_monarch
    config.attn_module_save_path = "vit/sparsemax_params.pt"
    config.num_steps = 3
    config.block_size = 14
    evaluator.evaluate_and_save(config)

    # Linformer
    config = get_config()
    config.attention_type = AttentionType.linformer
    config.rank = 64
    config.share_kv = False
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Performer
    config = get_config()
    config.attention_type = AttentionType.performer
    config.rank = 64
    config.estimator_type = "trig"
    config.ortho_features = False
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Nystromformer
    config = get_config()
    config.attention_type = AttentionType.nystromformer
    config.rank = 64
    config.conv_kernel_size = None
    print(config.attention_type)
    evaluator.evaluate_and_save(config)

    # Cosformer
    config = get_config()
    config.attention_type = AttentionType.cosformer
    print(config.attention_type)
    evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
