import torch
from config import AttentionType, get_config
from eval import Evaluator

NUM_SAMPLES = 128
TOP_K = 5
BATCH_SIZE = 4
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
    evaluator.evaluate_and_save(config)

    # Sparsemax
    config = get_config()
    config.attention_type = AttentionType.sparsemax
    config.scale_attention_temperature = True
    evaluator.evaluate_and_save(config)

    # Monarch
    config = get_config()
    config.attention_type = AttentionType.monarch
    config.scale_attention_temperature = True
    config.efficient_attention_num_steps = 3
    config.efficient_attention_step_size = 2.5
    config.efficient_attention_block_size = 14
    evaluator.evaluate_and_save(config)

    # Linformer
    config = get_config()
    config.attention_type = AttentionType.linformer
    config.efficient_attention_rank = 64
    config.share_kv = False
    evaluator.evaluate_and_save(config)

    # Performer
    config = get_config()
    config.attention_type = AttentionType.performer
    config.efficient_attention_rank = 64
    config.estimator_type = "trig"
    config.ortho_features = False
    evaluator.evaluate_and_save(config)

    # Nystromformer
    config = get_config()
    config.attention_type = AttentionType.nystromformer
    config.efficient_attention_rank = 64
    config.conv_kernel_size = None
    evaluator.evaluate_and_save(config)

    # Cosformer
    config = get_config()
    config.attention_type = AttentionType.cosformer
    evaluator.evaluate_and_save(config)


if __name__ == "__main__":
    main()
