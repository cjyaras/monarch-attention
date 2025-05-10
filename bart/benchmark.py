import torch

from bart.config import AttentionType, get_config
from bart.evaluation import Evaluator

NUM_SAMPLES = None 
BATCH_SIZE = 4
SAVE_DIR = "bart/results"

def print_results(res):
    max_key_characters = 0
    for k in res.keys():
        if len(k) > max_key_characters:
            max_key_characters = len(k)
    max_key_characters += 4
    for k in res.keys():
        print(f"{k:{max_key_characters}s}{res[k]}")

@torch.no_grad()
def main():
    for max_length, nystrom_rank, block_size, num_steps in [(1024, 64, 32, 3), (2048, 80, 32, 2), (4096, 112, 64, 2), (8192, 160, 64, 2)]:
        print(f"Max Length: {max_length}, nystrom_rank: {nystrom_rank}, block_size: {block_size}, num_steps: {num_steps}")
        evaluator = Evaluator(
            num_samples=NUM_SAMPLES,
            batch_size=BATCH_SIZE,
            save_dir=SAVE_DIR,
            max_length=max_length,
        )

        # efficient_attn_layers = [0, 1, 2, 3, 8, 9, 10, 11]
        # efficient_attn_layers = [1, 3, 5, 7, 9, 11]
        # efficient_attn_layers = [8, 9, 10, 11]
        # efficient_attn_layers = [6, 7, 8, 9, 10, 11]
        efficient_attn_layers = range(12)

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
        res = evaluator.evaluate(config)
        print_results(res)
        # evaluator.evaluate_and_save(config)



        # Nystromformer
        config = get_config()
        config.attention_type = get_mixed_type(
            AttentionType.nystromformer, AttentionType.softmax
        )
        config.rank = nystrom_rank
        config.conv_kernel_size = None
        print(config.attention_type)
        res = evaluator.evaluate(config)
        print_results(res)
        #evaluator.evaluate_and_save(config)


        # Monarch
        config = get_config()
        config.attention_type = get_mixed_type(
            AttentionType.monarch_attention, AttentionType.softmax
        )
        config.num_steps = num_steps
        config.block_size = block_size
        print(config.attention_type)
        res = evaluator.evaluate(config)
        print_results(res)
        # evaluator.evaluate_and_save(config)



if __name__ == "__main__":
    main()
