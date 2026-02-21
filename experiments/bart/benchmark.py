import argparse
import json
import os

from experiments.bart.config import AttentionType, get_config
from experiments.bart.evaluation import Evaluator
import torch


parser = argparse.ArgumentParser(description='A benchmark for various attention types.')

parser.add_argument(
    "--model_checkpoint_path", 
    default="./bart/finetuned/output/",
    help="The path to the Bart checkpoint.",
)
parser.add_argument(
    "--save_dir",
    default="./bart/results_softmax",
    help="The path to the output json files."
)

NUM_SAMPLES = None
BATCH_SIZE = 4

args = parser.parse_args()
print(args)

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
    file_names = {}

    for max_length, nystrom_rank, block_size, num_steps in [(1024, 64, 32, 3), (2048, 80, 32, 2), (4096, 112, 64, 2), (8192, 160, 64, 2)]:
        print(f"Max Length: {max_length}, nystrom_rank: {nystrom_rank}, block_size: {block_size}, num_steps: {num_steps}")
        evaluator = Evaluator(
            num_samples=NUM_SAMPLES,
            batch_size=BATCH_SIZE,
            save_dir=args.save_dir,
            max_length=max_length,
            model_checkpoint_path=args.model_checkpoint_path,
        )

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
        # res = evaluator.evaluate(config)
        file_name, res = evaluator.evaluate_and_save(config)
        print_results(res)
        file_names[f"softmax_{max_length}"] = file_name



        # Nystromformer
        config = get_config()
        config.attention_type = get_mixed_type(
            AttentionType.nystromformer, AttentionType.softmax
        )
        config.rank = nystrom_rank
        config.conv_kernel_size = None
        print(config.attention_type)
        # res = evaluator.evaluate(config)
        file_name, res = evaluator.evaluate_and_save(config)
        print_results(res)
        file_names[f"nystrom_{max_length}_rank{nystrom_rank}"] = file_name


        # Monarch
        config = get_config()
        config.attention_type = get_mixed_type(
            AttentionType.monarch_attention, AttentionType.softmax
        )
        config.num_steps = num_steps
        config.block_size = block_size
        print(config.attention_type)
        # res = evaluator.evaluate(config)
        file_name, res = evaluator.evaluate_and_save(config)
        print_results(res)
        file_names[f"monarch_{max_length}_b{block_size}_t{num_steps}"] = file_name

    with open(os.path.join(args.save_dir, "filenames.json"), "w") as f:
        json.dump(file_names, f)


if __name__ == "__main__":
    main()
