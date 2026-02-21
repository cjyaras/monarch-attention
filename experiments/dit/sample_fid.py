from typing import List, Dict
import argparse
import torch
import os
import numpy as np

from math import ceil 
from tqdm import tqdm


from experiments.dit.config import AttentionType
from experiments.dit.pipeline import get_pipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from common.utils import get_device

device = get_device()



def save_images_as_npz(output: ImagePipelineOutput, save_fname: str):
    images = output.images # (B x H x W x C)
    np.savez(save_fname, arr0=images)
    print(f"Saved .npz file to {save_fname} [shape = {images.shape}]")


def generate_attn_dict(attn_type: AttentionType, layers_to_replace: List, num_layers: int = 28) -> Dict:
    assert max(layers_to_replace) <= num_layers

    attn_dict = {}
    for i in range(num_layers):
        attn_dict[i + 1] = attn_type if i in layers_to_replace else AttentionType.softmax

    return attn_dict


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--attention_type', type=str, default='softmax', choices=['softmax', 'monarch', 'linformer', 'performer', 'nystromformer', 'cosformer', 'linear_attention'])
    parser.add_argument('--replace_all_layers', action='store_true')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=50_000)
    parser.add_argument('--num_inference_steps', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cfg_scale', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--save_dir', type=str, default='/scratch/qingqu_root/qingqu1/alecx/dit_generations')

    parser.add_argument('--monarch_num_steps', type=int, default=3, help='Number of steps for monarch attention')
    parser.add_argument('--monarch_block_size', type=int, default=16, help='Block size for monarch attention')
    parser.add_argument('--rank', type=int, default=32, help='Rank for low-rank attentions')

    args = parser.parse_args()

    return args

# Performer model is only generating all black images
@torch.no_grad()
def main():
    # Get args
    args = parse_args()
    attention_type = args.attention_type
    replace_all_layers = args.replace_all_layers
    num_classes = args.num_classes
    num_samples = args.num_samples
    num_inference_steps = args.num_inference_steps
    batch_size = args.batch_size
    cfg_scale = args.cfg_scale
    seed = args.seed
    num_steps = args.monarch_num_steps
    block_size = args.monarch_block_size
    rank = args.rank

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare sampling process
    parent_save_dir = args.save_dir
    if attention_type == "softmax":
        experiment_dir = str(num_inference_steps) + "_sampling_steps_seed_" + str(seed)
        save_path = os.path.join(parent_save_dir, attention_type)
        pipe = get_pipeline(attn_type=AttentionType.softmax, 
                            rank=rank, 
                            block_size=block_size,
                            num_steps=num_steps)
    else:
        if replace_all_layers:
            experiment_dir = "all_layers_" + str(num_inference_steps) + "_sampling_steps_seed_" + str(seed)
            save_path = os.path.join(parent_save_dir, attention_type, experiment_dir)
            pipe = get_pipeline(attn_type=getattr(AttentionType, attention_type),
                                rank=rank,
                                block_size=block_size,
                                num_steps=num_steps) 
        else:
            layers_to_replace = list(range(14, 28))
            experiment_dir = "layers_" + str(layers_to_replace).replace('[', '').replace(']', '').replace(', ', '_') + "_" + str(num_inference_steps) + "_sampling_steps_seed_" + str(seed)
            save_path = os.path.join(parent_save_dir, attention_type, experiment_dir)

            attn_dict = generate_attn_dict(getattr(AttentionType, attention_type), layers_to_replace=layers_to_replace) #, *attn_params)
            print(attn_dict)
            pipe = get_pipeline(attn_type=attn_dict,
                                rank=rank,
                                block_size=block_size,
                                num_steps=num_steps)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size

    # Sample images
    total_samples = int(ceil(num_samples / batch_size)) * batch_size
    num_iters = int(total_samples // batch_size)
    output_type = "numpy"
    pbar = tqdm(range(num_iters))
    for (i, _) in enumerate(pbar):         
        class_ids = list( np.random.choice( np.arange(num_classes), size=batch_size ) ) 
        latents = torch.randn(batch_size, latent_channels, latent_size, latent_size) # Use same latents for all models

        # Pipeline output
        output = pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type, guidance_scale=cfg_scale)
        save_fname = os.path.join(save_path, "batch_" + str(i))
        save_images_as_npz(output, save_fname)
        del output


if __name__ == "__main__":
    main()
