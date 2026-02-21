from typing import List, Dict
import argparse
import torch
import os

import numpy as np

from experiments.dit.config import AttentionType
from experiments.dit.pipeline import get_pipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from torchvision.utils import save_image
from common.utils import get_device

device = get_device()


def save_output_images(output: ImagePipelineOutput, save_path: str, num_images_per_row: int):
    images = torch.Tensor(output.images)
    images_reshaped = images.permute(0, 3, 1, 2)
    save_image(images_reshaped, save_path, nrow=num_images_per_row)

def generate_attn_dict(attn_type: AttentionType, layers_to_replace: List, num_layers: int = 28) -> Dict:
    assert max(layers_to_replace) <= num_layers

    attn_dict = {}
    for i in range(num_layers):
        attn_dict[i + 1] = attn_type if i in layers_to_replace else AttentionType.softmax

    return attn_dict

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--attention_type', type=str, default='softmax', choices=['softmax', 'monarch', 'linformer', 'performer', 'nystromformer', 'cosformer'])
    parser.add_argument('--replace_all_layers', action='store_true')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=36)
    parser.add_argument('--num_inference_steps', type=int, default=32)
    parser.add_argument('--cfg_scale', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--save_dir', type=str, default='./dit/generations')

    parser.add_argument('--monarch_num_steps', type=int, default=3, help='Number of steps for monarch attention')
    parser.add_argument('--monarch_block_size', type=int, default=16, help='Block size for monarch attention')
    parser.add_argument('--rank', type=int, default=32, help='Rank for low-rank attentions')

    args = parser.parse_args()

    return args

# Performer model is only generating all black images
@torch.no_grad()
def main():
    # Parse args
    args = parse_args()
    attention_type = args.attention_type
    replace_all_layers = args.replace_all_layers
    num_classes = args.num_classes
    num_samples = args.num_samples
    num_inference_steps = args.num_inference_steps
    cfg_scale = args.cfg_scale
    seed = args.seed
    num_steps = args.monarch_num_steps
    block_size = args.monarch_block_size
    rank = args.rank

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Prepare sampling
    if attention_type == "softmax":
        pipe = get_pipeline(attn_type=AttentionType.softmax)
    else:
        if replace_all_layers:
            pipe = get_pipeline(attn_type=getattr(AttentionType, attention_type), num_steps=num_steps, block_size=block_size, rank=rank)
        else:
            layers_to_replace = list(range(14, 28)) # Replace first half of layers
            attn_dict = generate_attn_dict(getattr(AttentionType, attention_type), layers_to_replace=layers_to_replace)
            print(attn_dict)
            pipe = get_pipeline(attn_dict, num_steps=num_steps, block_size=block_size, rank=rank)

    

    # Classes to generate
    class_ids = np.random.choice(np.arange(num_classes), size=num_samples)

    # Generate same random sample for all models
    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size

    latents = torch.randn(num_samples, latent_channels, latent_size, latent_size)


    # Generate images
    output_type = "numpy"
    output = pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type, guidance_scale=cfg_scale)

    # Save images
    parent_save_dir = args.save_dir
    if not os.path.exists(parent_save_dir):
        os.makedirs(parent_save_dir)
    
    if attention_type == "softmax":
        save_fname = attention_type + ".png"
    elif attention_type != "softmax" and replace_all_layers:
        save_fname = attention_type + "_all_layers.png"
    elif attention_type != "softmax" and not replace_all_layers:
        save_fname = attention_type + "_second_half_layers.png"

    save_path = os.path.join(parent_save_dir, save_fname)
    num_images_per_row = int(np.sqrt(num_samples))
    save_output_images(output, save_path, num_images_per_row=num_images_per_row)


if __name__ == "__main__":
    main()
