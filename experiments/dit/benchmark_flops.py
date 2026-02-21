from typing import List, Dict
import argparse
import torch
import os

import numpy as np

from experiments.dit.config import AttentionType
from experiments.dit.pipeline import get_pipeline, CustomDiTPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from torchvision.utils import save_image
from experiments.common.utils import get_device

import torchvision.models


def generate_attn_dict(attn_type: AttentionType, layers_to_replace: List, num_layers: int = 28) -> Dict:
    assert max(layers_to_replace) <= num_layers

    attn_dict = {}
    for i in range(1, num_layers + 1):
        attn_dict[i] = attn_type if i in layers_to_replace else AttentionType.softmax

    return attn_dict

def benchmark_flops(pipe: CustomDiTPipeline, num_classes: int, num_samples: int, num_inference_steps: int, cfg_scale: float):
    from torchtnt.utils.flops import FlopTensorDispatchMode
    
    # Classes to generate
    class_ids = np.random.choice(np.arange(num_classes), size=num_samples)

    # Generate same random sample for all models
    latent_channels = pipe.transformer.config.in_channels
    latent_size = pipe.transformer.config.sample_size

    latents = torch.randn(num_samples, latent_channels, latent_size, latent_size)

    with FlopTensorDispatchMode(pipe.transformer.transformer_blocks[0].attn1.processor.attn_module) as ftdm:
        _ = pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type="numpy", guidance_scale=cfg_scale)
        #print(ftdm.flop_counts)
        flops = ftdm.flop_counts["attn1"]["bmm.default"]
        return flops


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--attention_type', type=str, default='softmax', choices=['softmax', 'monarch', 'linformer', 'performer', 'nystromformer', 'cosformer'])
    parser.add_argument('--replace_all_layers', action='store_true')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=1)
    parser.add_argument('--cfg_scale', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--monarch_num_steps', type=int, default=3, help='Number of steps for monarch attention')
    parser.add_argument('--monarch_block_size', type=int, default=16, help='Block size for monarch attention')
    parser.add_argument('--rank', type=int, default=64, help='Rank for low-rank attentions')
    

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
            pipe = get_pipeline(attn_type=getattr(AttentionType, attention_type), rank=rank, block_size=block_size, num_steps=num_steps)
        else:
            layers_to_replace = list(range(14)) # Replace first half of layers
            attn_dict = generate_attn_dict(getattr(AttentionType, attention_type), layers_to_replace=layers_to_replace)
            pipe = get_pipeline(attn_dict, rank=rank, block_size=block_size, num_steps=num_steps)


    # Benchmark FLOPS
    flops = benchmark_flops(pipe, num_classes, num_samples, num_inference_steps, cfg_scale)

    print("Single-layer DiT", attention_type, "attention flops:", flops)



if __name__ == "__main__":
    main()
