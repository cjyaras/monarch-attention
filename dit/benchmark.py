from typing import List, Dict
import torch
import os

from dit.config import AttentionType
from dit.pipeline import get_pipeline

from PIL import Image
import time

def save_image(image: Image, save_path: str):
    image.save(save_path)

def generate_attn_dict(attn_type: AttentionType, layers_to_replace: List, num_layers: int = 28) -> Dict:
    assert max(layers_to_replace) <= num_layers

    attn_dict = {}
    for i in range(1, num_layers + 1):
        attn_dict[i] = attn_type if i in layers_to_replace else AttentionType.softmax

    return attn_dict

@torch.no_grad()
def main():
    # Classes to generate
    classes = ["German shepherd"]

    # Make pipelines
    sm_pipe = get_pipeline(attn_type=AttentionType.softmax)
    soba_pipe = get_pipeline(attn_type=AttentionType.soba_monarch)
    lin_pipe = get_pipeline(attn_type=AttentionType.linformer, rank=64)
    #perf_pipe = get_pipeline(attn_type=AttentionType.performer)
    nys_pipe = get_pipeline(attn_type=AttentionType.nystromformer, rank=64)
    cos_pipe = get_pipeline(attn_type=AttentionType.cosformer)

    # Generate same random sample for all models
    num_inference_steps = 25
    batch_size = len(classes)
    latent_channels = sm_pipe.transformer.config.in_channels
    latent_size = sm_pipe.transformer.config.sample_size

    latents = torch.randn(batch_size, latent_channels, latent_size, latent_size)

    # Generate images
    class_ids = sm_pipe.get_label_ids(classes)

    sm_output = sm_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    soba_output = soba_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    lin_output = lin_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    #perf_output = perf_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    nys_output = nys_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    cos_output = cos_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)

    # Save images
    parent_dir = "./dit/generations"
    for (i, class_) in enumerate(classes):
        save_path = os.path.join(parent_dir, class_)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path_sm = os.path.join(save_path, "softmax.png")
        save_image(sm_output.images[i], save_path_sm)

        save_path_soba = os.path.join(save_path, "soba.png")
        save_image(soba_output.images[i], save_path_soba)

        save_path_lin = os.path.join(save_path, "linformer.png")
        save_image(lin_output.images[i], save_path_lin)

        # save_path_perf = os.path.join(save_path, "performer.png")
        # save_image(perf_output.images[i], save_path_perf)

        save_path_nys = os.path.join(save_path, "nystromformer.png")
        save_image(nys_output.images[i], save_path_nys)

        save_path_cos = os.path.join(save_path, "cosformer.png")
        save_image(cos_output.images[i], save_path_cos)



if __name__ == "__main__":
    main()
