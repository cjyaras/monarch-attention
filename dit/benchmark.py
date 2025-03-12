from typing import List, Dict
import torch
import os

from dit.config import AttentionType
from dit.pipeline import get_pipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from torchvision.utils import save_image

from PIL import Image
import time

def save_output_images(output: ImagePipelineOutput, save_path: str):
    images = torch.Tensor(output.images)
    images_reshaped = images.permute(0, 3, 1, 2)
    save_image(images_reshaped, save_path)

def generate_attn_dict(attn_type: AttentionType, layers_to_replace: List, num_layers: int = 28) -> Dict:
    assert max(layers_to_replace) <= num_layers

    attn_dict = {}
    for i in range(1, num_layers + 1):
        attn_dict[i] = attn_type if i in layers_to_replace else AttentionType.softmax

    return attn_dict

# Performer model is only generating all black images
# so I have it commented out for now
@torch.no_grad()
def main():
    # Classes to generate
    classes = ["triceratops"]

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

    output_type = "numpy"

    sm_output = sm_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    soba_output = soba_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    lin_output = lin_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    #perf_output = perf_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    nys_output = nys_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    cos_output = cos_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)

    # Save images
    save_path = "./dit/generations"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sm_save_path = os.path.join(save_path, "softmax.png")
    save_output_images(sm_output, sm_save_path)

    soba_save_path = os.path.join(save_path, "soba.png")
    save_output_images(soba_output, soba_save_path)

    lin_save_path = os.path.join(save_path, "linformer.png")
    save_output_images(lin_output, lin_save_path)

    # save_path_perf = os.path.join(save_path, "performer.png")
    # save_output_images(perf_output, save_path_perf)

    nys_save_path = os.path.join(save_path, "nystromformer.png")
    save_output_images(nys_output, nys_save_path)

    cos_save_path = os.path.join(save_path, "cosformer.png")
    save_output_images(cos_output, cos_save_path)



if __name__ == "__main__":
    main()
