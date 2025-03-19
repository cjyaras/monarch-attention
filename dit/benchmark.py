from typing import List, Dict, Optional
import torch
import os
import numpy as np

from dit.config import AttentionType
from dit.pipeline import get_pipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from torchvision.utils import save_image

NUM_SAMPLES = 4
IMAGES_PER_ROW = int(np.sqrt(NUM_SAMPLES))

def save_output_images(output: ImagePipelineOutput, save_path: str, num_images_per_row: Optional[int] = 8):
    images = torch.Tensor(output.images)
    images_reshaped = images.permute(0, 3, 1, 2)
    save_image(images_reshaped, save_path, nrow=num_images_per_row)

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
    # Set random seed for reproducibility
    seed = 2025
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Softmax pipeline
    sm_pipe = get_pipeline(attn_type=AttentionType.softmax)

    # Classes to generate and layers to replace
    all_classes = list( sm_pipe.labels.keys() )
    classes = np.random.choice(all_classes, size=NUM_SAMPLES)
    layers_to_replace = list(range(14)) #[*range(7), *range(21, 28)]

    # Generate same random sample for all models
    num_inference_steps = 25
    batch_size = len(classes)
    latent_channels = sm_pipe.transformer.config.in_channels
    latent_size = sm_pipe.transformer.config.sample_size

    latents = torch.randn(batch_size, latent_channels, latent_size, latent_size)

    # Generate and save images
    class_ids = sm_pipe.get_label_ids(classes)
    save_path = "./dit/generations/layers_" + str(layers_to_replace).replace('[', '').replace(']', '').replace(', ', '_')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    # Softmax
    output_type = "numpy"
    sm_output = sm_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    sm_save_path = os.path.join(save_path, "softmax.png")
    save_output_images(sm_output, sm_save_path, num_images_per_row=IMAGES_PER_ROW)


    # SOBA
    soba_attn_dict = generate_attn_dict(AttentionType.soba_monarch, layers_to_replace)
    soba_pipe = get_pipeline(attn_type=soba_attn_dict)
    soba_output = soba_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    soba_save_path = os.path.join(save_path, "soba.png")
    save_output_images(soba_output, soba_save_path, num_images_per_row=IMAGES_PER_ROW)


    # Linformer
    lin_attn_dict = generate_attn_dict(AttentionType.linformer, layers_to_replace)
    lin_pipe = get_pipeline(attn_type=lin_attn_dict, rank=64)
    lin_output = lin_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    lin_save_path = os.path.join(save_path, "linformer.png")
    save_output_images(lin_output, lin_save_path, num_images_per_row=IMAGES_PER_ROW)


    # Performer
    # perf_attn_dict = generate_attn_dict(AttentionType.performer, layers_to_replace)
    # perf_pipe = get_pipeline(attn_type=perf_attn_dict)
    # perf_output = perf_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    # save_path_perf = os.path.join(save_path, "performer.png")
    # save_output_images(perf_output, save_path_perf, num_images_per_row=IMAGES_PER_ROW)


    # Nystromformer
    nys_attn_dict = generate_attn_dict(AttentionType.nystromformer, layers_to_replace)
    nys_pipe = get_pipeline(attn_type=nys_attn_dict, rank=64)
    nys_output = nys_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    nys_save_path = os.path.join(save_path, "nystromformer.png")
    save_output_images(nys_output, nys_save_path, num_images_per_row=IMAGES_PER_ROW)

    # Cosformer
    cos_attn_dict = generate_attn_dict(AttentionType.cosformer, layers_to_replace)
    cos_pipe = get_pipeline(attn_type=cos_attn_dict)
    cos_output = cos_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type=output_type)
    cos_save_path = os.path.join(save_path, "cosformer.png")
    save_output_images(cos_output, cos_save_path, num_images_per_row=IMAGES_PER_ROW)



if __name__ == "__main__":
    main()
