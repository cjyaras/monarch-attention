from typing import List, Dict

from dit.config import AttentionType
from dit.pipeline import get_pipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
import torch
from torchvision.utils import save_image
import os

import time

def save_output_images(output: ImagePipelineOutput, save_path: str):
    images = torch.Tensor(output.images)
    images_reshaped = images.permute(0, 3, 1, 2)
    save_image(images_reshaped, save_path)



# Create dict of which attention layers should be replaced
def generate_attn_dict(attn_type: AttentionType, layers_to_replace: List, num_layers: int = 28) -> Dict:
    assert max(layers_to_replace) <= num_layers

    attn_dict = {}
    for i in range(1, num_layers + 1):
        attn_dict[i] = attn_type if i in layers_to_replace else AttentionType.softmax

    return attn_dict


# Make pipelines
model_path = "facebook/DiT-XL-2-256"
model_subfolder = "transformer"


layers_to_replace = list(range(14))
attn_dict = generate_attn_dict(AttentionType.soba_monarch, layers_to_replace)

sm_pipe = get_pipeline(attn_type=AttentionType.softmax, model_path=model_path, model_subfolder=model_subfolder)
soba_pipe = get_pipeline(attn_type=attn_dict, model_path=model_path, model_subfolder=model_subfolder)


# pick words that exist in ImageNet
#idx = 0 
words = ["triceratops", "German shepherd"]


# Seed and inference steps
# generator = torch.manual_seed(33)
num_inference_steps = 25
batch_size = len(words)
latent_channels = sm_pipe.transformer.config.in_channels
latent_size = sm_pipe.transformer.config.sample_size

# Use same random sample for both models
latents = torch.randn(len(words), latent_channels, latent_size, latent_size)


# Save images
save_path = './dit/generations/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Get softmax attention Transformer generation
class_ids = sm_pipe.get_label_ids(words)

start = time.time()
sm_output = sm_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type = "numpy")
sm_gen_time = time.time() - start

print("Softmax:", sm_gen_time, "seconds for", num_inference_steps, "inference steps")
sm_save_path = os.path.join(save_path, "softmax.png")
save_output_images(sm_output, sm_save_path)


# Get SOBA attention Transformer generation
class_ids = soba_pipe.get_label_ids(words)

start = time.time()
soba_output = soba_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps, output_type = "numpy") 
soba_gen_time = time.time() - start

print("soba:", soba_gen_time, "seconds for", num_inference_steps, "inference steps")
soba_save_path = os.path.join(save_path, "soba_layers_" + str(layers_to_replace[0]) + "_" + str(layers_to_replace[-1])) + ".png"
save_output_images(soba_output, soba_save_path)
