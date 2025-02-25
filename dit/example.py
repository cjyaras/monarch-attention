from dit.config import AttentionType
from dit.pipeline import get_pipeline
import torch
import os

import time

# Make pipelines
model_path = "facebook/DiT-XL-2-256"
model_subfolder = "transformer"

sm_pipe = get_pipeline(attn_type=AttentionType.softmax, model_path=model_path, model_subfolder=model_subfolder)
soba_pipe = get_pipeline(attn_type=AttentionType.soba_monarch, model_path=model_path, model_subfolder=model_subfolder)


# pick words that exist in ImageNet
idx = 0 
words = ["triceratops"]


# Seed and inference steps
# generator = torch.manual_seed(33)
num_inference_steps = 25
batch_size = len(words)
latent_channels = sm_pipe.transformer.config.in_channels
latent_size = sm_pipe.transformer.config.sample_size

latents = torch.randn(len(words), latent_channels, latent_size, latent_size)


# Save images
save_path = './dit/generations/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Get softmax attention Transformer generation
class_ids = sm_pipe.get_label_ids(words)

start = time.time()
sm_output = sm_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
sm_gen_time = time.time() - start

print("Softmax:", sm_gen_time, "seconds for", num_inference_steps, "inference steps")

sm_image = sm_output.images[idx] 
sm_image.save(os.path.join(save_path, "softmax_" + words[idx] + ".png"))


# Get SOBA attention Transformer generation
class_ids = soba_pipe.get_label_ids(words)

start = time.time()
soba_output = soba_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps) 
soba_gen_time = time.time() - start

print("soba:", soba_gen_time, "seconds for", num_inference_steps, "inference steps")

soba_image = soba_output.images[idx]  
soba_image.save(os.path.join(save_path, "soba_" + words[idx] + ".png"))
