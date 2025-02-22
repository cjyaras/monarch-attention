from dit.custom_model import CustomDiTTransformer2DModel
from dit.custom_attention_processor import EfficientAttnConfig
from diffusers import DiTTransformer2DModel, DiTPipeline, DPMSolverMultistepScheduler
import torch
import os

import time

model_path = "facebook/DiT-XL-2-256"
model_subfolder = "transformer"

# Make pipeline with softmax attention Transformer
sm_pipe = DiTPipeline.from_pretrained(model_path)

# Make pipeline with SOBA attention Transformer
attn_type = "soba"
attn_config = EfficientAttnConfig(efficient_attention_type=attn_type)
soba_model = CustomDiTTransformer2DModel.from_pretrained(model_path, subfolder=model_subfolder, efficient_attention_config=attn_config)
soba_pipe = DiTPipeline.from_pretrained(model_path)
soba_pipe.transformer = soba_model

# pick words that exist in ImageNet
idx = 0 
words = ["triceratops"]

# Seed and inference steps
generator = torch.manual_seed(33)
num_inference_steps = 25

# Save images
save_path = './dit/generations/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Get softmax attention Transformer generation
# class_ids = sm_pipe.get_label_ids(words)

# start = time.time()
# sm_output = sm_pipe(class_labels=class_ids, num_inference_steps=num_inference_steps, generator=generator)
# sm_gen_time = time.time() - start

# print("Softmax:", sm_gen_time, "seconds for", num_inference_steps, "inference steps")

# sm_image = sm_output.images[idx] 
# sm_image.save(os.path.join(save_path, "softmax_" + words[idx] + ".png"))


# Get SOBA attention Transformer generation
class_ids = soba_pipe.get_label_ids(words)

start = time.time()
soba_output = soba_pipe(class_labels=class_ids, num_inference_steps=num_inference_steps, generator=generator)
soba_gen_time = time.time() - start

print("soba:", soba_gen_time, "seconds for", num_inference_steps, "inference steps")

soba_image = soba_output.images[idx]  
soba_image.save(os.path.join(save_path, "soba_" + words[idx] + ".png"))
