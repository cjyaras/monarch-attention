import torch
import os

from dit.config import AttentionType
from dit.pipeline import get_pipeline

from PIL import Image
import time

def save_image(image: Image, save_path: str):
    image.save(save_path)


@torch.no_grad()
def main():
    # Classes to generate
    classes = ["triceratops"]

    # Make pipelines
    sm_pipe = get_pipeline(attn_type=AttentionType.softmax)
    soba_pipe = get_pipeline(attn_type=AttentionType.soba_monarch)
    lin_pipe = get_pipeline(attn_type=AttentionType.linformer)
    perf_pipe = get_pipeline(attn_type=AttentionType.performer)
    nys_pipe = get_pipeline(attn_type=AttentionType.nystromformer)
    cos_pipe = get_pipeline(attn_type=AttentionType.cosformer)

    # Generate same random sample
    num_inference_steps = 25
    batch_size = len(classes)
    latent_channels = sm_pipe.transformer.config.in_channels
    latent_size = sm_pipe.transformer.config.sample_size

    latents = torch.randn(batch_size, latent_channels, latent_size, latent_size)

    # Save path
    save_path = './dit/generations/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate images
    class_ids = sm_pipe.get_label_ids(classes)

    sm_output = sm_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    soba_output = soba_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    lin_output = lin_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    perf_output = perf_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    nys_output = nys_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)
    cos_output = cos_pipe(class_labels=class_ids, latents=latents, num_inference_steps=num_inference_steps)

    # Save images
    save_dir = "./dit/generations"
    for (i, class_) in enumerate(classes):
        save_path = os.path.join(save_dir, "softmax_" + class_ + ".png")
        save_image(sm_output.images[i], save_path)

        save_path = os.path.join(save_dir, "soba_" + class_ + ".png")
        save_image(soba_output.images[i], save_path)

        save_path = os.path.join(save_dir, "linformer_" + class_ + ".png")
        save_image(lin_output.images[i], save_path)

        save_path = os.path.join(save_dir, "performer_" + class_ + ".png")
        save_image(perf_output.images[i], save_path)

        save_path = os.path.join(save_dir, "nystromformer_" + class_ + ".png")
        save_image(nys_output.images[i], save_path)

        save_path = os.path.join(save_dir, "cosformer_" + class_ + ".png")
        save_image(cos_output.images[i], save_path)



if __name__ == "__main__":
    main()
