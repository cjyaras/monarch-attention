from typing import List, Dict
import torch
import os
import numpy as np

from math import ceil 
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp

from dit.config import AttentionType
from dit.pipeline import get_pipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

from common.utils import get_device

GLOBAL_SEED = 0
NUM_INFERENCE_STEPS = 32
BATCH_SIZE = 32
NUM_SAMPLES = 50000
NUM_CLASSES = 1842
device = get_device()


# def prepare_ddp():
#     assert torch.cuda.is_available()

#     world_size = torch.cuda.device_count()
#     if world_size == 1:
#         device = "cuda"
#         seed = GLOBAL_SEED
#     else:
#         dist.init_process_group("nccl", world_size=world_size)
#         rank = dist.get_rank()
#         device = rank % world_size
#         seed = GLOBAL_SEED * dist.get_world_size() + rank

#     return world_size, device, seed


def save_images_as_npz(output: ImagePipelineOutput, save_fname: str):
    images = output.images # (B x H x W x C)
    np.savez(save_fname, arr0=images)
    print(f"Saved .npz file to {save_fname} [shape = {images.shape}]")


def generate_attn_dict(attn_type: AttentionType, layers_to_replace: List, num_layers: int = 28) -> Dict:
    assert max(layers_to_replace) <= num_layers

    attn_dict = {}
    for i in range(1, num_layers + 1):
        attn_dict[i] = attn_type if i in layers_to_replace else AttentionType.softmax

    return attn_dict


def prepare_sampling(num_classes: int = 1000):
    tmp_pipe = get_pipeline(attn_type = AttentionType.softmax)
    all_classes = list( tmp_pipe.labels.keys() )

    assert num_classes <= len(all_classes)
    if num_classes == len(all_classes):
        classes_to_sample = all_classes
    else:
        classes_to_sample = np.random.choice(np.array(all_classes), size=num_classes)

    all_class_ids = tmp_pipe.get_label_ids(classes_to_sample)
    latent_channels = tmp_pipe.transformer.config.in_channels
    latent_size = tmp_pipe.transformer.config.sample_size

    del tmp_pipe

    return all_class_ids, latent_channels, latent_size


# Performer model is only generating all black images
# so I have it commented out for now
@torch.no_grad()
def main():
    # Prepare sampling process
    all_class_ids, latent_channels, latent_size = prepare_sampling(NUM_CLASSES)
    total_samples = int(ceil(NUM_SAMPLES / BATCH_SIZE)) * BATCH_SIZE
    
    layers_to_replace = list(range(14)) #[*range(7), *range(21, 28)]
    output_type = "numpy"

    #save_path = "./dit/generations/layers_" + str(layers_to_replace).replace('[', '').replace(']', '').replace(', ', '_') + "_" + str(num_inference_steps) + "_sampling_steps"
    save_path = "./dit/generations/all_layers_" + str(NUM_INFERENCE_STEPS) + "_sampling_steps"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Begin sampling
    num_iters = int(total_samples // BATCH_SIZE)
    pbar = tqdm(range(num_iters))
    for (i, _) in enumerate(pbar):  
        class_ids = list( np.random.choice( all_class_ids, size=BATCH_SIZE ) ) 
        latents = torch.randn(BATCH_SIZE, latent_channels, latent_size, latent_size) # Use same latents for all models

        # Softmax output
        sm_pipe = get_pipeline(attn_type=AttentionType.softmax, device=device)
        sm_output = sm_pipe(class_labels=class_ids, latents=latents, num_inference_steps=NUM_INFERENCE_STEPS, output_type=output_type)
        sm_save_path = os.path.join(save_path, "softmax")
        if not os.path.exists(sm_save_path):
            os.makedirs(sm_save_path)
        sm_save_fname = os.path.join(sm_save_path, "batch_" + str(i))
        save_images_as_npz(sm_output, sm_save_fname)
        del sm_pipe
        del sm_output


        # SOBA
        # soba_attn_dict = generate_attn_dict(AttentionType.soba_monarch, layers_to_replace)
        soba_pipe = get_pipeline(attn_type=AttentionType.soba_monarch, device=device)
        soba_output = soba_pipe(class_labels=class_ids, latents=latents, num_inference_steps=NUM_INFERENCE_STEPS, output_type=output_type)
        soba_save_path = os.path.join(save_path, "soba")
        if not os.path.exists(soba_save_path):
            os.makedirs(soba_save_path)
        soba_save_fname = os.path.join(soba_save_path, "batch_" + str(i))
        save_images_as_npz(soba_output, soba_save_fname)
        del soba_pipe
        del soba_output
    

        # Linformer
        # lin_attn_dict = generate_attn_dict(AttentionType.linformer, layers_to_replace)
        lin_pipe = get_pipeline(attn_type=AttentionType.linformer, rank=64, device=device, module_device=device)
        lin_output = lin_pipe(class_labels=class_ids, latents=latents, num_inference_steps=NUM_INFERENCE_STEPS, output_type=output_type)
        lin_save_path = os.path.join(save_path, "linformer")
        if not os.path.exists(lin_save_path):
            os.makedirs(lin_save_path)
        lin_save_fname = os.path.join(lin_save_path, "batch_" + str(i)) 
        save_images_as_npz(lin_output, lin_save_fname)
        del lin_pipe
        del lin_output

        # Performer
        # perf_attn_dict = generate_attn_dict(AttentionType.performer, layers_to_replace)
        # perf_pipe = get_pipeline(attn_type=AttentionType.performer, device=device, module_device=device)
        # perf_output = perf_pipe(class_labels=class_ids, latents=latents, num_inference_steps=NUM_INFERENCE_STEPS, output_type=output_type)
        # perf_save_path = os.path.join(save_path, "performer")
        # if not os.path.exists(perf_save_path):
        #    os.makedirs(perf_save_path)
        # perf_save_fname = os.path.join(perf_save_path, "batch_" + str(i))
        # save_images_as_npz(perf_output, perf_save_fname)
        # del perf_pipe
        # del perf_output


        # Nystromformer
        # nys_attn_dict = generate_attn_dict(AttentionType.nystromformer, layers_to_replace)
        nys_pipe = get_pipeline(attn_type=AttentionType.nystromformer, rank=64, device=device)
        nys_output = nys_pipe(class_labels=class_ids, latents=latents, num_inference_steps=NUM_INFERENCE_STEPS, output_type=output_type)
        nys_save_path = os.path.join(save_path, "nystromformer")
        if not os.path.exists(nys_save_path):
            os.makedirs(nys_save_path)
        nys_save_fname = os.path.join(nys_save_path, "batch_" + str(i))
        save_images_as_npz(nys_output, nys_save_fname)
        del nys_pipe
        del nys_output


        # Cosformer
        # cos_attn_dict = generate_attn_dict(AttentionType.cosformer, layers_to_replace)
        cos_pipe = get_pipeline(attn_type=AttentionType.cosformer, device=device)
        cos_output = cos_pipe(class_labels=class_ids, latents=latents, num_inference_steps=NUM_INFERENCE_STEPS, output_type=output_type)
        cos_save_path = os.path.join(save_path, "cosformer")
        if not os.path.exists(cos_save_path):
            os.makedirs(cos_save_path)
        cos_save_fname = os.path.join(cos_save_path, "batch_" + str(i))
        save_images_as_npz(cos_output, cos_save_fname)
        del cos_pipe
        del cos_output


if __name__ == "__main__":
    main()
