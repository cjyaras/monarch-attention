from typing import Optional, Dict, List, Union

import torch

from dit.config import AttentionType, get_config
from dit.model import CustomDiTTransformer2DModel, get_model
from diffusers import DiTPipeline
from diffusers.models import DiTTransformer2DModel
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor

from common.utils import get_device

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class CustomDiTPipeline(DiTPipeline):
    def __init__(
        self,
        transformer: DiTTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[Dict[int, str]] = None
    ):
        
        super().__init__(transformer,
                         vae,
                         scheduler,
                         id2label)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        latents: Optional[torch.Tensor] = None, # Optional random sample input
        guidance_scale: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True
    ):

        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        if latents is not None: # Use random sample from input
            assert latents.shape == (batch_size, latent_channels, latent_size, latent_size)
            latents = latents.to(self.transformer.dtype)
            latents = latents.to(self._execution_device)
        else: # Generate random sample
            latents = randn_tensor(
                shape=(batch_size, latent_channels, latent_size, latent_size),
                generator=generator,
                device=self._execution_device,
                dtype=self.transformer.dtype,
            )

        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                is_npu = latent_model_input.device.type == "npu"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if (is_mps or is_npu) else torch.float64
                else:
                    dtype = torch.int32 if (is_mps or is_npu) else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input, timestep=timesteps, class_labels=class_labels_input
            ).sample

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

            if XLA_AVAILABLE:
                xm.mark_step()

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)

        
def get_pipeline(attn_type: Union[AttentionType, Dict[int, AttentionType]],
                 model_path: str = "facebook/DiT-XL-2-256",
                 model_subfolder: str = "transformer",
                 **kwargs):

    device = get_device()
    config = get_config(attention_type=attn_type, **kwargs)
    model = get_model(config, model_path, model_subfolder)

    pipe = CustomDiTPipeline.from_pretrained(model_path)
    pipe.transformer = model

    return pipe.to(device)