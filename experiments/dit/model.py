from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel

from experiments.common.utils import get_device
from experiments.dit.attention_processor import EfficientAttnProcessor
from experiments.dit.config import EfficientAttnConfig

NUM_LAYERS = 28  # DiT-XL/2


def get_model(
    config: EfficientAttnConfig,
    model_path: str = "facebook/DiT-XL-2-256",
    model_subfolder: str = "transformer",
) -> DiTTransformer2DModel:
    model = DiTTransformer2DModel.from_pretrained(model_path, subfolder=model_subfolder)
    for layer_num, block in enumerate(model.transformer_blocks):
        block.attn1.set_processor(EfficientAttnProcessor(config, layer_num))
    model = model.to(get_device())  # type: ignore
    model.eval()
    return model
