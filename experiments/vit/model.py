
from transformers.models.vit.modeling_vit import (
    ViTForImageClassification,
    ViTModel,
)
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.utils.logging import ERROR, set_verbosity  # type: ignore

from experiments.common.attention import get_attn_module
from experiments.common.utils import get_device
from experiments.vit.config import CustomViTConfig

set_verbosity(ERROR)

class CustomViTModel(ViTModel):

    def __init__(
        self,
        config: CustomViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__(config, add_pooling_layer, use_mask_token)
        for layer_num, layer in enumerate(self.layers):
            layer.attention.attn_module = get_attn_module(config, layer_num)  # type: ignore
        self.post_init()


# Transformers only renames old-format checkpoint weights for its own classes.
register_checkpoint_conversion_mapping(
    "CustomViTModel", get_checkpoint_conversion_mapping("ViTModel")
)


class CustomViTForImageClassification(ViTForImageClassification):

    def __init__(self, config: CustomViTConfig):
        super().__init__(config)
        self.vit = CustomViTModel(config, add_pooling_layer=False)
        self.post_init()


def get_model(config: CustomViTConfig) -> CustomViTForImageClassification:
    device = get_device()
    model = CustomViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", config=config
    )
    model = model.to(device)  # type: ignore
    model.eval()
    return model
