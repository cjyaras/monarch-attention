from transformers.models.vit import ViTImageProcessorFast


def get_processor():
    return ViTImageProcessorFast.from_pretrained("google/vit-base-patch16-224")
