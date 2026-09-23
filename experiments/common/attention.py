"""Route Hugging Face attention layers through this repo's attention modules.

Models whose config sets ``_attn_implementation = ATTN_IMPLEMENTATION`` call
``custom_attention_forward`` in every attention layer. Layers that have an
``attn_module`` attribute use it; all other layers (e.g. the BART decoder)
fall back to standard SDPA attention.
"""

from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

ATTN_IMPLEMENTATION = "custom"


def custom_attention_forward(module, query, key, value, attention_mask, **kwargs):
    attn_module = getattr(module, "attn_module", None)
    if attn_module is None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        )

    if attention_mask is not None:
        # (B, 1, Q, K) boolean mask from sdpa_mask -> (B, K) padding mask, 1 = keep
        attention_mask = attention_mask[:, 0, 0, :].to(query.dtype)

    attn_output = attn_module(query, key, value, attention_mask)
    return attn_output.transpose(1, 2).contiguous(), None


AttentionInterface.register(ATTN_IMPLEMENTATION, custom_attention_forward)
AttentionMaskInterface.register(ATTN_IMPLEMENTATION, sdpa_mask)
