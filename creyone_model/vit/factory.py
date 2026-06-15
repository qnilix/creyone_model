from ..utils.registry import register_model

from .base import ViT
from .utils import get_encoder_vit


__all__ = ['vit']


@register_model('image_encoder')
def vit(*args, **kwargs) -> ViT: return get_encoder_vit(*args, **kwargs)