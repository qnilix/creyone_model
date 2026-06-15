from functools import partial

from ..utils.registry import register_model
from ..utils.book import BuildFlyer
from .base import Attention, AttnCfg


__all__ = ['base']


def get_attention(cfg: AttnCfg, module: Attention, **kwargs):
    book = BuildFlyer(cfg, **kwargs)
    return partial(module, cfg=book.get_config()[0])


@register_model('attn')
def base(*args, **kwargs) -> Attention:
    return get_attention(AttnCfg, Attention, **kwargs)