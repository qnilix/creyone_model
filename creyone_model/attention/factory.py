from functools import partial

from ..utils.registry import register_model
from ..utils.book import BuildFlyer
from .base import Attention, AttnCfg
from .lora import AttentionLoRA, AttnLoRACfg


__all__ = ['base']


def get_attention(module: Attention, 
                  cfg: AttnCfg = None, 
                  cfg_cls: type[AttnCfg] = None, **kwargs):
    if cfg is not None: return partial(module, cfg=cfg)
    book = BuildFlyer(cfg_cls or AttnCfg, **kwargs)
    return partial(module, cfg=book.get_config()[0])


@register_model('attn')
def base(*args, cfg_cls: type[AttnCfg] = None, **kwargs) -> Attention:
    return get_attention(Attention, cfg_cls=cfg_cls, **kwargs)

@register_model('attn')
def lora(*args, cfg_cls: type[AttnLoRACfg] = None, **kwargs) -> AttentionLoRA:
    return get_attention(AttentionLoRA, cfg_cls=cfg_cls or AttnLoRACfg, **kwargs)