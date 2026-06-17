from functools import partial

from ..utils.registry import register_model
from ..utils.book import BuildFlyer
from .base import Mlp, MlpCfg


__all__ = ['base']


def get_mlp(cfg: MlpCfg, module: Mlp, **kwargs):
    book = BuildFlyer(cfg, **kwargs)
    return partial(module, cfg=book.get_config()[0])


@register_model('mlp')
def base(*args, **kwargs) -> Mlp:
    return get_mlp(MlpCfg, Mlp, **kwargs)