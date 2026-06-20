from functools import partial

from ..utils.registry import register_model
from ..utils.book import BuildFlyer
from .base import Mlp, MlpCfg


__all__ = ['base']


def get_mlp(module: Mlp, 
            cfg: MlpCfg = None,
            cfg_cls: type[MlpCfg] = None, 
            **kwargs):
    if cfg is not None: return partial(module, cfg=cfg)
    book = BuildFlyer(cfg_cls or MlpCfg, **kwargs)
    return partial(module, cfg=book.get_config()[0])


@register_model('mlp')
def base(*args, cfg_cls: MlpCfg = None, **kwargs) -> Mlp:
    return get_mlp(Mlp, cfg_cls=cfg_cls, **kwargs)