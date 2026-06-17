import re

from ..utils import ModelBuilder, BuildShelf

from .base import ViT, ViTCfg
from .pretrained import pret_cfgs


MAGN_KEYS = ('embed_dim', 'depth', 'num_heads')
MAGNITUDE = {'tiny': ( 192, 12, 3), 'small': ( 384, 12, 6), 'base':  ( 768, 12, 12), 'large': (1024, 24, 16)}


def vit_update_kwargs(cfgs: dict, size: str, patch: int, res: int):
    cfgs.update({k: v for k, v in zip(MAGN_KEYS, MAGNITUDE[size])})
    cfgs['patch_size'] = int(patch)
    cfgs['image_size'] = ViT.solve_resolution(res)


def get_encoder_vit(size: str, patch: str, image_size: str, *_,
                    shelf: type[BuildShelf] = BuildShelf,
                    default: type[ViT] = ViT, default_cfg: type[ViTCfg] = ViTCfg,
                    keep_head: bool = True, **kwargs):
    #def vit(size: str, patch: str = '16', image_size: str = '224', *_,
    #    default_cfg: type[ViTCfg] = ViTCfg,
    #    keep_head: bool = True,
    #    **kwargs) -> ViT:
    """vit
        Wshen calling the model, specify it as vit_<size>_<patch>_<img_size>.
            e.g. vit_base_16_224
        Note that img_size is also used when calling a pre-trained model. 
        To specify a new image resolution, use <img_size(new_size)>.
            e.g. vit_base_16_224(256) or vit_base_16_224(128x256)
    """
    var_size, new_size = re.match(r'(\d+)(\([0-9x]+\))?', image_size).groups()
    shelf = shelf(default_cfg, pret_cfgs(keep_head))
    builder = ModelBuilder(default = default_cfg, shelf = shelf)
    vit_update_kwargs(kwargs, size, patch, new_size or var_size)
    book = builder.ref_shelf(variant=f'vit-{size}-patch{patch}-{var_size}', **kwargs)
    return builder.build(default, book=book, **kwargs)