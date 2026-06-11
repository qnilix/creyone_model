from typing import Optional
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
from torchvision.transforms import Compose

from ..cynn import CreYonT
from ..base.config import ModelCfg
from ..base.module import ModuleBase
from .transform import ImageTransformCfg


@dataclass
class ImageProcessorCfg(ModelCfg):

    transform: ImageTransformCfg = field(default_factory=ImageTransformCfg)

    daug_type: str = None
    daug_ratio: float = 0.0

    norm_type: str = 'base' 

    def transform_wrap(self, image_size, **kwargs) -> Compose:
        return partial(self.transform.compose, image_size=image_size, **kwargs)


def normalize_rgb(a: str, b: str):
    if a == 'base': return (.5, .5, .5)
    if a == 'tune':
        if b == 'mean': return (0.48145466, 0.4578275, 0.40821073)
        return (0.26862954, 0.26130258, 0.27577711)
    if b == 'mean': return (103.530, 116.280, 123.675)
    return (1., 1., 1.)


def init_normalize(dim: int, norm_type: str, hparam: str, meta: str = 'rgb') -> torch.Tensor:
    _shape = [1, -1] + [1 for _ in range(dim)]
    t = normalize_rgb(norm_type, hparam)
    return torch.Tensor(t).reshape(_shape)


class RawImage(nn.Module):

    def __init__(self, cfg: ImageProcessorCfg, image_norm: str = None, **kwargs):
        super().__init__()
        norm = image_norm or cfg.norm_type
        self._image_mean = init_normalize(3, norm, 'mean')
        self._image_std = init_normalize(3, norm, 'std')
        self.daug_type = cfg.daug_type
        self.daug_lam = cfg.daug_ratio
        if cfg.daug_type == 'mixup': self.daug_lam = self.get_lambda()

    def forward(self, x: torch.Tensor) -> CreYonT:
        img = CreYonT(x)
        if not self.training: return img
        if self.daug_type == 'mixup': img = self.mixup(img)
        return img
    
    def mixup(self, img: CreYonT) -> CreYonT:
        img.lam = self.daug_lam.sample((img.shape[0], 1)).to(img.tensor())
        lam = img.lam[:, :, None, None]
        return img * lam + img.tensor().flip(dims=[0]) * (1 - lam)
    
    def get_lambda(self) -> torch.distributions.beta.Beta:
        return torch.distributions.beta.Beta(self.daug_ratio, self.daug_ratio)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x.sub_(self._image_mean.to(x.device))
        x.div_(self._image_std.to(x.device))
        return x

    def normalize_inverse(self, x: torch.Tensor) -> torch.Tensor:
        x.mul_(self._image_std.to(x.device))
        x.add_(self._image_mean.to(x.device))
        return x


class ImageProcessor(ModuleBase):

    def __init__(self, body: nn.Module, cfg: ImageProcessorCfg = None,
                 model_meta: Optional[dict] = None,
                 target_layers: Optional[list[str]] = None,
                 image_norm: str = None, 
                 crop_pct: Optional[float] = None,
                 crop_mode: Optional[str] = None,
                 interpolation: Optional[str] = None, **kwargs):
        self.cfg = self.get_cfg(cfg, **kwargs)
        super().__init__(model_meta=model_meta, target_layers=target_layers)
        self.body = body
        self.image_first = self.image_init(self.cfg, image_norm)
        self.image_size = body.cyns_config().get('image_size', (224, 224))
        self._image_transform = self.cfg.transform_wrap(
            self.image_size, crop_pct=crop_pct, crop_mode=crop_mode,
            interpolation=interpolation)
    
    def get_target_layers(self, target_layers: Optional[list[str]] = None) -> list:
        return self.cfg.target_layers(target_layers)
    
    def get_cfg(self, cfg): return cfg or None

    def image_init(self, cfg: ImageProcessorCfg, image_norm: str = None) -> RawImage:
        return RawImage(cfg, image_norm)

    def model_summary(self):
        return 
    
    def image_transform(self, is_training: bool = False):
        return self._image_transform(is_training)