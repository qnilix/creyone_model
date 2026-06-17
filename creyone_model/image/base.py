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
    """Configuration dataclass for image preprocessing and augmentation.

    Attributes:
        transform: Transform pipeline configuration.
        daug_type: Data augmentation type (e.g. ``'mixup'``). ``None`` disables augmentation.
        daug_ratio: Strength parameter for data augmentation (e.g. Beta distribution alpha for mixup).
        norm_type: Normalization preset. ``'base'`` uses 0.5/0.5, ``'tune'`` uses CLIP statistics.
    """

    transform: ImageTransformCfg = field(default_factory=ImageTransformCfg)

    daug_type: str = None
    daug_ratio: float = 0.0

    norm_type: str = 'base'

    def transform_wrap(self, image_size, **kwargs) -> Compose:
        """Return a partial of the transform compose function bound to ``image_size``."""
        return partial(self.transform.compose, image_size=image_size, **kwargs)


def normalize_rgb(a: str, b: str):
    """Return an RGB normalization constant (mean or std) for a given norm preset.

    Args:
        a: Norm type — ``'base'`` for 0.5 triplet, ``'tune'`` for CLIP statistics,
           anything else for ImageNet BGR-order constants.
        b: Which statistic to return — ``'mean'`` or ``'std'``.

    Returns:
        A 3-tuple of float values for R, G, B channels.
    """
    if a == 'base': return (.5, .5, .5)
    if a == 'tune':
        if b == 'mean': return (0.48145466, 0.4578275, 0.40821073)
        return (0.26862954, 0.26130258, 0.27577711)
    if b == 'mean': return (103.530, 116.280, 123.675)
    return (1., 1., 1.)


def init_normalize(dim: int, norm_type: str, hparam: str, meta: str = 'rgb') -> torch.Tensor:
    """Build a normalization tensor shaped for broadcasting over image tensors.

    Args:
        dim: Number of spatial dimensions (e.g. 2 for H×W images).
        norm_type: Norm preset forwarded to :func:`normalize_rgb`.
        hparam: Statistic to retrieve — ``'mean'`` or ``'std'``.
        meta: Reserved for future colour-space variants (currently unused).

    Returns:
        A ``(1, C, 1, …, 1)`` tensor ready to broadcast against a batch of images.
    """
    _shape = [1, -1] + [1 for _ in range(dim)]
    t = normalize_rgb(norm_type, hparam)
    return torch.Tensor(t).reshape(_shape)


class RawImage(nn.Module):
    """Lightweight preprocessing module that applies data augmentation during training.

    Registers per-channel mean and std as non-persistent buffers so they travel
    with the model but are excluded from ``state_dict``.  Supports mixup augmentation
    when ``cfg.daug_type == 'mixup'``.
    """

    tensor_dims: int = 2

    def __init__(self, cfg: ImageProcessorCfg, image_norm: str = None, **kwargs):
        """
        Args:
            cfg: Image processor configuration supplying norm type and augmentation settings.
            image_norm: Override for ``cfg.norm_type`` when loading a pretrained checkpoint
                        with a different normalization scheme.
        """
        super().__init__()
        norm = image_norm or cfg.norm_type
        self._image_mean = init_normalize(self.tensor_dims, norm, 'mean')
        self._image_std = init_normalize(self.tensor_dims, norm, 'std')
        self.daug_type = cfg.daug_type
        self.daug_lam = cfg.daug_ratio
        if cfg.daug_type == 'mixup': self.daug_lam = self.get_lambda()

    def forward(self, x: torch.Tensor) -> CreYonT:
        """Wrap the input tensor and apply training-time augmentation if enabled."""
        img = CreYonT(x)
        if not self.training: return img
        if self.daug_type == 'mixup': img = self.mixup(img)
        return img
    
    def mixup(self, img: CreYonT) -> CreYonT:
        """Apply mixup augmentation by linearly interpolating each sample with a shuffled peer."""
        img.lam = self.daug_lam.sample((img.shape[0], 1)).to(img.tensor())
        for i in range(self.tensor_dims): img.lam = img.lam.unsqueeze(2+i)
        return img * img.lam + img.tensor().flip(dims=[0]) * (1 - img.lam)
    
    def get_lambda(self) -> torch.distributions.beta.Beta:
        """Return a symmetric Beta distribution used to sample mixup coefficients."""
        return torch.distributions.beta.Beta(self.daug_ratio, self.daug_ratio)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``x`` in-place by subtracting the mean and dividing by the std."""
        x.sub_(self._image_mean.to(x.device))
        x.div_(self._image_std.to(x.device))
        return x

    def normalize_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Undo normalization in-place — multiply by std then add mean."""
        x.mul_(self._image_std.to(x.device))
        x.add_(self._image_mean.to(x.device))
        return x


class ImageProcessor(ModuleBase):
    """Top-level image model wrapper that couples a backbone with preprocessing.

    Composes a :class:`RawImage` augmentation/normalization front-end, the backbone
    ``body``, and a configurable ``torchvision`` transform pipeline suitable for both
    training and inference.
    """

    def __init__(self, body: nn.Module, cfg: ImageProcessorCfg = None,
                 model_meta: Optional[dict] = None,
                 target_layers: Optional[list[str]] = None,
                 image_norm: str = None,
                 crop_pct: Optional[float] = None,
                 crop_mode: Optional[str] = None,
                 interpolation: Optional[str] = None, **kwargs):
        """
        Args:
            body: Backbone model exposing a ``cyns_config()`` method.
            cfg: Image processor configuration; defaults to ``None`` (subclass may override via ``get_cfg``).
            model_meta: Optional metadata dict forwarded to :class:`ModuleBase`.
            target_layers: Layer names used for feature extraction or GradCAM.
            image_norm: Normalization preset override (e.g. ``'tune'`` for CLIP weights).
            crop_pct: Fraction of the image to crop during inference resizing.
            crop_mode: Cropping strategy (e.g. ``'center'``, ``'squash'``).
            interpolation: Resampling filter (e.g. ``'bicubic'``).
        """
        self.cfg = self.get_cfg(cfg, **kwargs)
        super().__init__(model_meta=model_meta, target_layers=target_layers)
        self.body = body
        self.image_first = self.image_init(self.cfg, image_norm)
        self.image_size = body.cyns_config().get('image_size', (224, 224))
        self._image_transform = self.cfg.transform_wrap(
            self.image_size, crop_pct=crop_pct, crop_mode=crop_mode,
            interpolation=interpolation)
    
    def get_target_layers(self, target_layers: Optional[list[str]] = None) -> list:
        """Resolve target layer names via the config, falling back to ``target_layers``."""
        return self.cfg.target_layers(target_layers)

    def get_cfg(self, cfg): return cfg or None

    def image_init(self, cfg: ImageProcessorCfg, image_norm: str = None) -> RawImage:
        """Instantiate the :class:`RawImage` preprocessing front-end."""
        return RawImage(cfg, image_norm)

    def model_summary(self):
        """Return a human-readable model summary (not yet implemented)."""
        return

    def image_transform(self, is_training: bool = False):
        """Return the composed torchvision transform for the given mode.

        Args:
            is_training: If ``True``, returns the training-time transform (with augmentation);
                         otherwise returns the inference transform.
        """
        return self._image_transform(is_training)