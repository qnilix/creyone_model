from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from creyone_layer import create_layer
from creyone_layer.utils import ntuple

from ..utils import BaseCfg
from ..cynn import CreYonT


@dataclass
class PatchEmbedCfg(BaseCfg):
    """Configuration for PatchEmbed.

    Attributes:
        image_size: Spatial (H, W) of the input image; None disables static grid.
        image_chan: Number of input channels (e.g. 3 for RGB).
        patch_size: Side length of each square patch in pixels.
        num_frames: Number of video frames; 0 means image (2-D) mode.
        tubelet_size: Temporal depth of each tubelet when num_frames > 0.
        conv_cls: Key passed to ``create_layer`` to select the Conv class.
        bias: Whether the projection Conv has a bias term.
    """

    image_size: Optional[tuple[int, int]] = (224, 224)
    image_chan: int = 3
    patch_size: int = 16

    # for video
    num_frames: int = 0
    tubelet_size: int = 2

    conv_cls: str = 'base'
    bias: bool = True


class PatchEmbed(nn.Module):
    """2D image (or 3D video) to patch embedding via a strided convolution projection."""

    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            cfg: PatchEmbedCfg,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True
    ):
        """
        Args:
            cfg: Patch embedding configuration (image size, patch size, etc.).
            embed_dim: Output channel dimension for each patch token.
            norm_layer: Optional normalization applied after projection.
            flatten: If True, flatten spatial dims and transpose to (B, N, C);
                     otherwise permute to channel-last layout.
        """
        super().__init__()
        self.cfg = cfg
        self.patch_size = ntuple(2)(cfg.patch_size)

        s = self.patch_size; g = [s // p for s, p in zip(cfg.image_size, s)]
        if cfg.num_frames > 0:
            s = [cfg.tubelet_size] + list(s)
            g = [cfg.num_frames // cfg.tubelet_size] + g
        self.grid_size = tuple(g)

        conv_cls = create_layer(cfg.conv_cls, 'conv', nn.Conv2d)
        conv = conv_cls(len(s), optional='grid')
        self.proj = conv(cfg.image_chan, embed_dim,
                         kernel_size = s, bias = cfg.bias)

        self.flatten = flatten
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def cyns_config(self) -> dict:
        """Return serializable config fields consumed by CreYon serialization."""
        return {
            'image_size': self.cfg.image_size,
            'image_chan': self.cfg.image_chan
        }

    @property
    def num_patches(self) -> int:
        """Total number of patch tokens produced per sample."""
        v = 1
        for i in self.grid_size: v *= i
        return v

    def forward(self, x: CreYonT) -> CreYonT:
        """Project image to patch tokens and apply normalization.

        Args:
            x: Input image wrapped in CreYonT, shape (B, C, H, W) or (B, C, T, H, W).

        Returns:
            Patch token sequence of shape (B, N, embed_dim) when flatten=True.
        """
        x = x(self.proj)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NC... -> NLC
        else:
            x = x.permute(0, *(i for i in range(2, len(x.shape))), 1)
        return x(self.norm)