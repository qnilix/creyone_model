"""Stacked transformer encoder built from :class:`~creyone_model.transformer.block.Block` sub-layers.

Defines :class:`TransformerCfg` (configuration dataclass) and :class:`Transformer` (``nn.Module``).
The encoder applies optional pre/post normalization around a sequence of transformer blocks::

    x = pre_norm(x)
    for block in blocks:
        x = block(x)
    x = post_norm(x)

References:
    `Attention Is All You Need` (Transformer)
        - https://arxiv.org/abs/1706.03762
    `Training data-efficient image transformers & distillation through attention` (DeiT; first to apply stochastic depth to ViT)
        - https://arxiv.org/abs/2012.12877
    `Deep Networks with Stochastic Depth` (DropPath / stochastic depth)
        - https://arxiv.org/abs/1603.09382
"""

from typing import Optional
from dataclasses import dataclass, field

import torch
from torch import nn
from creyone_layer import create_layer

from ..cynn import CreYonT
from ..utils import BaseCfg
from .block import BlockCfg


@dataclass
class TransformerCfg(BaseCfg):
    """Configuration dataclass for the stacked transformer encoder.

    Attributes:
        depth: Number of transformer blocks.
        embed_dim: Token embedding dimension shared across all blocks.
        norm_eps: Epsilon for the normalization layer.
        pre_norm: If ``True``, apply a normalization layer before the block stack.
        post_norm: If ``True``, apply a normalization layer after the block stack.
        drop_path_rate: Maximum stochastic depth drop probability; linearly
            scaled per block from 0 to this value.
        block: Configuration for each transformer block.
    """

    depth: int = 12
    embed_dim: int = 768

    norm_eps: float = 1e-6
    pre_norm: bool = False
    post_norm: bool = False

    drop_path_rate: float = .0

    block: BlockCfg = field(default_factory=BlockCfg)

    def block_kwargs(self, idx: int) -> dict:
        """Return per-block keyword arguments for block index ``idx``.

        The stochastic depth rate is linearly scaled so that the first block
        receives 0 and the last block receives ``drop_path_rate``.

        Args:
            idx: Zero-based index of the block within the encoder.

        Returns:
            A dict containing ``path_drop`` and ``layer_id`` for the block constructor.
        """
        return {
            'path_drop': idx * self.drop_path_rate / self.depth,
            'layer_id': idx
        }


class Transformer(nn.Module):
    """Stacked transformer encoder.

    Applies an optional pre-norm, a sequence of transformer blocks, and an optional
    post-norm to the input token tensor.

    Args:
        cfg: Encoder configuration produced by :class:`TransformerCfg`.
        norm_layer: Normalization layer factory (e.g. ``nn.LayerNorm``).  When
            ``None``, a standard LayerNorm with ``cfg.norm_eps`` is used.
    """

    def __init__(self,
                 cfg: TransformerCfg,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()

        self.embed_dim = cfg.embed_dim
        self.block_depth = cfg.depth

        if norm_layer is None:
            norm_layer = create_layer('layer', 'norm')(eps=cfg.norm_eps)

        self.norm_pre = norm_layer(self.embed_dim) if cfg.pre_norm else nn.Identity()
        self.norm = norm_layer(self.embed_dim) if cfg.post_norm else nn.Identity()

        for i in range(cfg.depth):
            block = cfg.block.block_module(self.embed_dim, norm_layer=norm_layer, **cfg.block_kwargs(i))
            self.add_module(f'block{i}', block)
    
    def forward(self, x: CreYonT) -> CreYonT:
        """Pass tokens through pre-norm, all blocks, and post-norm.

        Args:
            x: Input token tensor wrapped as :class:`~creyone_model.cynn.CreYonT`.

        Returns:
            Transformed token tensor of the same shape.
        """
        x = x(self.norm_pre)
        for i in range(self.block_depth):
            x = self.get_submodule(f'block{i}')(x)
        return x(self.norm)
    
    def trainable_parameters(self, mode: str = ''):
        """Set which parameters require gradients.

        Args:
            mode: One of ``'all'`` (unfreeze everything), ``'none'`` (freeze
                everything), or a block-level mode forwarded to each block's
                ``trainable_parameters`` method (e.g. ``'add'`` for adapters).
        """
        if mode == 'all': self.requires_grad_(True); return []
        self.requires_grad_(False)
        if mode == 'none': return []
        for i in range(self.block_depth):
            self.get_submodule(f'block{i}').trainable_parameters(mode)
    
    def reset_parameters(self, mode: str = 'trunc_'):
        """Re-initialize weights of all transformer blocks.

        Args:
            mode: Initialization scheme forwarded to each block's
                ``reset_parameters`` method (e.g. ``'trunc_'`` for
                truncated-normal initialization).
        """
        for i in range(self.block_depth):
            self.get_submodule(f'block{i}').reset_parameters(mode)
