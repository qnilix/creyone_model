"""Vision Transformer (ViT) model with optional learnable prompt tokens.

Defines :class:`ViTCfg` (configuration dataclass) and :class:`ViT` (``nn.Module``).
The forward pass follows the standard ViT pipeline::

    x = patch_embed(x)          # split image into flattened patch tokens
    x = embed(x, prompts)       # prepend prompt tokens + add positional embedding
    x = transformer(x)          # stacked self-attention blocks
    x = head(x)                 # classification / projection head

References:
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale` (ViT)
        - https://arxiv.org/abs/2010.11929
    `Training data-efficient image transformers & distillation through attention` (DeiT)
        - https://arxiv.org/abs/2012.12877
    `Visual Prompt Tuning` (VPT; learnable prompt tokens prepended to the sequence)
        - https://arxiv.org/abs/2203.12119
"""

import re
from typing import Iterator, Optional, Union
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from ..cynn import CreYonT
from ..embed import PatchEmbed, PatchEmbedCfg, ViTEmbed, ViTEmbedCfg
from ..utils import BaseCfg
from ..transformer import Transformer, TransformerCfg

from .head import ViTHead

@dataclass
class ViTCfg(BaseCfg):
    """Configuration dataclass for Vision Transformer (ViT) models.

    Composes sub-configs for the transformer backbone, patch embedding, and
    positional embedding, and provides factory methods that instantiate the
    corresponding layers.
    """

    output_dim: int = 0
    force_fc_norm: bool = False

    transformer: TransformerCfg = field(default_factory=TransformerCfg)
    patch_embed: PatchEmbedCfg = field(default_factory=PatchEmbedCfg)
    embed: ViTEmbedCfg = field(default_factory=ViTEmbedCfg)

    mask_token: bool = False

    prompt_len: int = 1
    no_embed_class: bool = False

    svd_ratio: float = 1.0
    svd_type: str = ''

    def patch_embed_layer(self, embed_dim: int,
                          flatten: bool = True) -> PatchEmbed:
        """Return a PatchEmbed layer built from this config."""
        return PatchEmbed(cfg = self.patch_embed,
                          embed_dim = embed_dim, flatten = flatten)
    
    def embed_layer(self, embed_len: int, embed_dim: int) -> ViTEmbed:
        """Return a ViTEmbed layer built from this config."""
        return ViTEmbed(self.embed, embed_len, embed_dim)
    
    def head_layer(self, embed_dim: int):
        """Return a ViTHead layer built from this config."""
        return ViTHead(embed_dim, self.output_dim,
                       fc_norm = True if self.force_fc_norm else None)

class ViT(Transformer):
    """Vision Transformer (ViT) model.

    Extends :class:`Transformer` with a patch embedding front-end, optional
    learnable prompt tokens, and a classification / projection head.  The
    forward pass follows the standard ViT pipeline:
    image -> patch embed -> positional embed -> transformer blocks -> head.
    """

    def __init__(self, cfg: ViTCfg,
                 embed_len: int = 0,
                 norm_layer: type[nn.Module] = None):
        """Initialize ViT.

        Args:
            cfg: Full ViT configuration including transformer, patch embed, and
                positional embed sub-configs.
            embed_len: Number of additional positional slots to prepend (e.g.
                extra registers) before the patch tokens.
            norm_layer: Optional norm class to override the transformer default.
        """
        super().__init__(cfg = cfg.transformer,
                         norm_layer = norm_layer)
        self.patch_embed = cfg.patch_embed_layer(self.embed_dim)

        embed_len += self.patch_embed.num_patches
        if not cfg.no_embed_class: embed_len += cfg.prompt_len
        self.embed = cfg.embed_layer(embed_len, self.embed_dim)

        self.prompts = None
        if cfg.prompt_len > 0:
            p = torch.zeros(1, cfg.prompt_len, self.embed_dim)
            self.prompts = nn.Parameter(p)

        self.head = cfg.head_layer(self.embed_dim)
        self.out_dim = self.head.out_dim

        self.svd_ratio = cfg.svd_ratio
        self.svd_type = cfg.svd_type
    
    def cyns_config(self) -> dict:
        """Return image-level fields required for CreYon serialization.

        Delegates to :meth:`PatchEmbed.cyns_config` and returns::

            {'image_size': int | tuple[int, int], 'image_chan': int}

        These values describe the expected input shape of the model and are used
        when saving or reconstructing a :class:`~creyone_model.cynn.CreYonT` context.
        """
        return self.patch_embed.cyns_config()

    @staticmethod
    def solve_resolution(res: Union[int|str|list[int]|tuple[int]]) -> tuple[int]:
        """Normalize an image resolution specification to a ``(H, W)`` tuple.

        Accepted formats:
        - ``int`` -> ``(res, res)``
        - ``list`` / ``tuple`` -> cast to ``tuple``
        - ``str`` such as ``"224x224"`` or ``"(224x224)"`` -> parsed via regex
        """
        if isinstance(res, int): return (res, res)
        if isinstance(res, (tuple, list)): return tuple(res)
        res = re.match(r'\(?([0-9x]+)\)?', res).group(1).split('x')
        res = list(map(int, res))
        return tuple(res) if len(res) != 1 else tuple(res * 2)
    
    def transformer_forward(self, x: CreYonT) -> CreYonT:
        """Run the transformer backbone only, bypassing patch embedding and head."""
        return super().forward(x)
    
    def forward(self, x: CreYonT) -> CreYonT:
        """Full ViT forward pass: patch embed -> positional embed -> transformer -> head."""
        x = self.patch_embed(x) # .mask()
        x = self.transformer_forward(self.embed(x, self.prompts))
        return self.head(x)
    
    def unchanged_param(self, model: nn.Module, prefix: str = '') -> dict:
        """Return parameters that should remain frozen when loading from *model*.

        Subclasses override this to exclude specific weights from adaptation.
        The base ViT implementation excludes nothing.
        """
        return dict()
    
    def modify_param(self, stdt: dict, prefix: str = '') -> dict:
        """Return parameter overrides to apply on top of a loaded state dict.

        Subclasses override this to inject or transform specific weights before
        they are loaded into the model.  The base ViT implementation returns an
        empty dict (no modifications).
        """
        return dict()
    
    def reset_parameters(self, mode: str = 'trunc_') -> None:
        """Re-initialize all sub-module weights.

        Args:
            mode: Initialization scheme forwarded to sub-modules and the parent
                transformer (e.g. ``'trunc_'`` for truncated-normal).
        """
        self.embed.reset_parameters()
        if self.prompts is not None: nn.init.normal_(self.prompts, std=1e-6)
        self.head.reset_parameters(mode)
        super().reset_parameters(mode)
    
    def trainable_parameters(self, mode: str = 'all', is_remain: bool = False):
        """Configure which parameters require gradients.

        Args:
            mode: A ``'/'``-separated list of sub-module specifiers, or one of
                the shorthand strings ``'all'``, ``'none'``, or ``'adw'``.
                Each specifier may optionally carry a dotted sub-mode suffix
                (e.g. ``'transformer.attn'``).  Recognized prefixes are
                ``transformer``, ``head``, and ``prompts``; anything else is
                returned in the list if *is_remain* is ``True``, or raises
                :class:`KeyError` otherwise.
            is_remain: When ``True``, unknown specifiers are collected and
                returned instead of raising an error.

        Returns:
            A list of unrecognized specifier strings when *is_remain* is
            ``True``; otherwise the return value matches the parent
            :meth:`Transformer.trainable_parameters` contract.
        """
        if mode in ('adw', 'all', 'none'): return super().trainable_parameters(mode)
        def tkey(key, alt: str = 'all'): return alt
        subs = mode.split('/'); temp = list()
        self.requires_grad_(False)
        for sub in subs:
            if not sub.startswith('transformer'): temp.append(sub); continue
            super().trainable_parameters(tkey(*sub.split('.', 1)))
        subs = temp; temp = []
        for sub in subs:
            if sub.startswith('head'):
                self.head.trainable_parameters(tkey(*sub.split('.', 1)))
                continue
            if sub == 'prompts':
                self.prompts.requires_grad_(True)
                continue
            if not is_remain: raise KeyError(f"{sub} cannot be recognized")
            temp.append(sub)
        return temp