from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from timm.layers import trunc_normal_

from ..cynn import CreYonT
from ..utils import BaseCfg
from .pos import PosEmbed, PosEmbedV2


@dataclass
class ViTEmbedCfg(BaseCfg):
    """Configuration for ViTEmbed.

    Attributes:
        pos_version: Positional embedding variant — 1 for PosEmbed, 2 for PosEmbedV2.
        pos_apply: When False, positional embedding is skipped (replaced by Identity).
    """

    pos_version: int = 1
    pos_apply: bool = True


class ViTEmbed(nn.Module):
    """Positional embedding stage for Vision Transformer, with optional prompt prepending."""

    def __init__(self, cfg: ViTEmbedCfg, embed_len: int, embed_dim: int):
        """
        Args:
            cfg: Positional embedding configuration.
            embed_len: Number of position slots (sequence length after patch embed).
            embed_dim: Feature dimension of each token.
        """
        super().__init__()
        self.pos = nn.Identity()
        if cfg.pos_apply:
            self.pos = self.pos_class(cfg.pos_version)(embed_len, embed_dim)

    def pos_class(self, version: int = 1):
        """Return PosEmbed for version=1, PosEmbedV2 otherwise."""
        return PosEmbed if version == 1 else PosEmbedV2

    @staticmethod
    def reshape_embed(token: Optional[torch.Tensor] = None,
                      batch: int = 1) -> Optional[torch.Tensor]:
        """Expand a singleton token tensor to the current batch size."""
        if token is None: return None
        return token.expand(batch, -1, -1)

    def forward(self, x: CreYonT, prompt: torch.Tensor = None) -> CreYonT:
        """Apply positional embedding and optionally prepend prompt tokens.

        When ``prompt`` is provided, positional embedding is assigned so that
        prompt tokens always occupy the leading slots regardless of whether the
        position weights cover only patches or the full (prompt + patch) sequence.

        Args:
            x: Patch token sequence wrapped in CreYonT, shape (B, N, D).
            prompt: Optional prompt/prefix tokens of shape (1, P, D).

        Returns:
            Token sequence with positional embedding applied, shape (B, N[+P], D).
        """
        if prompt is None: return x(self.pos)
        prompt = x.__class__(prompt)(self.reshape_embed, batch=x.B)
        if self.pos.weight.shape[1] == x.shape[1]:
            return prompt.cat(x(self.pos), dim=1)
        return prompt.cat(x, dim=1)(self.pos)

    def trainable_parameters(self, mode: str = 'all'):
        """Enable ('all') or freeze ('') positional embedding parameters."""
        if not isinstance(self.pos, nn.Identity):
            self.pos.trainable_parameters(mode)

    def reset_parameters(self) -> None:
        """Re-initialize positional embedding weights."""
        if not isinstance(self.pos, nn.Identity): self.pos.reset_parameters()