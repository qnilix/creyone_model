from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from creyone_layer.init import init_linear


class PosEmbed(nn.Module):
    """Learnable absolute positional embedding added directly to the input tensor."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Args:
            num_embeddings: Number of position slots (sequence length).
            embedding_dim: Feature dimension of each position vector.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        """Add positional embedding to x. Shape: (B, N, D) -> (B, N, D)."""
        return x + self.weight

    def trainable_parameters(self, mode: str = 'all'):
        """Enable ('all') or freeze ('') this module's parameters."""
        self.requires_grad_(mode == 'all')

    def reset_parameters(self, mode: str = 'trunc_', std: float=.02):
        """Re-initialize weights with truncated-normal or other init strategy."""
        self.apply(init_linear(mode=mode, std=std))


class PosEmbedV2(nn.Embedding):
    """Offset-aware positional embedding backed by nn.Embedding.

    Supports an explicit ``start`` offset so that prefix/prompt tokens can
    occupy the first few position slots and patch tokens begin from ``start``.
    """

    def __init__(self, *args, **kwargs):
        """Accepts the same arguments as nn.Embedding."""
        super().__init__(*args, **kwargs)
        self.register_buffer("pos_ids", torch.arange(self.num_embeddings).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor,
                positions: Optional[torch.Tensor] = None,
                start: int = 2, **_):
        """Add positional embedding to x.

        Args:
            x: Input tensor of shape (B, N, D).
            positions: Explicit position indices (B, N). Auto-generated when None.
            start: Index offset applied to auto-generated positions (default 2
                   reserves slots 0-1 for prefix tokens).

        Returns:
            Tensor of shape (B, N, D) with positional embeddings added.
        """
        if positions is None: positions = self.pos_ids + start
        return x + F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def reset_parameters(self, mode: str = 'trunc_', std: float=.02):
        """Re-initialize weights with truncated-normal or other init strategy."""
        self.apply(init_linear(mode=mode, std=std))

    def trainable_parameters(self, mode: str = 'all'):
        """Enable ('all') or freeze ('') this module's parameters."""
        self.requires_grad_(mode == 'all')