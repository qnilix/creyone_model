"""Pre-norm transformer block with attention and MLP sub-layers.

Defines :class:`BlockCfg` (configuration dataclass) and :class:`Block` (``nn.Module``).
Each block applies the standard pre-norm residual pattern::

    x = x * alpha + LayerScale(DropPath(Attention(Norm(x))))
    x = x * alpha + LayerScale(DropPath(MLP(Norm(x))))

Optional features: LayerScale, stochastic depth (DropPath), sub-layer norm scaling,
and selective parameter freezing for fine-tuning / adapter insertion.

References:
    `Attention Is All You Need`
        - https://arxiv.org/abs/1706.03762
    `Going deeper with Image Transformers` (LayerScale)
        - https://arxiv.org/abs/2103.17239
    `Deep Networks with Stochastic Depth` (DropPath)
        - https://arxiv.org/abs/1603.09382
"""

import math
from typing import Optional
from dataclasses import dataclass, field

from torch import nn

from timm.layers import DropPath

from creyone_layer import layer_scale

from ...cynn import CreYonT
from ...utils import BaseCfg
from ...base.config import get_model_for_task
from ...attention import Attention, AttnCfg
from ...mlp import Mlp, MlpCfg


@dataclass
class BlockCfg(BaseCfg):
    """Configuration dataclass for a transformer block.

    Attributes:
        mlp_ratio: Expansion ratio for the MLP hidden dimension relative to ``dim``.
        act_name: Activation function name used inside the MLP (e.g. ``'gelu'``).
        mlp_norm: Whether to apply a normalization layer inside the MLP.
        sub_norm: Whether to apply sub-layer normalization scaling during weight reset.
        init_values: Initial value for LayerScale parameters; ``None`` disables LayerScale.
        attn: Configuration for the attention sub-layer.
        mlp: Configuration for the MLP sub-layer.
    """

    mlp_ratio: float = 4.

    act_name: str = 'gelu'
    mlp_norm: bool = False
    sub_norm: bool = False
    init_values: Optional[float] = None

    attn: AttnCfg = field(default_factory=AttnCfg)
    mlp: MlpCfg = field(default_factory=MlpCfg)

    def block_module(self, dim, **kwargs):
        """Instantiate a :class:`Block` with this configuration."""
        return Block(dim, cfg=self, **kwargs)

    def attn_layer(self, dim: int,
                   norm_layer: nn.Module = nn.LayerNorm,
                   layer_id: int = -1,
                   **kwargs) -> Attention:
        """Build and return an :class:`Attention` module for the given embedding dimension.

        Args:
            dim: Token embedding dimension.
            norm_layer: Normalization layer class passed to the attention module.
            layer_id: Index of this layer within the model; may be used for
                layer-specific behaviour (e.g. talking-heads adjustments).
            **kwargs: Additional keyword arguments forwarded to the attention factory.
        """
        create_fn, args, _ = get_model_for_task(self.attn.name, 'attn')
        return create_fn(*args, cfg=self.attn, **kwargs)(dim=dim, norm_layer=norm_layer, layer_id=layer_id)
    
    def mlp_layer(self, dim: int,
                  norm_layer: nn.Module = nn.LayerNorm,
                  **kwargs) -> Mlp:
        """Build and return an :class:`Mlp` module for the given embedding dimension.

        Args:
            dim: Token embedding dimension.
            norm_layer: Normalization layer class; only applied when ``mlp_norm`` is ``True``.
            **kwargs: Additional keyword arguments forwarded to the MLP factory.
        """
        create_fn, args, _ = get_model_for_task(self.mlp.name, 'mlp')
        nl = norm_layer if self.mlp_norm else nn.Identity
        return create_fn(*args, cfg=self.mlp, **kwargs)(dim=dim, norm_layer=nl)
    
    def layer_scale(self, dim: int) -> layer_scale.LayerScale:
        """Return a :class:`LayerScale` module, or ``nn.Identity`` when disabled.

        LayerScale is disabled when ``init_values`` is ``None``.
        """
        if self.init_values is None: return nn.Identity()
        return layer_scale.LayerScale(dim, init_values=self.init_values)
    
    def drop_path(self, path_drop: float = 0.) -> DropPath:
        """Return a :class:`DropPath` module, or ``nn.Identity`` when ``path_drop`` is 0."""
        if path_drop > 0.: return DropPath(path_drop)
        return nn.Identity()


class Block(nn.Module):
    """A single transformer block with pre-norm attention and MLP sub-layers.

    The residual connection follows the pre-norm pattern::

        x = x * alpha + LayerScale(DropPath(Attention(Norm(x))))
        x = x * alpha + LayerScale(DropPath(MLP(Norm(x))))

    Args:
        dim: Token embedding dimension.
        cfg: Block configuration produced by :class:`BlockCfg`.
        path_drop: Stochastic depth drop probability.
        norm_layer: Normalization layer class applied before each sub-layer.
        layer_id: Index of this block within the model stack.
    """

    def __init__(
            self,
            dim: int,
            cfg: BlockCfg,
            path_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            layer_id: int = -1
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim); self.ls1 = cfg.layer_scale(dim)
        self.norm2 = norm_layer(dim); self.ls2 = cfg.layer_scale(dim)

        self.attn = cfg.attn_layer(dim=dim, norm_layer=norm_layer, layer_id=layer_id)
        self.mlp  = cfg.mlp_layer(dim=dim, norm_layer=norm_layer)

        self.drop_path1 = cfg.drop_path(path_drop)
        self.drop_path2 = cfg.drop_path(path_drop)

        self.alpha = 1.0
        self.sub_norm = cfg.sub_norm

    def forward(self, x: CreYonT) -> CreYonT:
        """Apply attention and MLP sub-layers with residual connections."""
        x = x * self.alpha + self.attn(x(self.norm1))(self.ls1)(self.drop_path1)
        x = x * self.alpha + x(self.norm2)(self.mlp)(self.ls2)(self.drop_path2)
        return x
    
    def trainable_parameters(self, mode: str):
        """Set which parameters require gradients.

        Args:
            mode: One of ``'all'`` (unfreeze everything), ``'none'`` (freeze everything),
                or ``'add'`` (freeze the block but keep adapter/LoRA parameters in the
                attention and MLP sub-layers trainable).
        """
        if mode == 'all': self.requires_grad_(True); return
        self.requires_grad_(False)
        if mode == 'none': return 
        if mode == 'add':
            self.attn.trainable_parameters(mode='add')
            self.mlp.trainable_parameters(mode='add')
    
    def reset_parameters(self, mode: str = 'trunc_'):
        """Re-initialize weights of the attention and MLP sub-layers.

        When ``sub_norm`` is enabled, projection weights (``fc1``, ``fc2``,
        ``o_proj``, ``v_proj``) are additionally scaled by
        ``sqrt(log(depth * 2))`` to stabilize deep networks.

        Args:
            mode: Initialization scheme forwarded to the sub-layer ``reset_parameters``
                methods (e.g. ``'trunc_'`` for truncated-normal initialization).
        """
        self.attn.reset_parameters(mode)
        self.mlp.reset_parameters(mode)
        if not self.sub_norm: return 
        for name, p in self.named_parameters():
            if ("fc1" in name or "fc2" in name or "o_proj" in name or "v_proj" in name):
                p.data.mul_(math.sqrt(math.log(self.depth * 2)))