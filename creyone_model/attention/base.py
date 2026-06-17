"""Base multi-head attention module for the CreYon model family."""

from dataclasses import dataclass
from typing import Iterator

from torch import nn
from creyone_layer.init import init_linear

from ..cynn import CreYonT
from ..utils.config import BaseCfg


@dataclass
class AttnCfg(BaseCfg):
    """Configuration for the :class:`Attention` module.

    Attributes:
        num_heads: Number of attention heads.
        out_dim: Output projection dimension. ``-1`` inherits the input ``dim``.
        softmax: If ``True``, apply softmax to attention weights; otherwise use
            ``attn_normalize``.
        bias: String containing the projection names (``'q'``, ``'k'``, ``'v'``,
            ``'o'``) whose linear layers should include a bias term.
        qk_norm: String containing the projection names (``'q'``, ``'k'``) whose
            per-head outputs should be normalized before computing attention scores.
        attn_norm: If ``True``, apply a normalization layer to the concatenated
            attention output before the output projection.
        attn_drop: Dropout probability applied to the attention weight matrix.
        proj_drop: Dropout probability applied after the output projection.
    """

    num_heads: int = 8
    out_dim: int = -1
    softmax: bool = True

    bias: str = 'qkvo'
    qk_norm: str = ''
    attn_norm: bool = False

    attn_drop: float = 0.
    proj_drop: float = 0.


class Attention(nn.Module):
    """Scaled dot-product multi-head attention.

    Supports optional per-head Q/K normalization, attention-output normalization,
    and separate dropout on attention weights and the output projection.

    Class Attributes:
        qk_scale: Shared QK scale override. When ``None`` each instance defaults
            to ``head_dim ** -0.5``.
    """

    qk_scale = None

    def __init__(self, cfg: AttnCfg, dim: int, layer_id: int = -1, **kwargs) -> None:
        """Initialize the attention module.

        Args:
            cfg: Attention configuration.
            dim: Input (and default output) channel dimension.  Must be divisible
                by ``cfg.num_heads``.
            layer_id: Optional index of this layer within a larger stack; stored
                for downstream use (e.g. logging or per-layer LR scaling).
            **kwargs: Forwarded to :meth:`_init_norm` (e.g. ``norm_layer``).
        """
        super().__init__()
        assert dim % cfg.num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads  = cfg.num_heads
        self.head_dim   = dim // cfg.num_heads
        self.qk_scale   = (self.qk_scale or self.head_dim ** -0.5)
        self.do_softmax = cfg.softmax
        self.layer_id   = layer_id

        self._init_proj(dim, cfg)
        self._init_norm(cfg, **kwargs)

        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.proj_drop = nn.Dropout(cfg.proj_drop)

    def _init_proj(self, dim: int, cfg: AttnCfg):
        """Register Q, K, V, and output linear projections as named sub-modules.

        Args:
            dim: Shared input dimension for all four projections.
            cfg: Attention configuration (used for ``bias`` and ``out_dim``).
        """
        def fn(s): return nn.Linear(dim, dim, bias = s in cfg.bias)
        for s in 'qkv': self.add_module(f'{s}_proj', fn(s))
        o_dim = cfg.out_dim if cfg.out_dim >= 0 else dim
        self.add_module(f'o_proj', nn.Linear(dim, o_dim, bias = 'o' in cfg.bias))

    def _init_norm(self, cfg: AttnCfg, norm_layer: nn.Module = nn.LayerNorm, **_):
        """Register per-head Q/K normalization layers and an optional attention-output norm.

        Args:
            cfg: Attention configuration (used for ``qk_norm`` and ``attn_norm``).
            norm_layer: Normalization class to instantiate; defaults to
                ``nn.LayerNorm``.
            **_: Ignored keyword arguments (allows clean ``**kwargs`` forwarding).
        """
        def fn(s): return norm_layer(self.head_dim) if s in cfg.qk_norm else nn.Identity()
        for s in 'qk': self.add_module(f'{s}_norm', fn(s))
        if not cfg.attn_norm: self.attn_norm = nn.Identity(); return
        self.attn_norm = norm_layer(self.head_dim * self.num_heads)

    def trainable_parameters(self, mode = 'none'):
        """Set which parameters require gradients and return any extra parameter groups.

        Args:
            mode: Gradient control mode.

                * ``'all'``  - unfreeze everything; returns ``[]``.
                * ``'none'`` - freeze everything; returns ``[]``.
                * ``'add'``  - freeze everything then unfreeze parameters whose
                  leaf name starts with ``'adw'``; returns ``[]``.

        Returns:
            An empty list (parameter groups are managed via ``requires_grad_``
            rather than returned groups).
        """
        if mode == 'all': self.requires_grad_(True); return []
        self.requires_grad_(False)
        if mode == 'none': return []
        assert mode == 'add'
        for k, v in self.named_parameters():
            if k.split('.')[-1].startswith('adw'): v.requires_grad_(True)
        return []

    def reset_parameters(self, mode: str = 'trunc_', std: float=.02):
        """Re-initialize all linear weights using the specified scheme.

        Args:
            mode: Initialization mode passed to :func:`creyone_layer.init.init_linear`
                (e.g. ``'trunc_'`` for truncated-normal).
            std: Standard deviation used by the initializer.
        """
        self.apply(init_linear(mode=mode, std=std))

    def forward(self, x: CreYonT, query: CreYonT = None, key: CreYonT = None) -> CreYonT:
        """Compute multi-head attention.

        Args:
            x: Value source tensor of shape ``(B, L, D)``.
            query: Query source tensor; defaults to ``x`` when ``None``.
            key: Key source tensor; defaults to ``x`` when ``None``.

        Returns:
            Output tensor of shape ``(B, L, out_dim)`` after the output
            projection and dropout.
        """
        q, k, v = self.qkv_proj(x, (query or x), (key or x))

        x = self.fwd_attn(q(self.q_norm), k(self.k_norm), v)
        x = x.rearrange('(b h) l d -> b l (h d)', h=self.num_heads)
        return x(self.attn_norm)(self.o_proj)(self.proj_drop)

    def fwd_attn(self, q: CreYonT, k: CreYonT, v: CreYonT) -> CreYonT:
        """Compute scaled dot-product attention for a single head batch.

        Args:
            q: Query tensor of shape ``(B*H, L_q, D_h)``.
            k: Key tensor of shape ``(B*H, L_k, D_h)``.
            v: Value tensor of shape ``(B*H, L_v, D_h)``; ``L_v`` must equal
               ``L_k``.

        Returns:
            Context tensor of shape ``(B*H, L_q, D_h)`` after attention dropout.
        """
        attn = (q * self.qk_scale).bmm(k.transpose(-2, -1))
        attn = attn.softmax(dim=-1) if self.do_softmax else attn.attn_normalize()
        return attn(self.attn_drop).bmm(v)

    def qkv_proj(self, x: CreYonT, q: CreYonT, k: CreYonT) -> Iterator:
        """Project inputs to Q, K, V and split into per-head tensors.

        Args:
            x: Value source tensor of shape ``(B, L, D)``.
            q: Query source tensor of shape ``(B, L_q, D)``.
            k: Key source tensor of shape ``(B, L_k, D)``.

        Returns:
            An iterator of three tensors — Q, K, V — each reshaped to
            ``(B*H, L, D_h)``.
        """
        t = [q(self.q_proj), k(self.k_proj), x(self.v_proj)]
        s = 'b l (h d) -> (b h) l d'
        return map(lambda x: x.rearrange(s, h=self.num_heads), t)
