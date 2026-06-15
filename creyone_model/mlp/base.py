from typing import Optional, Union
from dataclasses import dataclass

from torch import nn

from creyone_layer.init import init_linear
from creyone_layer.utils import ntuple, factory

from ..utils.config import BaseCfg

@dataclass
class MlpCfg(BaseCfg):

    mlp_ratio: float = 4.
    
    mlp_act: str = 'gelu'
    act_inplace: bool = True

    mlp_bias: int = False
    mlp_drop: float = 0.


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks.

    Two-layer fully-connected network with the structure:
        fc1 -> act -> drop1 -> norm -> fc2 -> drop2
    """

    def __init__(self, cfg: MlpCfg, dim: int, 
                 output_dim: int = -1, hidden_dim: int = -1,
                 norm_layer=None, **kwargs):
        """Initialize Mlp.

        Args:
            dim: Input (and default output) feature dimension.
            norm_layer: Optional normalisation constructor applied between the
                two linear layers. Receives ``hidden_dim`` as its argument.
            **kwargs: Extra keyword arguments forwarded to ``_fc_layer`` and
                the activation factory.
        """
        super().__init__()
        dims = [dim, max(int(dim * cfg.mlp_ratio), hidden_dim), max(dim, output_dim)]
        drop_probs = ntuple(2)(cfg.mlp_drop)

        bias = cfg.mlp_bias
        if isinstance(bias, bool): bias = 3 if bias else 0

        self.fc1 = self._fc_layer(**kwargs)(*dims[:2], bias = bias % 2 == 1)
        self.fc2 = self._fc_layer(**kwargs)(*dims[1:], bias = bias > 0)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.act  = factory.create_layer(cfg.mlp_act, 'act')(inplace=cfg.act_inplace, **kwargs)()
        self.norm = norm_layer(dim[1]) if norm_layer is not None else nn.Identity()

    def _fc_layer(self, **kwargs) -> type[nn.Module]:
        """Return the linear layer class used to build ``fc1`` and ``fc2``.

        Subclasses can override this to substitute a custom linear layer
        (e.g. low-rank or quantised) without touching the rest of ``__init__``.
        """
        return nn.Linear

    def forward(self, x):
        """Run the forward pass.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Output tensor of shape ``(..., output_dim)``.
        """
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(self.norm(x)))
        return x

    def trainable_parameters(self, mode: str = 'all'):
        """Configure which parameters are trainable and return extra param groups.

        Args:
            mode: ``'all'`` enables gradients for the entire module.
                Any other value freezes the module.

        Returns:
            An empty list (no extra parameter groups are defined at this level).
        """
        if mode == 'all': self.requires_grad_(True); return []
        self.requires_grad_(False); return []

    def reset_parameters(self, mode: str = 'trunc_', std: float = .02):
        """Re-initialise all linear weights with ``init_linear``.

        Args:
            mode: Initialisation scheme passed to ``init_linear``
                (e.g. ``'trunc_'`` for truncated-normal).
            std: Standard deviation used by the initialiser.
        """
        fc_cls = self._fc_layer()
        if not issubclass(fc_cls, (nn.Linear, nn.Conv2d)):
            raise NotImplementedError(
                f"{type(self).__name__}.reset_parameters() does not support "
                f"fc layer type '{fc_cls.__name__}'. "
                f"Override reset_parameters() in the subclass."
            )
        self.apply(init_linear(mode=mode, std=std))
        if not isinstance(self.norm, nn.Identity):
            self.norm.reset_parameters()
