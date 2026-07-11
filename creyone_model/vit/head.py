from typing import Optional

import torch
import torch.nn as nn
from creyone_layer.init import init_linear

from ..cynn import CreYonT


class ViTHead(nn.Module):

    def __init__(self,
                 dim: int,
                 out_dim: int = 0,
                 global_pool: str = 'token',
                 fc_norm: Optional[bool] = None,
                 norm_layer: nn.Module = nn.LayerNorm,
                 drop_rate: float = .0):
        super().__init__()

        self.global_pool = global_pool

        use_fc_norm = (fc_norm or global_pool in ('avg', 'avgmax', 'max'))
        self.fc_norm = norm_layer(dim) if use_fc_norm else nn.Identity()

        self.head_drop = nn.Dropout(drop_rate)
        self.linear = nn.Linear(dim, out_dim) if out_dim > 0 else nn.Identity()

        self.out_dim = dim if out_dim == 0 else out_dim

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        pool_type = (pool_type or self.global_pool)
        if pool_type == 'token': return x[:, 0]
        return x

    def forward(self, x: CreYonT, pre_logits: bool = False) -> CreYonT:
        x = x(self.pool)(self.fc_norm)(self.head_drop)
        return x if pre_logits else x(self.linear)
    
    def reset_parameters(self, mode: str = 'trunc_', std: float = .02):
        self.apply(init_linear(mode=mode, std=std))
    
    def trainable_parameters(self, mode: str = 'all'):
        if mode == 'all': return self.requires_grad_(True)
        return self.requires_grad_(False)