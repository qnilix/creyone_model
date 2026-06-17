import math
from typing import Optional
from dataclasses import dataclass

from torch import nn

from timm.layers import DropPath

from creyone_layer import layer_scale

from ...cynn import CreYonT
from ...utils import BaseCfg
from ...base.config import get_model_for_task
from ...attention import Attention
from ...mlp import Mlp


@dataclass
class BlockCfg(BaseCfg):

    mlp_ratio: float = 4.

    act_name: str = 'gelu'
    attn_name: str = 'base'
    mlp_name: str = 'base'
    
    mlp_norm: bool = False
    sub_norm: bool = False
    init_values: Optional[float] = None

    def block_module(self, dim, **kwargs):
        return Block(dim, cfg=self, **kwargs)

    def attn_layer(self, dim: int,
                   norm_layer: nn.Module = nn.LayerNorm,
                   layer_id: int = -1, 
                   **kwargs) -> Attention:
        create_fn, args, _ = get_model_for_task(self.attn_name, 'attn')
        return create_fn(*args, **kwargs)(dim, norm_layer=norm_layer, layer_id=layer_id)
    
    def mlp_layer(self, dim: int,
                  norm_layer: nn.Module = nn.LayerNorm,
                  **kwargs) -> Mlp:
        create_fn, args, _ = get_model_for_task(self.attn_name, 'mlp')
        nl = norm_layer if self.mlp_norm else nn.Identity()
        return create_fn(*args, **kwargs)(dim, norm_layer=nl)
    
    def layer_scale(self, dim: int) -> layer_scale.LayerScale:
        if self.init_values is None: return nn.Identity()
        return layer_scale.LayerScale(dim, init_values=self.init_values)
    
    def drop_path(self, path_drop: float = 0.) -> DropPath:
        if path_drop > 0.: return DropPath(path_drop)
        return nn.Identity()


class Block(nn.Module):

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

        self.attn = cfg.attn_layer(dim, norm_layer=norm_layer, layer_id=layer_id)
        self.mlp  = cfg.mlp_layer(dim, norm_layer=norm_layer)

        self.drop_path1 = cfg.drop_path(path_drop)
        self.drop_path2 = cfg.drop_path(path_drop)

        self.alpha = 1.0
        self.sub_norm = cfg.sub_norm

    def forward(self, x: CreYonT) -> CreYonT:
        x = x * self.alpha + self.attn(x(self.norm1))(self.ls1)(self.drop_path1)
        x = x * self.alpha + x(self.norm2)(self.mlp)(self.ls2)(self.drop_path2)
        return x
    
    def trainable_parameters(self, mode: str):
        if mode == 'all': self.requires_grad_(True); return
        self.requires_grad_(False)
        if mode == 'none': return 
        if mode == 'add':
            self.attn.trainable_parameters(mode='add')
            self.mlp.trainable_parameters(mode='add')
    
    def reset_parameters(self, mode: str = 'trunc_'):
        self.attn.reset_parameters(mode)
        self.mlp.reset_parameters(mode)
        if not self.sub_norm: return 
        for name, p in self.named_parameters():
            if ("fc1" in name or "fc2" in name or "o_proj" in name or "v_proj" in name):
                p.data.mul_(math.sqrt(math.log(self.depth * 2)))