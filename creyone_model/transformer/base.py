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

    embed_dim: int = 768

    norm_eps: float = 1e-6
    pre_norm: bool = False
    post_norm: bool = False

    drop_path_rate: float = .0

    block: BlockCfg = field(default_factory=BlockCfg)

    def block_kwargs(self, idx: int) -> dict:
        return {
            'path_drop': idx * self.drop_path_rate / self.block.depth,
            'layer_id': idx
        }


class Transformer(nn.Module):

    def __init__(self, 
                 cfg: TransformerCfg,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()

        self.embed_dim = cfg.embed_dim
        self.block_depth = cfg.block.depth

        if norm_layer is None:
            norm_layer = create_layer('layer', 'norm')(eps=cfg.norm_eps)

        self.norm_pre = norm_layer(self.embed_dim) if cfg.pre_norm else nn.Identity()
        self.norm = norm_layer(self.embed_dim) if cfg.post_norm else nn.Identity()

        for i in range(cfg.block.depth):
            block = cfg.block.block_module(self.embed_dim, norm_layer=norm_layer, **cfg.block_kwargs(i))
            self.add_module(f'block{i}', block)
    
    def forward(self, x: CreYonT) -> CreYonT:
        x = x(self.norm_pre)
        for i in range(self.block_depth):
            x = self.get_submodule(f'block{i}')(x)
        return x(self.norm)
    
    def trainable_parameters(self, mode: str = ''):
        if mode == 'all': self.requires_grad_(True); return []
        self.requires_grad_(False)
        if mode == 'none': return []
        for i in range(self.block_depth):
            self.get_submodule(f'block{i}').trainable_parameters(mode)
    
    def reset_parameters(self, mode: str = 'trunc_'):
        for i in range(self.block_depth):
            self.get_submodule(f'block{i}').reset_parameters(mode)
        
    def attention_ops(self):
        def f(x: torch.Tensor, **_): return x.softmax(dim=-1)
        return [f]
