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
        return PatchEmbed(cfg = self.patch_embed, 
                          embed_dim = embed_dim, flatten = flatten)
    
    def embed_layer(self, embed_len: int, embed_dim: int) -> ViTEmbed:
        return ViTEmbed(self.embed, embed_len, embed_dim)
    
    def head_layer(self, embed_dim: int):
        return ViTHead(embed_dim, self.output_dim, 
                       fc_norm = True if self.force_fc_norm else None)

class ViT(Transformer):

    def __init__(self, cfg: ViTCfg, 
                 embed_len: int = 0,
                 norm_layer: type[nn.Module] = None):
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
        return self.patch_embed.cyns_config()

    @staticmethod
    def solve_resolution(res: Union[int|str|list[int]|tuple[int]]) -> tuple[int]:
        if isinstance(res, int): return (res, res)
        if isinstance(res, (tuple, list)): return tuple(res)
        res = re.match(r'\(?([0-9x]+)\)?', res).group(1).split('x')
        res = list(map(int, res))
        return tuple(res) if len(res) != 1 else tuple(res * 2)

    #if mask is not None:
    #    assert self.mask_token is not None
    #    t = self.embed.reshape_embed(self.mask_token, batch=x.shape[0])
    #    w = mask.unsqueeze(-1).type_as(t)
    #   x = x * (1 - w) + t * w
    
    def transformer_forward(self, x: CreYonT) -> CreYonT:
        return super().forward(x)
    
    def attn_norm(self, x: CreYonT):
        return x.softmax(dim=-1)
    
    def forward(self, x: CreYonT) -> CreYonT:
        x.attn_norm = self.attn_norm
        x = self.patch_embed(x) # .mask()
        x = self.transformer_forward(self.embed(x, self.prompts))
        return self.head(x)
    
    def unchanged_param(self, model: nn.Module, prefix: str = '') -> dict:
        return dict()
    
    def modify_param(self, stdt: dict, prefix: str = '') -> dict:
        return dict()
    
    def reset_parameters(self, mode: str = 'trunc_') -> None:
        self.embed.reset_parameters()
        if self.prompts is not None: nn.init.normal_(self.prompts, std=1e-6)
        self.head.reset_parameters(mode)
        super().reset_parameters(mode)
    
    def trainable_parameters(self, mode: str = 'all', is_remain: bool = False):
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