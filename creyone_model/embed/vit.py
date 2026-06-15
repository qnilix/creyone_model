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

    pos_version: int = 1
    pos_apply: bool = True


class ViTEmbed(nn.Module):

    def __init__(self, cfg: ViTEmbedCfg, embed_len: int, embed_dim: int):
        super().__init__()
        self.pos = nn.Identity()
        if cfg.pos_apply:
            self.pos = self.pos_class(cfg.pos_version)(embed_len, embed_dim)
    
    def pos_class(self, version: int = 1):
        return PosEmbed if version == 1 else PosEmbedV2

    @staticmethod
    def reshape_embed(token: Optional[torch.Tensor] = None, 
                      batch: int = 1) -> Optional[torch.Tensor]:
        if token is None: return None
        return token.expand(batch, -1, -1)       

    def forward(self, x: CreYonT, prompt: torch.Tensor = None) -> CreYonT:
        if prompt is None: return x(self.pos)
        prompt = CreYonT(prompt)(self.reshape_embed, batch=x.B)
        if self.pos.weight.shape[1] == x.shape[1]:
            return prompt.cat(x(self.pos), dim=1)
        return prompt.cat(x, dim=1)(self.pos)
    
    def trainable_parameters(self, mode: str = 'all'):
        if not isinstance(self.pos, nn.Identity):
            self.pos.trainable_parameters(mode)
    
    def reset_parameters(self) -> None:
        if not isinstance(self.pos, nn.Identity): self.pos.reset_parameters()