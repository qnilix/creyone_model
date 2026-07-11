""" Linear layer with Low-Rank Adaptation (LoRA)

Copyright 2026 Rinka Kiriyama。
Licensed under the MIT License (MIT). 
"""

from torch import nn
from dataclasses import dataclass

from .base import Attention, AttnCfg
from creyone_layer.linear import LoRALinear
from creyone_layer.init import init_linear


@dataclass
class AttnLoRACfg(AttnCfg):

    name: str = 'lora'

    lora_w: str = 'qk'
    lora_r: int = 8

    general_init: bool = True
    zeroB: bool = True


class AttentionLoRA(Attention):

    def _init_proj(self, dim: int, cfg: AttnLoRACfg):
        def fn(s):
            if s in cfg.lora_w:
                return LoRALinear(dim, dim, lora_r = cfg.lora_r, bias = s in cfg.bias)
            return nn.Linear(dim, dim, bias = s in cfg.bias)
        self.general_init = cfg.general_init
        self.zeroB = cfg.zeroB
        for s in 'qkvo': self.add_module(f'{s}_proj', fn(s))

    def reset_parameters(self, mode: str = 'trunc_', std: float = .02):
        i = getattr(self, 'general_init', True)
        z = getattr(self, 'zeroB', True)
        self.apply(init_linear(mode=mode, std=std, general_init=i, linear_init=True, zeroB=z))

    def trainable_parameters(self, mode='none'):
        if mode == 'all': self.requires_grad_(True); return []
        self.requires_grad_(False)
        if mode == 'none': return []
        assert mode == 'add'
        for pname, p in self.named_parameters():
            if pname.split('.')[-1].startswith('adw'):
                p.requires_grad_(True)
        return []
