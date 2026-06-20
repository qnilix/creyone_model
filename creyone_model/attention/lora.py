from torch import nn
from torch.nn import init

from dataclasses import dataclass

from .base import Attention, AttnCfg
from creyone_layer.linear import LoRALinear


@dataclass
class AttnLoRACfg(AttnCfg):

    lora_w: str = ''
    lora_r: int = 0


class AttentionLoRA(Attention):

    def _init_proj(self, dim: int, cfg: AttnCfg):
        def fn(s):
            if s in cfg.lora_w:
                return LoRALinear(dim, dim, lora_r = cfg.lora_r, bias = s in cfg.bias)
            return nn.Linear(dim, dim, bias = s in cfg.bias)
        for s in 'qkvo': self.add_module(f'{s}_proj', fn(s))
    
        