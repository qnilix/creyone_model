""" ViT configuration using LoRA attention blocks.

Copyright 2026 Rinka Kiriyama。
Licensed under the MIT License (MIT).
"""

from ..attention import AttnLoRACfg
from .base import ViTCfg


ViTLoRACfg = ViTCfg.derive('transformer.block.attn', AttnLoRACfg)
