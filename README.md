# creyone_model

PyTorch building blocks for Vision Transformer (ViT) and CNN models in the CREYONE framework.

## Overview

`creyone_model` provides a **config-first API** for assembling ViT and CNN architectures.
All hyper-parameters live in composable dataclass configs, and every module exposes a
unified `trainable_parameters` / `reset_parameters` interface for fine-tuning workflows.

The library is built around **`CreYonT`** — a thin tensor wrapper that lets you pipe
`nn.Module` calls with method chaining (`x(layer1)(layer2)`) while preserving
subclass-specific metadata (masks, position IDs, etc.) through every operation.

## Installation

```bash
pip install creyone_model
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0. Depends on `creyone_layer` and `timm`.

---

## Package structure

```text
creyone_model/
├── cynn/          # CreYonT tensor wrapper
├── cnn/           # CNNBlockCfg, ConvNormAct
├── vit/           # ViTCfg, ViT, pretrained configs, weight filters
├── transformer/   # TransformerCfg, Transformer, BlockCfg, Block
├── attention/     # AttnCfg, Attention, AttentionLoRA (LoRA)
├── mlp/           # MlpCfg, Mlp
├── embed/         # PatchEmbedCfg, PatchEmbed, ViTEmbedCfg, ViTEmbed
├── image/         # ImageClassification task wrapper
├── base/          # ModelCfg, ModuleBase, registry helpers
└── utils/         # BaseCfg, BuildShelf / BuildBook, registry
```

---

## Quick start

### CNN block

```python
from creyone_model import CNNBlockCfg

cfg = CNNBlockCfg(tensor_dims=2, act_name='silu')
block = cfg.block_module(in_dim=3, out_dim=64, kernel_size=3)

from creyone_model.cynn import CreYonT
import torch
x = CreYonT(torchT=torch.randn(1, 3, 224, 224))
y = block(x)   # CreYonT (1, 64, 224, 224)
```

### ViT encoder

```python
from creyone_model.vit import ViT, ViTCfg
from creyone_model.transformer import TransformerCfg
from creyone_model.cynn import CreYonT
import torch

cfg = ViTCfg(
    output_dim=512,
    transformer=TransformerCfg(depth=12, embed_dim=768),
)
model = ViT(cfg)

x = CreYonT(torchT=torch.randn(1, 3, 224, 224))
out = model(x)   # CreYonT (1, 512)
```

### Image classification via `create_model`

```python
from creyone_model.factory import create_model

model = create_model(
    'vit_base_16_224',
    task_name='image_classification',
    pretrained=True,
    num_classes=1000,
)
```

Model name format: `vit_<size>_<patch>_<imgsize>` (e.g. `vit_base_16_224`,
`vit_small_16_384`). To use a different input resolution when loading a pretrained
checkpoint: `vit_base_16_224(256)` or `vit_base_16_224(128x256)`.

### Fine-tuning with `trainable_parameters`

Every module follows the same interface:

| `mode`              | Effect                                                  |
|---------------------|---------------------------------------------------------|
| `'all'`             | Unfreeze everything                                     |
| `'none'`            | Freeze everything                                       |
| `'add'`             | Freeze backbone, keep adapter / LoRA weights trainable  |
| `'transformer'`     | Unfreeze only the transformer blocks                    |
| `'head'`            | Unfreeze only the classification head                   |
| `'transformer/head'`| Unfreeze transformer blocks and head                    |

```python
# Fine-tune the head only
model.body.trainable_parameters('head')

# Fine-tune with LoRA adapters in the transformer
model.body.trainable_parameters('add')
```

---

## Core concepts

### `CreYonT`

`CreYonT` wraps a `torch.Tensor` and intercepts PyTorch operations via
`__torch_function__`, so existing `nn.Module` code works without modification.
Its `__call__` enables functional chaining:

```python
from creyone_model.cynn import CreYonT

x = CreYonT(torchT=tensor)
y = x(linear)(norm)(act)    # equivalent to act(norm(linear(tensor)))
```

### Config-first factory pattern

All modules are built from `BaseCfg` dataclasses. Nested configs compose cleanly:

```python
from creyone_model.transformer import TransformerCfg
from creyone_model.transformer.block import BlockCfg
from creyone_model.attention import AttnCfg

cfg = TransformerCfg(
    depth=6,
    embed_dim=384,
    drop_path_rate=0.1,
    block=BlockCfg(
        init_values=1e-5,          # LayerScale
        attn=AttnCfg(
            num_heads=6,
            bias='qkvo',
            qk_norm='qk',
        ),
    ),
)
```
