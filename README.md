# creyone_model

PyTorch layer-building utilities for the CREYONE framework.

## Overview

`creyone_model` provides a config-first API for assembling CNN building blocks.
All layer hyper-parameters live in a single `CNNBlockCfg` dataclass, making it
easy to swap layer types (convolution, normalisation, activation, pooling)
without changing model code.

## Installation

```bash
pip install creyone_model
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

## Quick start

```python
from creyone_model import CNNBlockCfg

cfg = CNNBlockCfg(tensor_dims=2, act_name='silu')
block = cfg.block_module(in_dim=3, out_dim=64, kernel_size=3)

import torch
y = block(torch.randn(1, 3, 224, 224))  # (1, 64, 224, 224)
```

## Documentation

Full documentation (API reference + getting started guide) is hosted at
<https://creyone-model.readthedocs.io/>.
