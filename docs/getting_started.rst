Getting Started
===============

Installation
------------

Install the package from PyPI::

    pip install creyone_model

``creyone_model`` requires Python ≥ 3.10 and PyTorch ≥ 2.0.
The companion ``creyone_layer`` package (which supplies the registered layer
factories) is installed automatically as a dependency.

Basic usage
-----------

1. Create a config object
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~creyone_model.CNNBlockCfg` is a plain Python dataclass.  All
fields have sensible defaults (2-D tensors, standard convolution,
BatchNorm, ReLU), so you only need to specify what differs::

    from creyone_model import CNNBlockCfg

    cfg = CNNBlockCfg(
        tensor_dims=2,   # 2-D spatial (image) tensors
        act_name='silu', # swap ReLU → SiLU
        norm_name='batch',
    )

2. Build a ConvNormAct block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call :meth:`~creyone_model.CNNBlockCfg.block_module` to get a ready-to-use
``nn.Module``::

    block = cfg.block_module(in_dim=3, out_dim=64, kernel_size=3)
    # block: Conv2d(3→64) → BatchNorm2d(64) → SiLU()

    import torch
    x = torch.randn(1, 3, 224, 224)
    y = block(x)          # shape: (1, 64, 224, 224)

3. Override per-block layer options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Factory methods accept keyword overrides so individual blocks can deviate
from the shared config without spawning a whole new config object::

    # One block with LayerNorm instead of BatchNorm
    special_block = ConvNormAct(
        cfg, in_dim=64, out_dim=128,
        kernel_size=3, norm_name='layer',
    )

Building configs from kwargs
-----------------------------

For programmatic construction (e.g. from a YAML file), use the class-method
:meth:`~creyone_model.BaseCfg.instance`::

    cfg, remaining = CNNBlockCfg.instance(
        tensor_dims=2,
        act_name='gelu',
        norm_eps=1e-6,
        unknown_key='ignored',   # returned in `remaining`
    )

Nested :class:`~creyone_model.BaseCfg` fields are built recursively; any
kwargs not consumed by a level are passed down to sub-configs.
