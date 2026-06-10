creyone_model
=============

**creyone_model** provides PyTorch layer-building utilities for the CREYONE
framework. It ships a lightweight, dataclass-driven configuration system
that lets you assemble convolutional building blocks — convolution,
normalisation, and activation — without touching boilerplate code.

Key features
------------

* **Config-first API** – every block is fully described by a
  :class:`~creyone_model.CNNBlockCfg` dataclass.  Swap layer types (e.g.
  ``BatchNorm`` → ``LayerNorm``) by changing a single field.
* **Hierarchical config composition** – :class:`~creyone_model.BaseCfg`
  supports nested configs; ``BaseCfg.instance()`` builds the whole tree
  from a flat ``**kwargs`` dict and returns any unconsumed keys.
* **Dimension-agnostic** – set ``tensor_dims=1/2/3`` to target 1-D
  sequences, 2-D images, or 3-D volumes with the same config object.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
