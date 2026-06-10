from dataclasses import dataclass

import torch
from torch import nn, Tensor

from creyone_layer import create_layer
from ..utils import BaseCfg
from ..cynn import CreYonT


@dataclass
class CNNBlockCfg(BaseCfg):
    """Configuration dataclass for CNN building blocks.

    Centralises all hyper-parameters needed to construct a
    :class:`ConvNormAct` block: convolution, normalisation, activation, and
    pooling layers are each described by a *name* (matched by
    ``creyone_layer.create_layer``) and a set of scalar options.

    Attributes:
        tensor_dims: Spatial dimensionality of the input tensor (e.g. ``2``
            for images, ``1`` for sequences, ``3`` for volumetric data).
        conv_option: Option string forwarded to the convolution factory
            (see creyone-layer).
        conv_name: Registered name of the convolution layer (default
            ``'base'`` -> standard convolution).
        conv_bias: Whether the convolution includes a learnable bias term.
            Typically ``False`` when batch normalisation follows.
        act_name: Registered name of the activation function
            (default ``'relu'``).
        act_inplace: Whether the activation operates in-place.
        norm_name: Registered name of the normalisation layer
            (default ``'batch'`` -> BatchNorm).
        norm_eps: Epsilon added to the denominator for numerical stability.
        norm_mom: Momentum used to update the running statistics.
        pool_name: Registered name of the pooling layer (default ``'max'``).
        pool_option: Option string forwarded to the pooling factory.

    Example::

        cfg = CNNBlockCfg(tensor_dims=2, act_name='silu')
        block = cfg.block_module(in_dim=32, out_dim=64, kernel_size=3)
    """

    tensor_dims: int = 2
    conv_option: str = 'ap'

    conv_name: str = 'base'
    conv_bias: bool = False

    act_name: str = 'relu'
    act_inplace: bool = False

    norm_name: str = 'batch'
    norm_eps: float = 1e-5
    norm_mom: float = 0.1

    pool_name: str = 'max'
    pool_option: str = 'ap'

    def block_module(self, in_dim: int, out_dim: int, *args, **kwargs) -> 'ConvNormAct':
        """Instantiate a :class:`ConvNormAct` block from this config.

        Args:
            in_dim: Number of input channels / features.
            out_dim: Number of output channels / features.
            *args: Positional arguments forwarded to :class:`ConvNormAct`.
            **kwargs: Keyword arguments forwarded to :class:`ConvNormAct`.

        Returns:
            A configured :class:`ConvNormAct` module.
        """
        return ConvNormAct(self, in_dim, out_dim, *args, **kwargs)

    def act_layer(self, name: str = None, inplace: bool = None, **kwargs) -> nn.Module:
        """Build and return an activation layer.

        Args:
            name: Override ``act_name``; uses the config value when ``None``.
            inplace: Override ``act_inplace``; uses the config value when
                ``None``.
            **kwargs: Extra keyword arguments passed to the layer constructor.

        Returns:
            An instantiated activation ``nn.Module``.
        """
        actl = create_layer(name or self.act_name, 'act')
        return actl(inplace=inplace if inplace is not None else self.act_inplace)(**kwargs)

    def conv_layer(self, *args, name: str = None, bias: bool = None,
                   c_opt: str = None, **kwargs) -> nn.Module:
        """Build and return a convolution layer.

        Args:
            *args: Positional arguments forwarded to the layer constructor
                (typically ``in_channels`` and ``out_channels``).
            name: Override ``conv_name``; uses the config value when ``None``.
            bias: Override ``conv_bias``; uses the config value when ``None``.
            c_opt: Override ``conv_option``; uses the config value when
                ``None``.
            **kwargs: Extra keyword arguments (e.g. ``kernel_size``,
                ``stride``) forwarded to the layer constructor.

        Returns:
            An instantiated convolution ``nn.Module``.
        """
        conv = create_layer(name or self.conv_name, 'conv')
        conv = conv(dim = self.tensor_dims, optional = c_opt or self.conv_option)
        return conv(*args, bias = bias if bias is not None else self.conv_bias, **kwargs)

    def norm_layer(self, dim: int, name: str = None, eps: float = None) -> nn.Module:
        """Build and return a normalisation layer.

        Args:
            dim: Number of features / channels to normalise.
            name: Override ``norm_name``; uses the config value when ``None``.
            eps: Override ``norm_eps``; uses the config value when ``None``.

        Returns:
            An instantiated normalisation ``nn.Module``.
        """
        norm = create_layer(name or self.norm_name, 'norm')
        return norm(dim = self.tensor_dims, eps=eps if eps is not None else self.norm_eps, mom=self.norm_mom)(dim)

    def pool_layer(self, *args, name: str = None, p_opt: str = None, **kwargs) -> nn.Module:
        """Build and return a pooling layer.

        Args:
            *args: Positional arguments forwarded to the layer constructor
                (e.g. ``kernel_size``).
            name: Override ``pool_name``; uses the config value when ``None``.
            p_opt: Override ``pool_option``; uses the config value when
                ``None``.
            **kwargs: Extra keyword arguments forwarded to the layer
                constructor.

        Returns:
            An instantiated pooling ``nn.Module``.
        """
        pool = create_layer(name or self.pool_name, 'pool')
        pool = pool(dim = self.tensor_dims, optional = p_opt or self.pool_option)
        return pool(*args, **kwargs)


class ConvNormAct(nn.Module):
    """Sequential Convolution → Normalisation → Activation block.

    A standard building block used throughout modern CNN architectures.
    All layer details are driven by a :class:`CNNBlockCfg` instance, which
    allows swapping convolution type, normalisation, and activation without
    changing the block's code.

    Attributes:
        stride: Stride used by the internal convolution (mirrors ``s``).
        out_dim: Number of output channels / features.
        conv: The convolution layer.
        norm: The normalisation layer.
        act: The activation layer.

    Example::

        cfg = CNNBlockCfg()
        block = ConvNormAct(cfg, in_dim=3, out_dim=64, kernel_size=3)
        y = block(x)  # x: (B, 3, H, W) -> y: (B, 64, H, W)
    """

    def __init__(self, cfg: CNNBlockCfg, in_dim: int, out_dim: int, *args, s=1,
                 norm_name: str = None, act_name: str = None,
                 inplace: bool = None, eps: float = None, **kwargs):
        """Initialise the ConvNormAct block.

        Args:
            cfg: Configuration object that supplies all layer factories.
            in_dim: Number of input channels / features.
            out_dim: Number of output channels / features.
            *args: Extra positional arguments forwarded to the convolution
                layer (e.g. ``kernel_size``).
            s: Stride for the convolution. Defaults to ``1``.
            norm_name: Override the normalisation layer name from ``cfg``.
            act_name: Override the activation layer name from ``cfg``.
            inplace: Override the in-place flag from ``cfg``.
            eps: Override the normalisation epsilon from ``cfg``.
            **kwargs: Additional keyword arguments forwarded to the
                convolution layer.
        """
        super().__init__()
        self.stride = s; self.out_dim = out_dim
        self.conv = cfg.conv_layer(in_dim, out_dim, *args, s=s, **kwargs)
        self.norm = cfg.norm_layer(out_dim, name=norm_name, eps=eps)
        self.act  = cfg.act_layer(name=act_name, inplace=inplace)

    def forward(self, x: CreYonT) -> CreYonT:
        """Apply conv -> norm -> act to the input tensor.

        Args:
            x: Input tensor of shape ``(B, in_dim, *spatial)``.

        Returns:
            Output tensor of shape ``(B, out_dim, *spatial')``, where
            ``spatial'`` depends on the convolution stride and padding.
        """
        return x(self.conv)(self.norm)(self.act)