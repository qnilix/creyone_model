"""
CreYonT — a tensor wrapper for intercepting nn.Module processing without modifying the module.

Design rationale
----------------
1. Inject behavior into nn.Module processing without touching the module itself.
   The __torch_function__ protocol lets CreYonT intercept every torch operation
   inside an existing module's forward pass, so inputs and outputs are transparently
   replaced with CreYonT instances — no changes to the module code required.

2. Carry supplementary information (e.g. masks) through operations alongside the tensor.
   plug() duplicates the wrapper via shallow copy, so subclass attributes such as
   mask or position_ids survive each operation and are passed on to downstream layers.
   Subclasses only need to define the extra attributes; the rest of the pipeline is
   unchanged.

Copyright 2026 Rinka Kiriyama。
Licensed under the MIT License (MIT). 
"""
import copy
from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def _matmul(a, b): return a @ b


class CreYonT:
    """
    A thin wrapper around torch.Tensor.

    Delegates torch operations transparently while preserving subclass-specific
    supplementary attributes (mask, metadata, etc.) across every operation via
    the shallow copy performed in plug().

    When a CreYonT is passed to an nn.Module, __torch_function__ intercepts each
    torch operation inside the module's forward and returns the result as a CreYonT.
    This allows pre/post-processing and attribute propagation at the wrapper level
    without any changes to the module.

    Subclassing example::

        class MaskedT(CreYonT):
            mask: torch.Tensor | None = None

            def with_mask(self, mask: torch.Tensor) -> "MaskedT":
                obj = self.plug(self._t)  # shallow copy carries the mask forward
                obj.mask = mask
                return obj
    """

    def __init__(self, *args, torchT: torch.Tensor = None, init_val: float = None, **kwargs):
        self._t = torchT if torchT is not None else torch.Tensor(*args, **kwargs)
        if init_val is not None: self._t = self._t * 0 + init_val

    def __call__(self, func: callable, *args, **kwargs) -> Self:
        return self.plug(func(self._t, *args, **kwargs))

    # --- arithmetic ---
    def __add__(self, other) -> Self: return self.calc(other, lambda a, b: a + b)
    def __radd__(self, other) -> Self: return self.__add__(other)
    def __sub__(self, other) -> Self: return self.calc(other, lambda a, b: a - b)
    def __rsub__(self, other) -> Self:
        if isinstance(other, CreYonT): other = other.tensor()
        return self.plug(other - self._t)
    def __mul__(self, other) -> Self: return self.calc(other, lambda a, b: a * b)
    def __rmul__(self, other) -> Self: return self.__mul__(other)
    def __truediv__(self, other) -> Self: return self.calc(other, lambda a, b: a / b)
    def __matmul__(self, other) -> Self: return self.calc(other, _matmul)
    def __neg__(self) -> Self: return self.plug(-self._t)
    def __pow__(self, other) -> Self: return self.calc(other, lambda a, b: a ** b)

    # --- comparison ---
    def __eq__(self, other) -> Self: return self.calc(other, lambda a, b: a == b)
    def __ne__(self, other) -> Self: return self.calc(other, lambda a, b: a != b)
    def __lt__(self, other) -> Self: return self.calc(other, lambda a, b: a < b)
    def __le__(self, other) -> Self: return self.calc(other, lambda a, b: a <= b)
    def __gt__(self, other) -> Self: return self.calc(other, lambda a, b: a > b)
    def __ge__(self, other) -> Self: return self.calc(other, lambda a, b: a >= b)

    # --- indexing ---
    def __getitem__(self, idx) -> Self: return self.plug(self._t[idx])
    def __setitem__(self, idx, val):
        if isinstance(val, CreYonT): val = val.tensor()
        self._t[idx] = val

    def __repr__(self): return f"CreYonT({self._t})"

    # --- torch function protocol ---
    @classmethod
    def __torch_function__(cls, func, _types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        def unwrap(a):
            if isinstance(a, cls): return a._t
            if isinstance(a, (list, tuple)): return type(a)(unwrap(x) for x in a)
            return a
        new_args = tuple(unwrap(a) for a in args)
        new_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        result = func(*new_args, **new_kwargs)
        if isinstance(result, torch.Tensor):
            first = next((a for a in args if isinstance(a, cls)), None)
            return first.plug(result) if first else cls(torchT=result)
        return result

    # --- properties ---
    @property
    def shape(self): return self._t.shape

    @property
    def B(self): return self._t.shape[0]

    @property
    def grad(self): return self._t.grad

    @property
    def dtype(self): return self._t.dtype

    @property
    def device(self): return self._t.device

    # --- core ---
    def tensor(self) -> torch.Tensor: return self._t

    def calc(self, other, func, **kwargs):
        if isinstance(other, CreYonT): other = other.tensor()
        return self.plug(func(self._t, other, **kwargs))

    def plug(self, x: torch.Tensor) -> Self:
        # shallow copy carries subclass attributes (mask, etc.) into the new instance
        result = copy.copy(self); result._t = x; return result

    # --- autograd ---
    def backward(self, **kwargs) -> Self: self._t.backward(**kwargs); return self
    def detach(self) -> Self: return self.plug(self._t.detach())
    def requires_grad_(self, requires_grad: bool = True) -> Self:
        self._t.requires_grad_(requires_grad); return self

    # --- device / dtype ---
    def to(self, *args, **kwargs) -> Self: return self.plug(self._t.to(*args, **kwargs))
    def cuda(self, **kwargs) -> Self: return self.plug(self._t.cuda(**kwargs))
    def cpu(self) -> Self: return self.plug(self._t.cpu())
    def float(self) -> Self: return self.plug(self._t.float())
    def half(self) -> Self: return self.plug(self._t.half())
    def double(self) -> Self: return self.plug(self._t.double())

    # --- tensor ops ---
    def chunk(self, chunks: int, dim: int = 0) -> tuple[Self, ...]:
        return tuple(self.plug(t) for t in self._t.chunk(chunks, dim=dim))

    def contiguous(self) -> Self: return self.plug(self._t.contiguous())

    def exp(self) -> Self: return self.plug(self._t.exp())

    def flatten(self, *args, **kwargs) -> Self:
        return self.plug(self._t.flatten(*args, **kwargs))

    def flip(self, *args, **kwargs) -> Self:
        return self.plug(self._t.flip(*args, **kwargs))

    def log(self) -> Self: return self.plug(torch.log(self._t))

    def mean(self, dim=None, keepdim: bool = False, *, dtype: torch.dtype = None) -> Self:
        return self.plug(self._t.mean(dim, keepdim=keepdim, dtype=dtype))

    def reshape(self, *args, **kwargs) -> Self:
        return self.plug(self._t.reshape(*args, **kwargs))

    def softmax(self, dim: int, dtype: torch.dtype = None) -> Self:
        return self.plug(self._t.softmax(dim, dtype=dtype))

    def init(self, name: str, **kwargs) -> Self:
        if name == 'trunc_normal': nn.init.trunc_normal_(self._t, **kwargs)
        return self

    def transpose(self, dim0: int, dim1: int) -> Self:
        return self.plug(self._t.transpose(dim0, dim1))
    
    def permute(self, *args, **kwargs) -> Self:
        return self.plug(self._t.permute(*args, **kwargs))

    def bmm(self, other) -> Self: return self.calc(other, torch.bmm)

    def cat(self, other, dim: int = 0) -> Self:
        if isinstance(other, CreYonT): other = other.tensor()
        return self.plug(torch.cat((self._t, other), dim=dim))

    def linear(self, other, bias: torch.Tensor = None) -> Self:
        return self.calc(other, F.linear, bias=bias)

    def split(self, split_size_or_sections: int | list[int], dim: int = 0) -> tuple[Self, ...]:
        return tuple(self.plug(t) for t in torch.split(self._t, split_size_or_sections, dim=dim))

    def normalize(self, p: float = 2, dim: int = 1, eps: float = 1e-12) -> Self:
        return self.plug(F.normalize(self._t, p=p, dim=dim, eps=eps))

    def rearrange(self, pattern: str, **axes_lengths: any) -> Self:
        return self.plug(rearrange(self._t, pattern, **axes_lengths))
    
    def where(self, condition, other) -> Self:
        if isinstance(condition, CreYonT): condition = condition._t
        if isinstance(other, CreYonT): other = other._t
        return self.plug(torch.where(condition, self._t, other))


