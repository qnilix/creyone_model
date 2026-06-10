import copy
from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def _matmul(a, b): return a @ b


class CreYonT:

    def __init__(self, *args, torchT: torch.Tensor = None, init_val: float = None, **kwargs):
        self._t = torchT if torchT is not None else torch.Tensor(*args, **kwargs)
        if init_val is not None: self._t = self._t * 0 + init_val
    
    def __call__(self, func: callable, *args, **kwargs) -> Self:
        return self.plug(func(self._t, *args, **kwargs))
    
    def __add__(self, other) -> Self: return self.calc(other, lambda a, b: a + b)

    def __mul__(self, other) -> Self: return self.calc(other, lambda a, b: a * b)

    def __truediv__(self, other) -> Self: return self.calc(other, lambda a, b: a / b)

    def __matmul__(self, other) -> Self: return self.calc(other, _matmul)
    
    @property
    def shape(self): return self._t.shape

    @property
    def B(self): return self._t.shape[0]
    
    def tensor(self) -> torch.Tensor: return self._t
    
    def calc(self, other, func, **kwargs):
        if isinstance(other, CreYonT): other = other.tensor()
        return self.plug(func(self._t, other, **kwargs))

    def plug(self, x: torch.Tensor) -> Self:
        result = copy.copy(self); result._t = x; return result

    def chunk(self, chunks: int, dim: int = 0) -> tuple[Self, ...]:
        return tuple(CreYonT(torchT=t) for t in self._t.chunk(chunks, dim=dim))
    
    def contiguous(self) -> Self: return self.plug(self._t.contiguous())

    def exp(self) -> Self: return self.plug(self._t.exp())

    def flatten(self, *args, **kwargs) -> Self:
        return self.plug(self._t.flatten(*args, **kwargs))
    
    def flip(self, *args, **kwargs) -> Self:
        return self.plug(self._t.flip(*args, **kwargs))
    
    def log(self) -> Self: return self.plug(torch.log(self._t))

    def mean(self, dim = None, keepdim: bool = False, *, dtype: torch.dtype = None) -> Self:
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

    def bmm(self, other) -> Self: return self.calc(other, torch.bmm)

    def cat(self, other, dim: int = 0) -> Self:
        if isinstance(other, CreYonT): other = other.tensor()
        return self.plug(torch.cat((self._t, other), dim=dim))
    
    def linear(self, other, bias: torch.Tensor = None) -> Self:
        return self.calc(other, F.linear, bias=bias)
    
    def split(self, split_size_or_sections: int | list[int], dim: int = 0) -> tuple[Self, ...]:
        a, *b = torch.split(self._t, split_size_or_sections, dim=dim)
        return tuple([self.plug(a)] + [CreYonT(torchT=t) for t in b])

    def normalize(self, p: float = 2, dim: int = 1, eps: float = 1e-12) -> Self:
        return self.plug(F.normalize(self._t, p=p, dim=dim, eps=eps))

    def rearrange(self, pattern: str, **axes_lengths: any) -> Self:
        return self.plug(rearrange(self._t, pattern, **axes_lengths))