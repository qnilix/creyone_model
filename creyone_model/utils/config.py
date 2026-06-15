import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Union

import torch
from torch.hub import load_state_dict_from_url

from .dictils import pop_keys
from .helper import load_state_dict, load_weights_only_compat


_CHECK_HASH = False
_DOWNLOAD_PROGRESS = False


def type_subclass(cls: type, classinfo) -> bool:
    """Call issubclass safely, returning False if cls is not a type (e.g. MISSING)."""
    try:
        return issubclass(cls, classinfo)
    except TypeError:
        return False


@dataclass
class PretrainedCfg:

    url: Optional[Union[str]] = None
    file: Optional[str] = None
    state_dict: Optional[dict[str, any]] = None  # in-memory state dict

    source: Optional[str] = None
    architecture: Optional[str] = None
    tag: Optional[str] = None
    custom_load: bool = False  # use custom model specific model.load_pretrained() (ie for npz files)

    @property
    def has_weights(self) -> bool:
        return self.url or self.file
    
    def resolve_source(self) -> tuple[str, any]:        
        if self.state_dict: return 'state_dict', self.state_dict
        if self.file: return 'file', self.file
        if self.url: return 'url', self.url
        return ('', '')
    
    def get_stdt(self) -> dict:
        load_from, loc = self.resolve_source()
        if load_from == 'state_dict': return loc
        if load_from == 'file': return load_state_dict(loc)
        if load_from == 'url':
            kw = {'map_location': 'cpu',
                  'progress': _DOWNLOAD_PROGRESS,
                  'check_hash': _CHECK_HASH,
                  'weights_only': True}
            return load_weights_only_compat(load_state_dict_from_url, loc, **kw)
        raise FileNotFoundError(f"Get state_dict from ({load_from}: {loc}) but not found")
    
    def get_meta(self) -> dict:
        if self.file is None: return {}
        return torch.load(self.file, weights_only=False).get('meta', {})
    
    def get_meta_for_save(self): return asdict(self)


@dataclass
class BaseCfg:
    """Base class for hierarchical configuration dataclasses.

    Subclasses should be decorated with ``@dataclass`` and may declare fields
    of primitive types or other ``BaseCfg`` subclasses. Fields whose
    ``default_factory`` is a ``BaseCfg`` subclass are built recursively by
    ``instance()``.
    """

    @staticmethod
    def get_pretrained(**kwargs) -> tuple[PretrainedCfg, dict]:
        name = [f.name for f in fields(PretrainedCfg)]
        temp = dict()
        pret = kwargs.pop('pretrained', None)
        if isinstance(pret, dict): temp.update(pop_keys(name, pret))
        temp.update(pop_keys(name, kwargs))
        return PretrainedCfg(**temp), kwargs

    @classmethod
    def instance(cls, **kwargs) -> tuple[Any, dict[str, Any]]:
        """Build an instance from kwargs and return any unconsumed kwargs.

        Iterates over all dataclass fields, popping matching keys from kwargs.
        For fields whose ``default_factory`` is a ``BaseCfg`` subclass, the
        method recurses, passing both the field-specific kwargs and all
        remaining kwargs so that shared parameters propagate down. Each nested
        call returns its own unconsumed kwargs, which continue up the chain.

        Args:
            **kwargs: Keyword arguments keyed by field name. Values for nested
                ``BaseCfg`` fields should be provided as a ``dict`` of
                sub-kwargs for that config.

        Returns:
            A ``(instance, remaining_kwargs)`` tuple where ``remaining_kwargs``
            holds any kwargs not consumed by this class or its nested configs.
        """
        kw: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in kwargs: kw[f.name] = kwargs.pop(f.name)
            if not type_subclass(f.default_factory, BaseCfg): continue
            # Field-specific kwargs take priority over shared remaining kwargs.
            # Build a new dict so the caller's original dict is never mutated.
            temp = {**kwargs, **kw.get(f.name, {})}
            kw[f.name], kwargs = f.default_factory.instance(**temp)
        return cls(**kw), kwargs













