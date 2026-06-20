import os, json, pathlib
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Self, Union

import torch
from torch.hub import load_state_dict_from_url

from .dictils import pop_keys
from .helper import load_state_dict, load_weights_only_compat
from .huggingface import has_hf_hub, check_cached_file, load_state_dict_from_hf, download_from_hf, parse_imgconf


_USE_OLD_CACHE = int(os.environ.get('CREYONE_USE_OLD_CACHE', 0)) > 0
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

class PretrainedHFTransCfg:
    
    model_type: str = ""
    base_config_key: str = ""
    sub_configs: dict[str, Self] = {}
    has_no_defaults_at_init: bool = False
    attribute_map: dict[str, str] = {}
    base_model_tp_plan: Optional[dict[str, any]] = None
    base_model_pp_plan: Optional[dict[str, tuple[list[str]]]] = None
    _auto_class: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, any], **kwargs) -> dict:
        _ = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # We remove it from kwargs so that it does not appear in `return_unused_kwargs`.
        config_dict["attn_implementation"] = kwargs.pop("attn_implementation", None)
        config_dict.update(kwargs)
        return config_dict

@dataclass
class PretrainedHFCfg(PretrainedCfg):
    
    hf_hub_id: Optional[str] = None
    hf_hub_weights: Optional[str] = 'pytorch_model.bin'
    hf_hub_imgconf: Optional[str] = 'preprocessor_config.json'

    @property
    def has_weights(self) -> bool:
        return self.hf_hub_id or super().has_weights
    
    def resolve_source(self) -> tuple[str, any]:
        hf_available = has_hf_hub(True)
        if self.source in ('transformers', 'hf-hub') and hf_available:
            return self.source, self.hf_hub_id

        if self.state_dict: return 'state_dict', self.state_dict
        if self.file: return 'file', self.file

        old_cache_valid = False
        if _USE_OLD_CACHE and self.url:
            old_cache_valid = check_cached_file(self.url)
        if not old_cache_valid and self.hf_hub_id and hf_available:
            return 'hf-hub', self.hf_hub_id
        elif self.url:
            return 'url', self.url

        return ('', '')
    
    def get_stdt(self):
        load_from, loc = self.resolve_source()
        if load_from in ('state_dict', 'file', 'url'):
            return super().get_stdt(loc)
        if load_from not in ('hf-hub', 'transformers'):
            raise ImportError(f"{load_from} is not supported")
        if self.hf_hub_weights:
            return load_state_dict_from_hf(loc, self.hf_hub_weights)
        return load_state_dict_from_hf(loc, weights_only=True)
    
    def get_hf_json(self, file):
        file = download_from_hf(self.hf_hub_id, file, raise_error = False)
        if file is None: return None
        return json.loads(pathlib.Path(file).read_text())
    
    def get_imgproc(self) -> Optional[dict]:
        if self.source == 'transformers':
            cfg, rem = parse_imgconf(**self.get_hf_json(self.hf_hub_imgconf))
            return cfg
        return None


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
        name = [f.name for f in fields(PretrainedHFCfg)]
        temp = dict()
        pret = kwargs.pop('pretrained', None)
        if isinstance(pret, dict): temp.update(pop_keys(name, pret))
        temp.update(pop_keys(name, kwargs))
        if temp.get('hf_hub_id', None) is not None:
            cfg = PretrainedHFCfg(**temp)
            src = cfg.get_imgproc()
            if src is not None: kwargs.update(src)
            return cfg, kwargs
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













