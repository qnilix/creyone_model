#Copyright 2026 Rinka Kiriyama。
#Licensed under the MIT License (MIT). 

import re, yaml, copy, pathlib
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Self, Callable, Optional, Union

import torch
import torch.nn as nn

from .config import BaseCfg
from .dictils import unflatten

def split_model_name_tag(model_name: str, no_tag: str = '') -> tuple[str, str]:
    """Split a dotted model name into a base name and a tag.

    Args:
        model_name: A string of the form ``"name"`` or ``"name.tag"``.
        no_tag: Value returned as the tag when no dot is present.

    Returns:
        ``(base_name, tag)`` where ``tag`` is ``no_tag`` if no dot was found.
    """
    model_name, *tag_list = model_name.split('.', 1)
    if len(tag_list) == 0: tag_list.append(no_tag)
    return model_name, tag_list[0]

def load_yaml(path: pathlib.Path, alt=None) -> dict:
    """Load a YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.
        alt: Optional callable invoked with ``path`` when the file does not
            exist, e.g. to create or download it.

    Returns:
        Parsed YAML content, or an empty dict if the file is absent.
    """
    if path.exists(): return yaml.safe_load(path.read_text())
    if callable(alt): alt(path)
    return {}

@dataclass
class TagManager:
    """Per-model registry of tags and their associated ``BuildBook`` entries.

    ``tags`` is ordered so that the first element is the default tag to use
    when no tag is explicitly requested. ``keepers`` maps each tag to its
    ``BuildBook``. ``is_pretrained`` is ``True`` as soon as any stored config
    has a pretrained weight source.
    """

    tags: deque[str] = field(default_factory=deque)  # priority queue of tags (first is default)
    keepers: dict[str, BaseCfg] = field(default_factory=dict)  # pretrained cfgs by tag
    default: BaseCfg = field(default_factory=BaseCfg)
    is_pretrained: bool = False  # at least one of the configs has a pretrained source set


class BuildFlyer:
    """Configuration builder that pairs a ``BaseCfg`` class with default kwargs.
    """

    def __init__(self, cfg: type[BaseCfg] = BaseCfg, **kwargs):
        """Initialize the builder.

        Args:
            cfg: The ``BaseCfg`` subclass to instantiate when ``get_config`` is
                called. Defaults to ``BaseCfg`` itself.
            **kwargs: Default keyword arguments forwarded to ``cfg.instance()``.
        """
        self.cfg = cfg
        self._int_cfg = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @property
    def int_cfg(self) -> dict:
        return copy.deepcopy(self._int_cfg)

    def get_config(self, base_cfg: type[BaseCfg] = None, **kwargs):
        """Instantiate the config class and return ``(instance, remaining_kwargs)``.

        Args:
            base_cfg: Config class to instantiate. Falls back to ``self.cfg`` when
                ``None``.
            **kwargs: Extra keyword arguments that are merged with ``int_cfg``
                before being passed to ``base_cfg.instance()``. Dot-notation keys
                are expanded via ``unflatten`` before instantiation.

        Returns:
            The ``(instance, remaining_kwargs)`` tuple produced by
            ``BaseCfg.instance()``.
        """
        if base_cfg is None: base_cfg = self.cfg
        kwargs.update(**self.int_cfg)
        return base_cfg.instance(**unflatten(kwargs))

    def cfg_overlay(self, **kwargs):
        """Merge ``kwargs`` into the local internal config, overriding existing keys."""
        self._int_cfg.update(kwargs)


class BuildBook(BuildFlyer):
    """``BuildFlyer`` extended with pretrained-weight loading and builder chaining.

    In addition to configuration building, ``BuildBook`` handles:

    - Separating pretrained kwargs (e.g. ``url``, ``file``) from model kwargs via
      ``BaseCfg.get_pretrained``.
    - Chaining multiple books through ``source``: the source book's config and
      state dict are used as the base, with the local values merged on top.
    - Loading and filtering a ``state_dict`` into a model via ``convert``.
    """

    def __init__(self, cfg: type[BaseCfg] = BaseCfg,
                 stdt_filter: Optional[Callable] = None,
                 source_book: Self = None, **kwargs):
        """Initialize the book.

        Args:
            cfg: The ``BaseCfg`` subclass used for config instantiation.
            stdt_filter: Optional callable ``(state_dict, model) -> state_dict``
                applied after loading weights, e.g. for key remapping.
            source_book: Another ``BuildBook`` whose config and state dict form
                the base. Local values are merged on top.
            **kwargs: Keyword arguments for the config; pretrained-specific keys
                (``url``, ``file``, etc.) are extracted into ``ext_cfg``.
        """
        super().__init__(cfg=cfg, **kwargs)
        self.ext_cfg, self._int_cfg = cfg.get_pretrained(**self._int_cfg)
        self.filter = stdt_filter
        self.source = source_book
    
    @property
    def pret_key(self) -> str:
        if not hasattr(self, '_pret_key'):
            self._pret_key = self.ext_cfg.get_stdt().get('pret_key', None)
        return self._pret_key

    @property
    def int_cfg(self) -> dict:
        """Return the merged internal config kwargs.

        Falls back to the parent ``BuildFlyer.int_cfg`` (a deep copy of
        ``_int_cfg``) when there is no source. Otherwise the source's
        ``int_cfg`` is used as the base and local values are merged on top.
        """
        if self.source is None: return super().int_cfg
        temp = self.source.int_cfg
        temp.update(self._int_cfg)
        return temp

    def pretrained_overlay(self, arch: str, **overlay) -> None:
        """Update ``ext_cfg`` in-place with ``overlay`` values.

        Args:
            arch: Fallback architecture name used when ``ext_cfg.architecture``
                is not already set.
            **overlay: Fields to overwrite on the ``PretrainedCfg`` dataclass.
        """
        overlay['architecture'] = self.ext_cfg.architecture or arch
        self.ext_cfg = replace(self.ext_cfg, **overlay)

    def convert(self, model: nn.Module, wrap_filter=None):
        """Load pretrained weights into ``model`` and record the result.

        Args:
            model: The ``nn.Module`` to load weights into.
            wrap_filter: Optional callable that wraps the existing ``filter``
                before weights are applied, e.g. to compose multiple transforms.
        """
        if wrap_filter is not None: self.filter = wrap_filter(self.filter)
        self.param_keys = model.load_state_dict(self.get_stdt(model))

    def get_stdt(self, model: nn.Module) -> dict:
        """Build and return the merged state dict for ``model``.

        If a ``source`` is set its state dict is loaded first; the local
        ``ext_cfg`` state dict is then merged on top. The combined dict is
        passed through ``self.filter`` when one is set.

        Args:
            model: The target model, forwarded to ``self.filter`` so it can
                perform architecture-aware key remapping.

        Returns:
            A state dict ready to be passed to ``model.load_state_dict``.
        """
        temp = dict()
        if self.source is not None:
            temp = self.source.get_stdt(model)
        temp.update(self.ext_cfg.get_stdt())
        return temp if self.filter is None else self.filter(temp, model)

    def get_meta_for_save(self):
        """Return a dict of metadata suitable for saving alongside a checkpoint."""
        temp = dict()
        temp.update(dict(pretrained_cfg=self.pret_key, int_cfg=self.int_cfg))
        return temp



def encoder_filter(stdt: dict, model: nn.Module) -> dict:
    """Strip the ``"body."`` prefix from every key in ``stdt``.

    Useful when a checkpoint was saved with a wrapper module (e.g. ``body``)
    that is absent in the standalone encoder being loaded.

    Args:
        stdt: Source state dict whose keys may carry a ``"body."`` prefix.
        model: Unused; present to satisfy the filter callable signature.

    Returns:
        New state dict with the prefix removed from all keys.
    """
    return {re.sub(r"body.", '', k): v for k, v in stdt.items()}


class BookRevision:
    """A checkpoint revision that wraps a local file as a ``BuildBook``.

    ``BookRevision`` reads metadata from the checkpoint to discover its parent
    book (if any) and wires the source chain automatically when ``book`` is
    called.
    """

    def __init__(self, pretrained_cfg: pathlib.Path):
        """Args:
            pretrained_cfg: Path to the checkpoint file that this revision wraps.
        """
        self.cfg = pretrained_cfg

    def __repr__(self):
        msg = f"revised by {str(self.__class__.__name__)}"
        if hasattr(self, 'parent'):
            msg += f' (parent={self.parent})'
        return msg + '.'

    def filter(self, stdt: dict, model: nn.Module) -> dict:
        """Strip the ``"body."`` prefix from state-dict keys (see ``encoder_filter``)."""
        return {re.sub(r"body.", '', k): v for k, v in stdt.items()}

    def book(self, default: BaseCfg, keepers: dict = {}):
        """Build and return a ``BuildBook`` backed by this revision's checkpoint.

        The checkpoint's metadata is inspected for a ``pretrained_cfg`` key; if
        found, the corresponding entry in ``keepers`` is set as the source book
        so that its state dict is applied first.

        Args:
            default: Default ``BaseCfg`` subclass forwarded to ``BuildBook``.
            keepers: Mapping of tag → ``BuildBook`` used to resolve the parent.

        Returns:
            A configured ``BuildBook`` with ``source`` and ``filter`` set.
        """
        book = BuildBook(default, stdt_filter=self.filter, file=self.cfg)
        item = book.ext_cfg.get_meta()
        self.parent = item.get('pretrained_cfg', None)
        if isinstance(self.parent, str): book.source = keepers[self.parent]
        return book


class BuildShelf:
    """Registry of named model variants, each represented by a ``BuildBook``.

    A shelf maps model names (optionally suffixed with a dot-separated tag) to
    ``BuildBook`` instances. Tags with pretrained weights are promoted toward
    the front of the priority queue so they become the default; tags marked
    with a trailing ``*`` are pinned as the explicit default for that model.
    """

    def __init__(self, default_cfg: BaseCfg, cfgs: dict[str, Union[BaseCfg, dict]] = {}):
        """Args:
            default_cfg: Fallback ``BaseCfg`` subclass used when no per-entry
                class is specified.
            cfgs: Mapping of ``"name"`` or ``"name.tag"`` strings to config
                dicts. Each entry is registered via ``store``.
        """
        temp = copy.deepcopy(cfgs)
        self.out = defaultdict(TagManager)

        self.default_cfg = default_cfg
        self.default_set = set()

        for k, v in temp.items(): self.store(k, v)

    @classmethod
    def init_withpath(cls, default_cfg: BaseCfg, cfgs: pathlib.Path):
        """Construct a shelf from a YAML file or an already-parsed dict.

        Args:
            default_cfg: Forwarded to ``__init__``.
            cfgs: Either a ``pathlib.Path`` to a ``.yaml`` file (loaded
                automatically) or a plain dict.

        Returns:
            A new ``BuildShelf`` instance.
        """
        if cfgs.suffix == '.yaml': cfgs = load_yaml(cfgs)
        return cls(default_cfg, cfgs)

    def store(self, model_name: str, cfg: dict):
        """Register a single model variant on the shelf.

        The ``model_name`` is split into a base name and an optional tag. The
        resulting ``BuildBook`` is inserted into the tag priority queue:

        - Pretrained variants with an explicit tag go to the front (default).
        - Pretrained variants without a tag go to the front only if no
          pretrained entry exists yet for that model.
        - Non-pretrained variants tagged with ``*`` go to the front.
        - Everything else is appended to the back.

        Args:
            model_name: ``"name"`` or ``"name.tag"`` string; a trailing ``*``
                on the tag marks it as the explicit default.
            cfg: Config dict forwarded to ``BuildBook``.
        """
        book = BuildBook(self.default_cfg, **cfg)
        has_weights = book.ext_cfg.has_weights

        name, tag = split_model_name_tag(model_name)
        tag = tag.strip('*')

        default_cfg = self.out[name]

        flag = 2
        if has_weights:
            flag = 0 if tag != '' else (2 if default_cfg.is_pretrained else 1)
            self.out[name].is_pretrained = True
        elif (tag.endswith('*') and name not in self.default_set):
            flag = 0

        if flag < 2:
            self.out[name].tags.appendleft(tag)
            if flag == 0: self.default_set.add(name)
        else:
            self.out[name].tags.append(tag)

        book._pret_key = tag
        self.out[name].keepers[tag] = book

    def take(self, name: str,
             pretrained_cfg: Optional[Union[str|list|pathlib.Path]] = None) -> BuildBook:
        """Retrieve the ``BuildBook`` for a given model name and pretrained tag.

        Args:
            name: Base model name (without tag).
            pretrained_cfg: How to select the variant:

                - ``list`` — returns a list by calling ``take`` for each element.
                - ``BookRevision`` — delegates to ``BookRevision.book`` with the
                  shelf's keepers for ``name``.
                - ``str`` — looks up that exact tag; an empty string returns a
                  plain ``BuildBook`` with no pretrained weights.
                - ``None`` — returns ``None`` (caller selects the default tag).

        Returns:
            The matching ``BuildBook``, a list thereof, or ``None``.
        """
        if isinstance(pretrained_cfg, list):
            return [self.take(name, cfg) for cfg in pretrained_cfg]
        if isinstance(pretrained_cfg, BookRevision):
            return pretrained_cfg.book(self.default_cfg,
                                       keepers=self.out[name].keepers)
        if isinstance(pretrained_cfg, str):
            if pretrained_cfg == '':
                return BuildBook(self.default_cfg)
            return self.out[name].keepers[pretrained_cfg]
        return None





    



