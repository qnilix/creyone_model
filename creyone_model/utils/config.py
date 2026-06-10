from dataclasses import dataclass, fields
from typing import Any


def type_subclass(cls: type, classinfo) -> bool:
    """Call issubclass safely, returning False if cls is not a type (e.g. MISSING)."""
    try:
        return issubclass(cls, classinfo)
    except TypeError:
        return False


@dataclass
class BaseCfg:
    """Base class for hierarchical configuration dataclasses.

    Subclasses should be decorated with ``@dataclass`` and may declare fields
    of primitive types or other ``BaseCfg`` subclasses. Fields whose
    ``default_factory`` is a ``BaseCfg`` subclass are built recursively by
    ``instance()``.
    """

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













