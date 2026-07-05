#Copyright 2026 Rinka Kiriyama。
#Licensed under the MIT License (MIT). 

from typing import Any

def pop_keys(keys: list[str], item: dict | None = None) -> dict:
    if item is None: return {}
    return {k: item.pop(k) for k in keys if k in item}


def set_nested(items: dict, key: str, val: Any) -> None:
    if '.' not in key:
        items[key] = val
        return
    head, tail = key.split('.', 1)
    if head not in items:
        items[head] = {}
    set_nested(items[head], tail, val)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` on top of ``base``.

    Unlike ``dict.update``/``{**base, **override}``, a key present as a dict
    on both sides is merged field-by-field instead of one side wholesale
    replacing the other. ``override`` wins on any non-dict conflict.
    """
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def unflatten(items: dict) -> dict:
    kw = {}
    for k, v in items.items():
        if not isinstance(v, dict):
            set_nested(kw, k, v)
            continue
        v = unflatten(v)
        if k in kw:
            v = deep_merge(v, kw[k])
        kw[k] = v
    return kw
