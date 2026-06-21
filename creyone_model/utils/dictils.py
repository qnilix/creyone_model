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


def unflatten(items: dict) -> dict:
    kw = {}
    for k, v in items.items():
        if not isinstance(v, dict):
            set_nested(kw, k, v)
            continue
        v = unflatten(v)
        if k in kw:
            v.update(kw[k])
        kw[k] = v
    return kw
