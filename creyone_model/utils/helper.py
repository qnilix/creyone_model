from typing import Any, Callable, Union

import torch


def load_weights_only_compat(func: Callable, *args, weights_only: bool = True, **kwargs):
    """Call *func* with ``weights_only`` kwarg, falling back without it on older PyTorch."""
    try:
        return func(*args, weights_only=weights_only, **kwargs)
    except TypeError:
        return func(*args, **kwargs)


def load_ckpt(
        checkpoint_path: str,
        device: Union[str, torch.device] = 'cpu',
        weights_only: bool = False):
    """Load a checkpoint file to *device*, gracefully handling older PyTorch without ``weights_only``."""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def define_stdt_key(ckpt: dict, use_ema: bool = True) -> str:
    """Return the key under which the state dict is stored in *ckpt*.

    Prefers EMA variants when *use_ema* is True.  Returns ``''`` when the
    checkpoint itself is the state dict (no nesting).
    """
    if use_ema:
        if ckpt.get('state_dict_ema'): return 'state_dict_ema'
        if ckpt.get('model_ema'): return 'model_ema'
    if 'state_dict' in ckpt: return 'state_dict'
    if 'model' in ckpt: return 'model'
    return ''


def clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip known wrapper prefixes (DDP ``module.``, torch.compile ``_orig_mod.``) from all keys."""
    cleaned_state_dict = {}
    to_remove = (
        'module.',      # DDP wrapper
        '_orig_mod.',   # torchcompile dynamo wrapper
    )
    for k, v in state_dict.items():
        for r in to_remove:
            k = k.removeprefix(r)
        cleaned_state_dict[k] = v
    return cleaned_state_dict


def load_state_dict(path: str, use_ema: bool = True, **kwargs) -> dict[str, Any]:
    """Load and return a cleaned state dict from a checkpoint file at *path*.

    Automatically selects the EMA state dict when available and *use_ema* is
    True.  Extra keyword arguments are forwarded to :func:`load_ckpt`.
    """
    checkpoint = load_ckpt(path, **kwargs)
    stdt_key = ''
    if isinstance(checkpoint, dict):
        stdt_key = define_stdt_key(checkpoint, use_ema=use_ema)
        if 'pretcfg' in checkpoint:
            stdt_key = ''
    state_dict = clean_state_dict(checkpoint[stdt_key] if stdt_key else checkpoint)
    return state_dict