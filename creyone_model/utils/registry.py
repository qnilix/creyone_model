import sys
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional


_model_entrypoints: dict[dict[str, Callable[..., any]]] = defaultdict(dict)  # mapping of model names to architecture entrypoint fns
_config_entrypoints: dict[str, Callable[..., any]] = dict()  # mapping of model names to architecture entrypoint fns
_processor_entrypoints: dict[str, Callable[..., any]] = dict()  # mapping of model names to architecture entrypoint fns


def _recfunc(fn: Callable[..., Any], entrys: dict) -> None:
    mod = sys.modules[fn.__module__]
    name = fn.__name__
    if not hasattr(mod, '__all__'): mod.__all__ = []
    mod.__all__.append(name)

    if name in entrys:
        warnings.warn(
            f'Overwriting {name} in registry with {fn.__module__}.{name}. This is' 
            f'because the name being registered conflicts with an existing name. '
            f'Please check if this is not expected.',
            stacklevel=2,
        )
    entrys[name] = fn
    return fn


def register_model(task_name: str = 'any'):

    def _register_model(fn: Callable[..., Any]) -> Callable[..., Any]:
        return _recfunc(fn, _model_entrypoints[task_name])
    
    return _register_model


def register_config():

    def _register_config(fn: Callable[..., Any]) -> Callable[..., Any]:
        return _recfunc(fn, _config_entrypoints)
    
    return _register_config


def register_processor(fn: Callable[..., Any]) -> Callable[..., Any]:
    print(f"record: {fn.__name__}")
    return _recfunc(fn, _processor_entrypoints)


def is_model(model_name: str, task_name: Optional[str] = None) -> bool:
    return model_name in _model_entrypoints[task_name]

def is_processor(name: str) -> bool: return name in _processor_entrypoints


def model_entrypoint(model_name: str, 
                     task_name: Optional[str] = None) -> Callable[..., Any]:
    """Fetch a model entrypoint for specified model name
    """
    if task_name is not None and model_name in _model_entrypoints[task_name]: 
        return _model_entrypoints[task_name][model_name]
    if model_name not in _model_entrypoints['any']:
        raise RuntimeError(f'Model ({model_name}) not found in this repository.')
    return _model_entrypoints['any'][model_name]


def processor_entrypoint(model_name: str) -> Callable[..., Any]:
    """Fetch a model entrypoint for specified model name
    """
    if model_name not in _processor_entrypoints:
        raise RuntimeError(f'Processor ({model_name}) not found in this repository.')
    return _processor_entrypoints[model_name]
