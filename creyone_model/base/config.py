from typing import Callable, Union, Optional
from dataclasses import dataclass

from torch import nn

from timm.layers import set_layer_config

from con24ma import DataClassConfig, ArgField, DictField
from .name import parse_model_name, parse_pretrained_cfg

def get_model_for_task(name: str, task: str) -> tuple[Callable, list[str], str]:
    """Look up the model factory function and associated metadata for *name*.

    The name is first parsed to determine its source (``"creyone"`` or
    ``"hf-hub"``), and the pretrained config tag is extracted.  The remainder
    of the name (after removing the base model identifier) is treated as
    positional arguments passed to the factory.

    Args:
        name: Full model name string, e.g. ``"resnet50-arg1"`` or a URI.
        task: Task identifier used to look up the correct registry entry.

    Returns:
        A ``(create_fn, model_args, pret_cfg)`` tuple where

        * ``create_fn`` - the model factory callable,
        * ``model_args`` - extra positional arguments encoded in the name,
        * ``pret_cfg``   - pretrained-config tag (may be an empty string).
    """
    model_source, model_name = parse_model_name(name)
    pret_cfg = parse_pretrained_cfg(model_name, model_source=model_source)

    model_name, *model_args = model_name.split('-')
    from ..utils.registry import model_entrypoint
    create_fn = model_entrypoint(model_name, task)
    return create_fn, model_args, pret_cfg

@dataclass
class ModelCfg(DataClassConfig):
    """Configuration dataclass for building and configuring a model.

    Attributes:
        model: Model name or URI (default: ``"resnet50"``).
        model_kwargs: Additional keyword arguments forwarded to the model factory.
        pretrained: Whether to load pretrained weights.
        trained_file: Path to a local checkpoint file (empty string = no file).
        task_name: Task identifier used for registry lookup (default: ``"any"``).
        scriptable: Passed to ``timm`` ``set_layer_config`` for TorchScript export.
        exportable: Passed to ``timm`` ``set_layer_config`` for ONNX/export mode.
        no_jit: Passed to ``timm`` ``set_layer_config`` to disable JIT.
    """

    model: str = ArgField('resnet50', ['-m'])
    model_kwargs: dict = DictField()

    pretrained: bool = False
    trained_file: str = ArgField('')

    task_name: str = 'any'

    scriptable: Optional[bool] = None
    exportable: Optional[bool] = None
    no_jit: Optional[bool] = None

    def set_layer_config(self):
        """Return a context manager that applies the layer config to ``timm`` layers."""
        return set_layer_config(scriptable=self.scriptable, exportable=self.exportable, no_jit=self.no_jit)

    def create_model(self, **kwargs) -> nn.Module:
        """Instantiate the model described by this config.

        Args:
            **kwargs: Extra keyword arguments merged into the model factory call,
                      taking precedence over ``model_kwargs``.

        Returns:
            The constructed ``nn.Module``.
        """
        create_fn, model_args, pret_cfg = get_model_for_task(self.model, self.task_name)
        with self.set_layer_config():
            model = create_fn(*model_args, pretrained=self.pretrained, pretrained_cfg=pret_cfg, **kwargs)
        return model