from typing import Optional
from dataclasses import dataclass

from timm.layers import set_layer_config

from con24ma import DataClassConfig, ArgField, DictField
from .name import parse_model_name, parse_pretrained_cfg


def get_target_layers(target_layers: Optional[list[str]] = None) -> list:
    t = target_layers or list()
    return t.split('/') if isinstance(t, str) else t


@dataclass
class ModelCfg(DataClassConfig):

    model: str = ArgField('resnet50', ['-m'])
    model_kwargs: dict = DictField()

    pretrained: Optional[bool] = None
    trained_file: Optional[str] = ArgField(None)

    task_name: str = 'any'

    scriptable: Optional[bool] = None
    exportable: Optional[bool] = None
    no_jit: Optional[bool] = None

    def set_layer_config(self):
        return set_layer_config(scriptable=self.scriptable, exportable=self.exportable, no_jit=self.no_jit)
    
    def create_model(self, **kwargs):
        model_source, model_name = parse_model_name(self.model)
        pret_cfg = parse_pretrained_cfg(model_name, model_source=model_source)
        pretrained = (self.pretrained or pret_cfg != '')
    
        model_name, *model_args = model_name.split('-')
        from ..utils.registry import model_entrypoint
        create_fn = model_entrypoint(model_name, self.task_name)
        with self.set_layer_config():
            model = create_fn(*model_args, pretrained=pretrained, pretrained_cfg=pret_cfg, **kwargs)
        return model
    
    def target_layers(self, target_layers: list[str] = None) -> list[str]:
        return get_target_layers(target_layers)