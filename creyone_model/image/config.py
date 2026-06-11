from dataclasses import dataclass, field

from .base import ImageProcessorCfg


@dataclass
class ImageClassificationCfg(ImageProcessorCfg):

    task_name: str = 'image_classification'
    num_classes: int = 1000

    attack_type: str = None

    def target_layers(self, target_layers: list[str] = None) -> list[str]:
        target_layers = super().target_layers(target_layers)
        if self.attack_type is None: return target_layers
        target_layers += ['image_first']
        return target_layers