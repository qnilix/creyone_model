from typing import Any, Optional, Union
from .base.config import get_model_for_task
from .utils.config import PretrainedCfg

def get_config(task_name: str):
    if task_name == 'image_classification':
        from .image.config import ImageClassificationCfg
        return ImageClassificationCfg


def create_model(
    model_name: str,
    task_name: str = 'any',
    pretrained: Optional[bool] = None,
    pretrained_cfg: Optional[Union[str, dict[str, Any], PretrainedCfg]] = None,
    **kwargs,
):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    model_cfg = model_cfg or get_config(task_name)()

    create_fn, model_args, pretrained_tag = get_model_for_task(model_name, task_name)
    pretrained_cfg = pretrained_cfg or pretrained_tag

    with model_cfg.set_layer_config():
        model = create_fn(
            *model_args,
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            **kwargs,
        )

    return model