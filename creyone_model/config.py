from .base.config import ModelCfg

class _DUMMY_LC_CTX:

    def __enter__(self): pass
    def __exit__(self, *args): pass


class DummyModelConfig:

    def set_layer_config(self): return _DUMMY_LC_CTX()



def get_config(task_name: str) -> ModelCfg:
    if task_name == 'image_classification':
        from .image.config import ImageClassificationCfg
        return ImageClassificationCfg
    return DummyModelConfig