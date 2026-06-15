from ..utils.registry import register_model

from .task import from_encoder, ImageClassification


@register_model('image_classification')
def vit(size: str, patch: str = '16', image_size: str = '224', *_,
        num_classes: int = 1000,
        **kwargs) -> ImageClassification:
    """vit
        When calling the model, specify it as vit_<size>_<patch>_<img_size>.
            e.g. vit_base_16_224
        Note that img_size is also used when calling a pre-trained model. 
        To specify a new image resolution, use <img_size(new_size)>.
            e.g. vit_base_16_224(256) or vit_base_16_224(128x256)
    """
    variant = f'vit-{size}-{patch}-{image_size}'
    return from_encoder(variant, output_dim=num_classes,
                        cynn_i=ImageClassification, **kwargs)