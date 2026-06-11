from typing import Optional
from dataclasses import dataclass

from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip

from con24ma import DataClassConfig, ArgField
from .crop import CropAndResize


@dataclass
class ImageTransformCfg(DataClassConfig):

    interpolation: str = ArgField('bicubic')
    crop_pct: float = ArgField(0.9)
    crop_mode: str = ArgField('center')

    augmix: bool = ArgField(False, action='store_true')
    randaug: str = ArgField('')

    train_interpolation: str = 'random'
    train_crop_pct_min: float = 0.8
    hflip: float = 0.5

    image_norm: str = 'base'

    no_model_default: bool = ArgField(False, action='store_true')

    def __repr__(self):
        msg = f"(train: interp={self.train_interpolation} crop=[{self.train_crop_pct_min}, 1.0], "
        msg += f"val: interp={self.interpolation} crop={self.crop_mode}[{self.crop_pct}]"
        if self.no_model_default: msg += f" strictly"
        return msg + ")"

    def crop_and_resize(self, is_training: bool = True,
                        crop_pct: Optional[float] = None,
                        crop_mode: Optional[str] = None,
                        interpolation: Optional[str] = None):
        if is_training:
            return {'crop_pct': (self.train_crop_pct_min, 1.0),
                    'interpolation': self.train_interpolation}
        if self.no_model_default:
            crop_pct = None; crop_mode = None; interpolation = None
        return {
            'crop_pct': crop_pct or float(self.crop_pct),
            'crop_mode': crop_mode or str(self.crop_mode),
            'interpolation': interpolation or str(self.interpolation)
        }
    
    def compose(self, is_training: bool = False, 
                image_size: tuple = (224, 224),
                crop_pct: Optional[float] = None,
                crop_mode: Optional[str] = None,
                interpolation: Optional[str] = None) -> Compose:
        kw = self.crop_and_resize(is_training=is_training, 
                                  crop_pct=crop_pct,
                                  crop_mode=crop_mode,
                                  interpolation=interpolation)
        temp = [CropAndResize(size = image_size, **kw)]
        if is_training and self.hflip > 0.0:
            temp += [RandomHorizontalFlip(self.hflip)]
        return Compose(temp + [ToTensor()])
