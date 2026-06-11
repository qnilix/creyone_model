import random
from typing import Optional, Union

import numpy as np
from PIL import Image as PILImage

import torch
import torchvision.transforms.functional as F


INTERPOLATION = {
    'nearest': F.InterpolationMode.NEAREST,
    'lanczos': F.InterpolationMode.LANCZOS,
    'bilinear': F.InterpolationMode.BILINEAR,
    'bicubic': F.InterpolationMode.BICUBIC,
    'box': F.InterpolationMode.BOX,
    'hamming': F.InterpolationMode.HAMMING, 
}


class CropAndResize:

    def __init__(self,
                 size: Union[int, list, tuple],
                 crop_pct: Optional[float|tuple] = None,
                 crop_mode: str = 'random',
                 interpolation: Union[str|int] = 'random', **kwargs):
        if isinstance(size, int): size = [size, size]
        self.size = np.array(size)

        if isinstance(interpolation, int):
            interpolation = list(INTERPOLATION.keys())[interpolation]
        self._interp = interpolation

        self._ratio = None
        if crop_mode == 'ratio':
            self._ratio = kwargs.get('ratio', (3. / 4., 4. / 3.))
        
        i = 0
        if isinstance(crop_pct, float):
            self.w, self.h = list(map(lambda v: int(v / crop_pct), self.size))
            self._corner = self.diff_size(*self.size) // 2
            i = crop_pct
        self._resize = (crop_mode != 'squash' or i < 1.0)

        self._crop_pct = crop_pct
        self._crop_mode = crop_mode

    
    def __repr__(self):
        return f"Crop&Resize: crop={self._crop_mode}({self._crop_pct}) interp={self._interp}"
    
    def __call__(self, image: Union[torch.Tensor, PILImage.Image]):
        self.check_size(image)
        if self._ratio is not None: return self.ratio_crop(image)

        if self._resize:
            image = F.resize(image, self.new_size(), self.interpolation)
            self.check_size(image)
        
        if self._crop_mode == 'pass':
            return image
        elif self._crop_mode in ('squash', 'center'):
            j, i = self.corner
        else:
            j, i = self.corner_random(*self.size)
        return F.crop(image, i, j, *self.size)

    def to_dict(self):
        return {
            'size': self.size.tolist(),
            'crop_pct': self._crop_pct,
            'crop_mode': self._crop_mode,
            'interpolation': self._interp
        }        
    
    def relative_call(self, 
                      image: Union[torch.Tensor, PILImage.Image],
                      shift: Optional[tuple] = None):
        if self._resize:
            image = F.resize(image, 
                             self.new_size(self.keep_pct), self.interpolation)
            self.check_size(image)
        
        if self._crop_mode in ('squash', 'center'):
            if shift is None: return F.crop(image, *self.corner, *self.size)
            i, j = self.corner
            if isinstance(shift, int): shift = (shift, shift)
            i += shift[0]; j += shift[1]
            return F.crop(image, i, j, *self.size)
    
    def check_size(self, image: Union[torch.Tensor, PILImage.Image]):
        self.w, self.h = F.get_image_size(image)
    
    def diff_size(self, target_w: int, target_h: int) -> np.ndarray:
        return np.array([self.w - target_w, self.h - target_h])
    
    def new_size(self, pct: Optional[float|tuple] = None):
        size = self.size
        if self._crop_mode == 'center':
            a = max(self.size[0] / self.w, self.size[1] / self.h)
            size = np.array([self.w * a, self.h * a])
        
        if pct is None: pct = (self._crop_pct or 1.0)
        if not isinstance(pct, float): pct = random.uniform(*self._crop_pct)
        self.keep_pct = pct
            
        return np.floor(size / self.keep_pct).astype(int)
    
    def corner_random(self, tw, th) -> tuple[int, int]:
        dw, dh = self.diff_size(tw, th)
        if min(dw, dh) < 0: return -1, -1
        if dw > 0: dw = random.randint(0, dw)
        if dh > 0: dh = random.randint(0, dh)
        return dw, dh
    
    @property
    def interpolation(self):
        key = self._interp
        if self._interp == 'random':
            key = random.choice([k for k in INTERPOLATION.keys()])
        return INTERPOLATION[key]

    @property
    def corner(self):
        j, i = self._corner
        return int(i), int(j)
    
    @corner.setter
    def corner(self, v): self._corner = v 
    
    

    
