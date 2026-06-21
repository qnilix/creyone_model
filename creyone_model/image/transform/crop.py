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
    """Crop and resize transform for PIL images and torch tensors.

    Supports multiple crop modes (random, center, squash, ratio, pass) and
    interpolation methods. Can be used as a callable transform in a pipeline.
    """

    def __init__(self,
                 size: Union[int, list, tuple],
                 crop_pct: Optional[float|tuple] = None,
                 crop_mode: str = 'random',
                 interpolation: Union[str|int] = 'random', **kwargs):
        """Initialize CropAndResize.

        Args:
            size: Output (width, height). An int is treated as a square.
            crop_pct: Fraction of the image to keep before resizing. A float
                uses a fixed scale; a tuple ``(min, max)`` samples uniformly.
            crop_mode: One of ``'random'``, ``'center'``, ``'squash'``,
                ``'ratio'``, or ``'pass'``.
            interpolation: Interpolation method name (``'bilinear'`` etc.) or
                index into ``INTERPOLATION``. ``'random'`` picks one each call.
            **kwargs: Extra options, e.g. ``ratio`` for ``crop_mode='ratio'``.
        """
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
        """Apply crop-and-resize to an image.

        Args:
            image: Input PIL image or torch tensor (C, H, W).

        Returns:
            Cropped (and optionally resized) image of the configured size.
        """
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
        """Return the transform configuration as a serialisable dict."""
        return {
            'size': self.size.tolist(),
            'crop_pct': self._crop_pct,
            'crop_mode': self._crop_mode,
            'interpolation': self._interp
        }        
    
    def relative_call(self,
                      image: Union[torch.Tensor, PILImage.Image],
                      shift: Optional[tuple] = None):
        """Crop with an optional pixel offset from the computed corner.

        Args:
            image: Input PIL image or torch tensor.
            shift: ``(row_offset, col_offset)`` added to the crop corner.
                An int is broadcast to both axes. ``None`` uses no offset.

        Returns:
            Cropped image of the configured size.
        """
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
        """Update ``self.w`` and ``self.h`` from the current image dimensions."""
        self.w, self.h = F.get_image_size(image)
    
    def diff_size(self, target_w: int, target_h: int) -> np.ndarray:
        """Return ``[w - target_w, h - target_h]`` as the available crop slack."""
        return np.array([self.w - target_w, self.h - target_h])
    
    def new_size(self, pct: Optional[float|tuple] = None):
        """Compute the resize target before cropping.

        Args:
            pct: Override for ``crop_pct``. If ``None``, uses the stored value.

        Returns:
            Integer ``[width, height]`` array for the resize step.
        """
        size = self.size
        if self._crop_mode == 'center':
            a = max(self.size[0] / self.w, self.size[1] / self.h)
            size = np.array([self.w * a, self.h * a])
        
        if pct is None: pct = (self._crop_pct or 1.0)
        if not isinstance(pct, float): pct = random.uniform(*self._crop_pct)
        self.keep_pct = pct
            
        return np.floor(size / self.keep_pct).astype(int)
    
    def corner_random(self, tw, th) -> tuple[int, int]:
        """Return a random (x, y) crop corner within the available slack.

        Returns ``(-1, -1)`` if the image is smaller than the target size.
        """
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
    
    

    
