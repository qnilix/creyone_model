import pathlib
from typing import Optional
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from .base import ImageProcessor

from .losses import ClassificationLoss
from .config import ImageClassificationCfg


class ImageClassification(ImageProcessor):

    def __init__(self, body: nn.Module, 
                 cfg: ImageClassificationCfg = None,
                 loss_bce: bool = False,
                 trainable: str = 'all',
                 model_meta: Optional[dict] = None,
                 **kwargs):
        super().__init__(body, cfg=cfg, model_meta=model_meta, **kwargs)
        self.body.trainable_parameters(trainable)
        self._loss_fn = ClassificationLoss(bce=(self.cfg.daug_type == 'mixup') or loss_bce)
    
    def get_cfg(self, cfg, **kwargs):
        return cfg or ImageClassificationCfg(**kwargs)
    
    def forward(self, inputs: dict) -> dict:
        self._cont = inputs
        out = self.body(self.image_first(inputs['images']))
        self._cont.update({'output': out, 'size': out.B})
        return self._cont
    
    def loss_func(self, o, t):
        p = self.container('lam')
        if p is None: return self._loss_fn(o, t)
        if len(t.shape) == 1: t = F.one_hot(t, o.shape[-1])
        t = t * p + t.flip(dims=[0]) * (1 - p)
        return self._loss_fn(o, t)

    @property
    def collate_fn(self) -> dict:
        return {'train': self.collate, 'val': self.collate}
    
    def collate(self, batch):
        batch = default_collate(batch)
        batch['images'] = self.image_first.normalize(batch['images'])
        return batch
    
    @property
    def cyng_targetLayers(self):
        return (self.cyng_hidden, None)
    
    def cyng(self) -> list:
        item = ['accuracy', 'loss']
        if len(self.target_layers) != 0: item.append('targetLayers')
        return item

    def metric_list(self) -> list: 
        return [('acc', 'acc5'), ('acc', 'acc1'), 'loss']