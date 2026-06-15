import logging
from typing import Callable, Optional, Union
from dataclasses import dataclass, field

from .config import BaseCfg
from .book import BuildShelf, BuildBook


NO_PRETCFG = "No pretrained configuration specified for {name}.{pret} model. \
    Using a default. Please add a config to the model pretrained_cfg registry or pass explicitly."


@dataclass
class ModelBuilder:

    default: BaseCfg = field(default_factory=BaseCfg)
    shelf: Optional[BuildShelf] = None

    log: logging.Logger = field(default=logging.getLogger(__file__))

    def ref_shelf(self,
                  variant: str = '',
                  pretrained_cfg: Optional[Union[str|dict]] = None,
                  pretrained_cfg_overlay: dict = {},
                  **kwargs):
        book = None
        if self.shelf is not None:
            book = self.shelf.take(variant, pretrained_cfg=pretrained_cfg)
        if book is None: return BuildBook(self.default, **kwargs)
        book.pretrained_overlay(variant, **pretrained_cfg_overlay)
        return book

    def build(self,
              model_cls: Callable,
              book: BuildBook,
              pretrained: bool = False, 
              wrap_fliter: Callable = None,
              return_remained_kwargs: bool = False, 
              **kwargs):
        cfg, rem = book.get_config(self.default, **kwargs)
        self.log.debug(cfg)
        
        model = model_cls(cfg=cfg)
        model.reset_parameters()
        if pretrained: book.convert(model, wrap_filter = wrap_fliter)
        rem['model_meta'] = book.get_meta_for_save()

        if return_remained_kwargs: return model, rem
        return model
        


