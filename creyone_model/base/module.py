from typing import Optional
from functools import partial

import torch
import torch.nn as nn

from timm.utils import accuracy
from .config import get_target_layers


class ModuleBase(nn.Module):

    def __init__(self, model_meta: dict = None, target_layers: Optional[list[str]] = None):
        super().__init__()
        self._cont = dict()
        self.model_meta = model_meta
        self.target_layers = self.get_target_layers(target_layers)
        self.midstate_init()
    
    def get_target_layers(self, target_layers: Optional[list[str]] = None) -> list:
        return get_target_layers(target_layers)
    
    def container(self, key: str, otherwise = None, error: bool = False) -> torch.Tensor:
        if error and key not in self._cont:
            raise ValueError(f"{key} not found in {self.__class__}._cont")
        return self._cont.get(key, otherwise)
    
    def loss_func(self, o: torch.Tensor, t: torch.Tensor):
        raise AttributeError('loss_func is needed but it is not defined.')
    
    @property
    def cyng_loss(self):
        o = self.container('output', error=True)
        t = self.container('target', error=True)
        self._cont['loss'] = self.loss_func(o, t)
        return self._cont['loss']
    
    @property
    def cyng_accuracy(self):
        return accuracy(self._cont['output'], self._cont['target'], topk=(1, 5))
    
    def midstate_init(self):
        self._keeping_hiddens_list = list(); self._handles = list()
        self._hiddens = dict(); self._gradients = dict()

        for t in self.target_layers: self.keep_hidden(t); self.keep_grad(t)
    
    def _keep_hidden_meta(self, module, *args, name: Optional[str] = None):
        output = args[-1]
        self._hiddens[name] = output
    
    def keep_hidden(self, name: str, with_kwargs: bool = False):
        if name in self._keeping_hiddens_list: return
        self._keeping_hiddens_list.append(name)
        f = partial(self._keep_hidden_meta, name=name); item = self
        for n in name.split('.'): item = getattr(item, n)
        h = item.register_forward_hook(f, with_kwargs=with_kwargs)
        self._handles.append(h)
    
    def _keep_grad_meta(self, module, *args, name: Optional[str] = None):
        output = args[-1]
        if not getattr(output, 'requires_grad', False): return 

        def _store_grad(grad): self._gradients[name] = grad
        output.register_hook(_store_grad)
    
    def keep_grad(self, name: str):
        f = partial(self._keep_grad_meta, name=name); item = self
        for n in name.split('.'): item = getattr(item, n)
        self._handles.append(item.register_forward_hook(f))
    
    @property
    def cyng_gradients(self) -> dict[str, torch.Tensor]:
        loss = self.container('loss', self.cyng_loss)
        loss.backward(retain_graph=True)
        return {k: t.clone().detach() for k, t in self._gradients.items()}
    
    @property
    def cyng_hidden(self) -> dict[str, torch.Tensor]:
        return {k: t.clone().detach() for k, t in self._hiddens.items()}
    
    # visual items ###########################################################    
    #def text_processing_init(self, tokenizer: str):
    #    self._tokenizer = Tokenizer(pathlib.Path(tokenizer))



    


