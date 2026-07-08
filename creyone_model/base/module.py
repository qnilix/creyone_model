#Copyright 2026 Rinka Kiriyama。
#Licensed under the MIT License (MIT). 

from typing import Optional
from functools import partial

import torch
import torch.nn as nn


class ModuleBase(nn.Module):
    """Base ``nn.Module`` with built-in intermediate-state capture utilities.

    Subclasses must override :meth:`loss_func` to use gradient-related
    properties.  Intermediate activations and gradients are captured via
    forward hooks registered for each name in *target_layers*.

    Attributes:
        _cont: Internal key-value container for passing tensors (e.g. outputs,
               targets, computed loss) between methods.
        model_meta: Arbitrary metadata dict forwarded from the builder.
        target_layers: Resolved list of dot-separated layer paths whose
                       activations and gradients will be tracked.
    """

    def __init__(self, model_meta: dict = None, target_layers: str | list[str] = ''):
        """Initialize the module and register hooks for all target layers.

        Args:
            model_meta: Optional metadata dict (e.g. class names, dataset info).
            target_layers: ``"/"``-separated string or list of dot-separated
                           layer attribute paths to hook (e.g. ``"layer4.1"``).
        """
        super().__init__()
        self._cont = dict()
        self.model_meta = model_meta
        self.target_layers = self.get_target_layers(target_layers)
        self.midstate_init()

    def get_target_layers(self, target_layers: str | list[str] = '') -> list:
        """Normalize a target-layer specification into a flat list of layer names.

        Args:
            target_layers: Either a ``"/"``-separated string of layer names or a
                           list of layer names.  An empty string or falsy value
                           returns an empty list.

        Returns:
            A list of layer name strings.
        """
        t = target_layers or list()
        return t.split('/') if isinstance(t, str) else t

    def container(self, key: str, otherwise=None, error: bool = False) -> torch.Tensor:
        """Retrieve a value from the internal tensor container ``_cont``.

        Args:
            key: Container key to look up (e.g. ``"output"``, ``"target"``).
            otherwise: Default value returned when *key* is absent and
                       *error* is ``False``.
            error: If ``True``, raise ``ValueError`` when *key* is missing.

        Returns:
            The stored tensor, or *otherwise* if not found.

        Raises:
            ValueError: When *error* is ``True`` and *key* is not present.
        """
        if error and key not in self._cont:
            raise ValueError(f"{key} not found in {self.__class__}._cont")
        return self._cont.get(key, otherwise)

    def loss_func(self, o: torch.Tensor, t: torch.Tensor):
        """Compute the loss between model output *o* and target *t*.

        Subclasses must override this method.

        Raises:
            AttributeError: Always, when called on the base class.
        """
        raise AttributeError('loss_func is needed but it is not defined.')

    def loss(self):
        """Compute and cache the loss using ``_cont["output"]`` and ``_cont["target"]``.

        Returns:
            The scalar loss tensor stored at ``_cont["loss"]``.

        Raises:
            ValueError: If ``"output"`` or ``"target"`` are absent from ``_cont``.
        """
        if 'loss' not in self._cont:
            o = self.container('output', error=True)
            t = self.container('target', error=True)
            self._cont['loss'] = self.loss_func(o, t)
        return self._cont['loss']

    def midstate_init(self):
        """Initialize hook state and register hooks for all target layers."""
        self._keeping_hiddens_list = list(); self._handles = list()
        self._hiddens = dict(); self._gradients = dict()

        for t in self.target_layers: self.keep_hidden(t); self.keep_grad(t)

    def _keep_hidden_meta(self, _module, *args, name: Optional[str] = None):
        """Forward hook that stores the layer output in ``_hiddens``."""
        output = args[-1]
        self._hiddens[name] = output

    def keep_hidden(self, name: str, with_kwargs: bool = False):
        """Register a forward hook to capture the output of the named layer.

        Idempotent — calling with the same *name* twice has no effect.

        Args:
            name: Dot-separated attribute path to the target sub-module
                  (e.g. ``"layer4.1"``).
            with_kwargs: Whether to pass keyword arguments to the hook.
        """
        if name in self._keeping_hiddens_list: return
        self._keeping_hiddens_list.append(name)
        f = partial(self._keep_hidden_meta, name=name); item = self
        for n in name.split('.'): item = getattr(item, n)
        h = item.register_forward_hook(f, with_kwargs=with_kwargs)
        self._handles.append(h)

    def _keep_grad_meta(self, _module, *args, name: Optional[str] = None):
        """Forward hook that registers a backward hook to capture output gradients."""
        output = args[-1]
        if not getattr(output, 'requires_grad', False): return

        def _store_grad(grad): self._gradients[name] = grad
        output.register_hook(_store_grad)

    def keep_grad(self, name: str):
        """Register hooks to capture gradients of the named layer's output.

        Args:
            name: Dot-separated attribute path to the target sub-module.
        """
        f = partial(self._keep_grad_meta, name=name); item = self
        for n in name.split('.'): item = getattr(item, n)
        self._handles.append(item.register_forward_hook(f))

    @property
    def cyng_gradients(self) -> dict[str, torch.Tensor]:
        """Run backward and return captured gradients for each target layer.

        Triggers :attr:`cyng_loss` if a loss is not already stored in ``_cont``.

        Returns:
            Dict mapping layer names to detached gradient tensors.
        """
        loss = self.container('loss', self.cyng_loss)
        loss.backward(retain_graph=True)
        return {k: t.clone().detach() for k, t in self._gradients.items()}

    @property
    def cyng_hidden(self) -> dict[str, torch.Tensor]:
        """Return captured intermediate activations for each target layer.

        Returns:
            Dict mapping layer names to detached activation tensors.
        """
        return {k: t.clone().detach() for k, t in self._hiddens.items()}
    
    # visual items ###########################################################    
    #def text_processing_init(self, tokenizer: str):
    #    self._tokenizer = Tokenizer(pathlib.Path(tokenizer))



    


