import torch
import torch.nn as nn

class ClassificationLoss(nn.Module):

    def __init__(self, bce: bool = False, smoothing: float = 0.0):
        super().__init__()
        fn = nn.BCEWithLogitsLoss() if bce else nn.CrossEntropyLoss(label_smoothing=smoothing)
        self._train_fn = fn
        self._valid_fn = nn.CrossEntropyLoss()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.training: return self._train_fn(output, target)
        return self._valid_fn(output, target)
        