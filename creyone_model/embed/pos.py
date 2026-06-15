from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from creyone_layer.init import init_linear


class PosEmbed(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, num_embeddings, embedding_dim))
    
    def forward(self, x: torch.Tensor, **_) -> torch.Tensor:
        return x + self.weight
    
    def trainable_parameters(self, mode: str = 'all'):
        self.requires_grad_(mode == 'all')
    
    def reset_parameters(self, mode: str = 'trunc_', std: float=.02):
        self.apply(init_linear(mode=mode, std=std))


class PosEmbedV2(nn.Embedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("pos_ids", torch.arange(self.num_embeddings).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor, 
                positions: Optional[torch.Tensor] = None, 
                start: int = 2, **_):
        if positions is None: positions = self.pos_ids + start
        return x + F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
    
    def reset_parameters(self, mode: str = 'trunc_', std: float=.02):
        self.apply(init_linear(mode=mode, std=std))
    
    def trainable_parameters(self, mode: str = 'all'):
        self.requires_grad_(mode == 'all')