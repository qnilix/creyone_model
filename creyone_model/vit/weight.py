import re
from collections import defaultdict

import torch
from torch import nn


class BaseFilter:

    @property
    def prompt_name(self) -> str: return 'prompts'

    @property
    def posemb_name(self) -> str: return 'embed.pos.weight'

    def __call__(self, stdt: dict[str, torch.Tensor], model: nn.Module) -> dict:
        prompt = stdt.pop(self.prompt_name, None)
        posemb = stdt.pop(self.posemb_name, None)
        stdt['prompts'] = prompt
        stdt['embed.pos.weight'] = posemb
        return stdt


class TimmViTFilter(BaseFilter):

    def __init__(self, keep_head: bool = False):
        self.keep_head = keep_head

    @property
    def prompt_name(self) -> str: return 'cls_token'

    @property
    def posemb_name(self) -> str: return 'pos_embed'

    def __call__(self, stdt: dict, model: nn.Module) -> dict:
        super().__call__(stdt, model)

        head = dict()
        keys = [k for k in stdt.keys()]
        for k in keys:
            assert isinstance(k, str)
            if k.startswith('head'): head[k] = stdt.pop(k)
            if k.startswith('blocks'): stdt.update(self.blocks(k, stdt.pop(k)))

        if not self.keep_head:
            param = model.named_parameters()
            head = {k: v for k, v in param if k.startswith('head')}
        
        for k, v in head.items():
            s = k.split('.')[-1]
            stdt[f'head.linear.{s}'] = v

        for k, v in model.named_parameters():
            s = k.split('.')
            if s[-1].startswith('adw'): stdt[k] = v
            if len(s) > 1 and s[-2].startswith('adw'):
                if s[-1] in ('weight', 'bias'): stdt[k] = v
        stdt.update(model.unchanged_param(model))
        stdt.update(model.modify_param(stdt))
        return stdt
    
    def blocks(self, old_key: str, weight: torch.Tensor):
        new_key = re.sub(r's\.', '', old_key, 1)
        if not re.search(r'qkv\.(weight|bias)', new_key):
            return {re.sub(r'attn.p', r'attn.o_p', new_key): weight}
        ret = dict()
        for key, val in zip('qkv', weight.split(weight.shape[0] // 3)):
            ret[re.sub(r'qkv', f'{key}_proj', new_key)] = val
        return ret


class CLIPFilter(BaseFilter):

    def __init__(self, keep_head: bool = False):
        self.keep_head = keep_head

    @property
    def prompt_name(self) -> str: return 'embeddings.class_embedding'

    @property
    def posemb_name(self) -> str: return 'embeddings.position_embedding.weight'

    def __call__(self, stdt: dict, model: nn.Module) -> dict:
        head = {'head.linear.weight': stdt.pop('visual_projection.weight')}
        stdt = dict((k.replace('vision_model.', ''), v) for k, v in stdt.items() if k.startswith('vision_model'))
        stdt['prompts'] = stdt.pop(self.prompt_name, None)[None, None, :]
        stdt['embed.pos.weight'] = stdt.pop(self.posemb_name, None)[None, :, :]
        pemb_w = stdt.pop('embeddings.patch_embedding.weight')
        stdt.pop('embeddings.position_ids')
        stdt['patch_embed.proj.weight'] = pemb_w

        keys = [k for k in stdt.keys()]
        for k in keys:
            assert isinstance(k, str)
            if k.startswith('pre_layrnorm'): stdt[k.replace('pre_layrnorm', 'norm_pre')] = stdt.pop(k)
            if k.startswith('post_layernorm'): head[k.replace('post_layernorm', 'head.fc_norm')] = stdt.pop(k)
            if k.startswith('encoder.layers'):
                new_k = re.sub(r'encoder\.layers\.', 'block', k, 1).replace('self_attn', 'attn')
                new_k = new_k.replace('attn.out', 'attn.o').replace('layer_n', 'n')
                stdt[new_k] = stdt.pop(k)

        if not self.keep_head:
            param = model.named_parameters()
            head.update({k: v for k, v in param if k.startswith('head.linear')})
        
        for k, v in head.items(): stdt[k] = v
        return stdt


