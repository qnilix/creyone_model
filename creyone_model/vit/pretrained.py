from .weight import TimmViTFilter, CLIPFilter

TRANS_IMAGE = {'interpolation': 'bicubic', 'crop_pct': 0.9, 'crop_mode': 'center'}
IMAGE_DEFAULT = {'image_mean': (0.5, 0.5, 0.5), 'image_std': (0.5, 0.5, 0.5)}

PRET_CFGS2 = {
    'vit-base-patch16-224.google_bert': {
        'pretrained': {
            'hf_hub_id': 'google-bert/bert-base-uncased',
            'hf_hub_filename': 'pytorch_model.bin'
        }
    },
    'vit-base-patch16-224.timm_openai_clip': {
        'pretrained': {
            'hf_hub_id': 'timm/vit_base_patch16_clip_224.openai',
            'hf_hub_filename': 'pytorch_model.bin'
        }
    }
}

def hub_pretrained(id: str):
    return {'hf_hub_id': id, 'hf_hub_weights': 'pytorch_model.bin'}


def augreg(item: dict, keep_head: bool = False):
    item['stdt_filter'] = TimmViTFilter(keep_head=keep_head)
    item['transformer'] = {'post_norm': True, 'block': {'attn_bias': 'qkvo'}}
    item.update(TRANS_IMAGE)
    item.update(IMAGE_DEFAULT)
    return item


def augreg_in21k(size: str, patch: int, res: int, keep_head: bool = False):
    _hub = f'timm/vit_{size}_patch{patch}_{res}.augreg_in21k'
    item = augreg({'pretrained': hub_pretrained(_hub)})
    return item


def augreg_in21k_ft_in1k(size: str, patch: int, res: int, keep_head: bool = False):
    _hub = f'timm/vit_{size}_patch{patch}_{res}.augreg_in21k_ft_in1k'
    item = augreg({'pretrained': hub_pretrained(_hub)}, keep_head=keep_head)
    return item


def clip_tformers(size: str, patch: int, res: int, keep_head: bool = False):
    _hub = f'openai/clip-vit-{size}-patch{patch}'
    pret = hub_pretrained(_hub); pret['source'] = 'transformers'
    item = {'pretrained': pret, 'force_fc_norm': True}
    item['stdt_filter'] = CLIPFilter(keep_head=keep_head)
    item['transformer'] = {'pre_norm': True, 'post_norm': False, 'block': {'attn_bias': 'qkvo', 'act_name': 'quickgelu'}, 'norm_eps': 1e-5}
    item['patch_embed.bias'] = False
    item.update(TRANS_IMAGE)
    item.update(IMAGE_DEFAULT)
    return item


def pret_cfgs(keep_head: bool = False):
    item = dict()
    for k in ['tiny', 'small', 'base', 'large']:
        item[f"vit-{k}-patch16-224.augreg_in21k"] = augreg_in21k(k, 16, 224, keep_head=keep_head)
    item.update(
        {
            'vit-base-patch16-224.tformers_openai_clip': clip_tformers('base', 16, 224, keep_head=keep_head),
            'vit-base-patch16-224.augreg_in21k_ft_in1k': augreg_in21k_ft_in1k('base', 16, 224, keep_head=keep_head)
        }
    )
    return item 
