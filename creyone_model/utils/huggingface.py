import os, hashlib, json
from typing import Optional, Union
from pathlib import Path

import torch.nn as nn

from torch.hub import HASH_REGEX, download_url_to_file, urlparse

try:
    from torch.hub import get_dir
except ImportError:
    from torch.hub import _get_torch_home as get_dir

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False



from PIL import Image as PILImage

from . import use_safetensors
from .helper import load_ckpt

HF_WEIGHTS_NAME = "pytorch_model.bin"  # default pytorch pkl
NO_HUGHUB = 'Hugging Face hub model specified but package not installed. \
    Run `pip install huggingface_hub`.'


def load_json(path: Union[str, Path]) -> dict:
    if isinstance(path, str): path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"{path} is not founc")
    try:
        item = json.loads(path.read_text())
    except json.JSONDecodeError:
        msg = f"It looks like the file at '{path}' is not a valid JSON file."
        raise OSError(msg)
    return item


def get_cache_dir(child_dir: tuple[str] = ()) -> Path:
    """
    Returns the location of the directory where models are cached (and creates it if necessary).
    """
    model_dir = Path(get_dir()).joinpath('checkpoints', *child_dir)
    model_dir.mkdir(exist_ok=True)
    return model_dir


def cached_file_tools(url: list | tuple, check_hash: bool = True) -> tuple[str, Path, Optional[str]]:
    flag = isinstance(url, (list, tuple))
    url, file = url if flag else (url, Path(urlparse(url).path).name)

    hash_prefix = None
    if check_hash:
        r = HASH_REGEX.search(file)  # r is Optional[Match[str]]
        if r: hash_prefix = r.group(1)

    return url, get_cache_dir().joinpath(file), hash_prefix


def check_cached_file(url, check_hash: bool = True) -> bool:
    _, file, prex = cached_file_tools(url, check_hash=check_hash)

    if not file.exists(): return False
    if prex is not None:
        with open(file, 'rb') as f:
            hd = hashlib.sha256(f.read()).hexdigest()
            if hd[:len(prex)] != prex: return False
    return True


def download_cached_file(url, check_hash=True, progress=False) -> Path:
    url, file, prex = cached_file_tools(url, check_hash=check_hash)
    if file.exists(): return file
    download_url_to_file(url, file, prex, progress=progress)
    return file


def has_hf_hub(necessary: bool = False) -> bool:
    if _has_hf_hub or not necessary: return _has_hf_hub
    raise RuntimeError(NO_HUGHUB)


def hf_split(hf_id: str):
    # FIXME I may change @ -> # and be parsed as fragment in a URI model name scheme
    rev_split = hf_id.split('@')
    assert 0 < len(rev_split) <= 2, 'hf_hub id should only contain one @ character to identify revision.'
    hf_model_id = rev_split[0]
    hf_revision = rev_split[-1] if len(rev_split) > 1 else None
    return hf_model_id, hf_revision


def download_from_hf(model_id: str, name: str, library: str = 'timm', raise_error = True):
    hf_model_id, hf_revision = hf_split(model_id)
    if raise_error:
        return hf_hub_download(hf_model_id, name, library_name=library, revision=hf_revision)
    try:
        return hf_hub_download(hf_model_id, name, library_name=library, revision=hf_revision)
    except EntryNotFoundError:
        return None


def load_model_config_from_hf(model_id: str):
    assert has_hf_hub(True)
    cached_file = download_from_hf(model_id, 'config.json')

    hf_config = load_json(cached_file)
    if 'pretrained_cfg' not in hf_config:
        # old form, pull pretrain_cfg out of the base dict
        pretrained_cfg = hf_config
        hf_config = {}
        hf_config['architecture'] = pretrained_cfg.pop('architecture')
        hf_config['num_features'] = pretrained_cfg.pop('num_features', None)
        if 'labels' in pretrained_cfg:  # deprecated name for 'label_names'
            pretrained_cfg['label_names'] = pretrained_cfg.pop('labels')
        hf_config['pretrained_cfg'] = pretrained_cfg

    # NOTE currently discarding parent config as only arch name and pretrained_cfg used in timm right now
    pretrained_cfg = hf_config['pretrained_cfg']
    pretrained_cfg['hf_hub_id'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'

    # model should be created with base config num_classes if its exist
    if 'num_classes' in hf_config:
        pretrained_cfg['num_classes'] = hf_config['num_classes']

    # label meta-data in base config overrides saved pretrained_cfg on load
    if 'label_names' in hf_config:
        pretrained_cfg['label_names'] = hf_config.pop('label_names')
    if 'label_descriptions' in hf_config:
        pretrained_cfg['label_descriptions'] = hf_config.pop('label_descriptions')

    model_args = hf_config.get('model_args', {})
    model_name = hf_config['architecture']
    return pretrained_cfg, model_name, model_args


def load_state_dict_from_hf(
        model_id: str,
        filename: str = HF_WEIGHTS_NAME,
        weights_only: bool = False,
):
    assert has_hf_hub(True)
    hf_model_id, hf_revision = hf_split(model_id)

    if use_safetensors.AVAIL:
        ckpt = use_safetensors.load_stdt_from_safehf(
            hf_model_id, filename=filename, revision=hf_revision)
        if ckpt: return ckpt
        
    # Otherwise, load using pytorch.load
    cached_file = hf_hub_download(hf_model_id, filename=filename, revision=hf_revision)
    return load_ckpt(cached_file, weights_only=weights_only)


def load_custom_from_hf(model_id: str, filename: str, model: nn.Module):
    assert has_hf_hub(True)
    hf_model_id, hf_revision = hf_split(model_id)
    cached_file = hf_hub_download(hf_model_id, filename=filename, revision=hf_revision)
    return model.load_pretrained(cached_file)


def parse_imgconf(size: int, crop_size: int, resample: int = 2,
                  do_center_crop: bool = True, do_resize: bool = True,
                  do_normalize: bool = True, feature_extractor_type: str = None, **kwargs):
    item = dict()
    if do_center_crop:
        pct = None if do_resize else size / crop_size
        item.update({'size': (crop_size, crop_size), 'crop_pct': pct, 'crop_mode': 'center'})
    if isinstance(resample, int):
        item['interpolation'] = interpolation_to_str(resample)
    if do_normalize:
        for k in ('image_mean', 'image_std'): item[k] = kwargs.pop(k)
    return item, kwargs


def interpolation_to_str(id: int):
    if id == PILImage.NEAREST: return 'nearest'
    if id == PILImage.LANCZOS: return 'lanczos'
    if id == PILImage.BILINEAR: return 'bilinear'
    if id == PILImage.BICUBIC: return 'bicubic'
    if id == PILImage.BOX: return 'box'

