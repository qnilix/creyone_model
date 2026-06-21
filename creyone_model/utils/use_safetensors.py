#Copyright 2026 Rinka Kiriyama。
#Licensed under the MIT License (MIT). 

from typing import Iterable
from functools import partial

try:
    import safetensors.torch
    AVAIL = True
except ImportError:
    AVAIL = False


try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    hf_hub_download = partial(hf_hub_download, library_name="timm")#, library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


# Default name for a weights file hosted on the Huggingface Hub.
HF_WEIGHTS_NAME = "pytorch_model.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "model.safetensors"  # safetensors version
HF_OPEN_CLIP_WEIGHTS_NAME = "open_clip_pytorch_model.bin"  # default pytorch pkl
HF_OPEN_CLIP_SAFE_WEIGHTS_NAME = "open_clip_model.safetensors"  # safetensors version


def load_file(file, device = 'cpu'):
    return safetensors.torch.load_file(file, device = device)


def load_stdt_from_safehf(repo_id, filename, revision):
    for safe_filename in _get_safe_alternatives(filename):
        try:
            cached_safe_file = hf_hub_download(repo_id=repo_id, filename=safe_filename, revision=revision)
            return load_file(cached_safe_file, device="cpu")
        except EntryNotFoundError:
            pass
    return None


def _get_safe_alternatives(filename: str) -> Iterable[str]:
    """Returns potential safetensors alternatives for a given filename.

    Use case:
        When downloading a model from the Huggingface Hub, we first look if a .safetensors file exists and if yes, we use it.
        Main use case is filename "pytorch_model.bin" => check for "model.safetensors" or "pytorch_model.safetensors".
    """
    if filename == HF_WEIGHTS_NAME: yield HF_SAFE_WEIGHTS_NAME
    if filename == HF_OPEN_CLIP_WEIGHTS_NAME: yield HF_OPEN_CLIP_SAFE_WEIGHTS_NAME
    if filename.endswith(".bin"): yield filename[:-4] + ".safetensors"