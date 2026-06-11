import re, os
from urllib.parse import urlsplit
from typing import Optional


def split_model_name_tag(model_name: str, no_tag: str = '') -> tuple[str, str]:
    model_name, *tag_list = model_name.split('.', 1)
    if len(tag_list) == 0: tag_list.append(no_tag)
    return model_name, tag_list[0]


def parse_model_name(model_name: str) -> tuple[str, str]:
    """
    scheme://netloc/path;parameters?query#fragment
    """
    parsed = urlsplit(re.sub(r'^hf_hub', 'hf-hub', model_name))
    # FIXME may use fragment as revision, currently `@` in URI path
    if parsed.scheme == 'hf-hub': return parsed.scheme, parsed.path
    assert parsed.scheme in ('', 'timm')
    return 'creyone', os.path.split(parsed.path)[-1]


def parse_pretrained_cfg(model_name, model_source = '', pretrained_cfg: Optional[str] = None):
    if model_source == 'creyone':
        model_name, tag = split_model_name_tag(model_name)
        return pretrained_cfg or tag