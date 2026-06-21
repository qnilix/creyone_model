#Copyright 2026 Rinka Kiriyama。
#Licensed under the MIT License (MIT). 

import re, os
from urllib.parse import urlsplit
from typing import Optional


def split_model_name_tag(model_name: str, no_tag: str = '') -> tuple[str, str]:
    """Split a model name string into its base name and optional tag.

    Args:
        model_name: Model name string, optionally with a dot-separated tag
                    (e.g. ``"resnet50.my_tag"``).
        no_tag: Default tag value returned when no tag is present.

    Returns:
        A ``(model_name, tag)`` tuple.
    """
    model_name, *tag_list = model_name.split('.', 1)
    if len(tag_list) == 0: tag_list.append(no_tag)
    return model_name, tag_list[0]


def parse_model_name(model_name: str) -> tuple[str, str]:
    """Parse a model name string and determine its source.

    Accepts plain names, ``timm://`` URIs, and ``hf-hub``/``hf_hub`` URIs.
    URI format: ``scheme://netloc/path;parameters?query#fragment``

    Args:
        model_name: Raw model name or URI string.

    Returns:
        A ``(source, name)`` tuple where *source* is one of
        ``"hf-hub"`` or ``"creyone"``.
    """
    parsed = urlsplit(re.sub(r'^hf_hub', 'hf-hub', model_name))
    # FIXME may use fragment as revision, currently `@` in URI path
    if parsed.scheme == 'hf-hub': return parsed.scheme, parsed.path
    assert parsed.scheme in ('', 'timm')
    return 'creyone', os.path.split(parsed.path)[-1]


def parse_pretrained_cfg(model_name, model_source = '', pretrained_cfg: Optional[str] = None):
    """Resolve the pretrained-config tag for a given model.

    For ``"creyone"`` models the tag is extracted from the dot-suffix of
    *model_name*; for all other sources an empty string is returned.

    Args:
        model_name: Base model name, potentially with a dot-separated tag.
        model_source: Source identifier (e.g. ``"creyone"`` or ``"hf-hub"``).
        pretrained_cfg: Explicit config tag that overrides the extracted tag.

    Returns:
        The resolved pretrained-config tag, or ``""`` when not applicable.
    """
    if model_source == 'creyone':
        model_name, tag = split_model_name_tag(model_name)
        return pretrained_cfg or tag
    return ''