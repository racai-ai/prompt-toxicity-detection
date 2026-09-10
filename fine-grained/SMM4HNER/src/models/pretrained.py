"""Helpers for resolving pretrained model identifiers and local disk paths."""

import json
import os


_LOCAL_MODEL_SCHEMES = ("disk://", "file://")


def resolve_model_name_or_path(model_name: str) -> str:
    """Resolve a model identifier into a local path when needed.

    Parameters
    ----------
    model_name: str
        Model identifier, either a standard Hugging Face name, a plain local
        path, or a ``disk://`` / ``file://`` URI.

    Returns
    -------
    str
        The original model identifier with any supported local URI prefix
        removed and ``~`` expanded.
    """
    for scheme in _LOCAL_MODEL_SCHEMES:
        if model_name.startswith(scheme):
            model_name = model_name[len(scheme) :]
            break
    return os.path.expanduser(model_name)


def is_local_model_path(model_name: str) -> bool:
    """Return whether *model_name* resolves to an existing local directory.

    Parameters
    ----------
    model_name: str
        Model identifier to resolve and inspect.

    Returns
    -------
    bool
        ``True`` when the resolved value is a directory on the local
        filesystem, otherwise ``False``.
    """
    return os.path.isdir(resolve_model_name_or_path(model_name))


def resolve_adapter_base_model_name_or_path(model_name: str) -> str:
    """Resolve the backbone source for an adapter checkpoint.

    For local adapter directories, prefer the ``base_model_name_or_path`` stored
    in ``adapter_config.json`` so callers can load the backbone/tokenizer even
    when the directory only contains adapter artifacts.
    """
    resolved_model_name = resolve_model_name_or_path(model_name)
    if not os.path.isdir(resolved_model_name):
        return resolved_model_name

    adapter_config_path = os.path.join(resolved_model_name, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return resolved_model_name

    try:
        with open(adapter_config_path, "r", encoding="utf-8") as fh:
            adapter_config = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return resolved_model_name

    # Adapter checkpoints written by different adapters library versions may use
    # either the newer ``base_model_name_or_path`` key or the older
    # ``model_name`` key for the backbone reference.
    base_model_name_or_path = (
        adapter_config.get("base_model_name_or_path")
        or adapter_config.get("model_name")
    )
    if not isinstance(base_model_name_or_path, str) or not base_model_name_or_path.strip():
        return resolved_model_name

    # Normalize any supported local-path prefix stored in the adapter config so
    # callers receive the actual filesystem path for local backbones.
    return resolve_model_name_or_path(base_model_name_or_path)
