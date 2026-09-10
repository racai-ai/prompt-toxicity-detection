"""Device utilities for selecting the best available PyTorch accelerator."""

from __future__ import annotations

import torch

_selected_device: torch.device | None = None


def _mps_is_available() -> bool:
    return bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )


def _xpu_is_available() -> bool:
    return bool(
        hasattr(torch, "xpu")
        and hasattr(torch.xpu, "is_available")
        and torch.xpu.is_available()
    )


def _normalize_device_spec(device: str | None) -> str:
    if device is None:
        return "auto"

    spec = device.strip().lower()
    if spec in {"", "auto"}:
        return "auto"
    if spec == "rocm":
        return "cuda"
    if spec.startswith("rocm:"):
        return f"cuda:{spec.split(':', 1)[1]}"
    return spec


def list_available_devices() -> list[str]:
    """Return the available runtime devices in preference order."""
    available: list[str] = []
    if torch.cuda.is_available():
        available.append("cuda")
    if _mps_is_available():
        available.append("mps")
    if _xpu_is_available():
        available.append("xpu")
    available.append("cpu")
    return available


def auto_detect_device() -> torch.device:
    """Return the best available device after scanning supported backends."""
    return torch.device(list_available_devices()[0])


def _is_device_available(device: torch.device) -> bool:
    if device.type == "cpu":
        return True
    if device.type == "cuda":
        return torch.cuda.is_available()
    if device.type == "mps":
        return _mps_is_available()
    if device.type == "xpu":
        return _xpu_is_available()
    return False


def _activate_device(device: torch.device) -> None:
    if device.type == "cuda" and device.index is not None:
        device_count = torch.cuda.device_count()
        if device.index < 0 or device.index >= device_count:
            raise ValueError(
                f"Requested CUDA device index {device.index} is out of range "
                f"for {device_count} available device(s)."
            )
        torch.cuda.set_device(device)
    elif (
        device.type == "xpu"
        and device.index is not None
        and hasattr(torch, "xpu")
        and hasattr(torch.xpu, "set_device")
    ):
        if hasattr(torch.xpu, "device_count"):
            device_count = torch.xpu.device_count()
            if device.index < 0 or device.index >= device_count:
                raise ValueError(
                    f"Requested XPU device index {device.index} is out of range "
                    f"for {device_count} available device(s)."
                )
        torch.xpu.set_device(device)


def set_device(device: str | None) -> torch.device:
    """Set the runtime device from a CLI/user-provided string."""
    global _selected_device

    normalized = _normalize_device_spec(device)
    try:
        selected = auto_detect_device() if normalized == "auto" else torch.device(normalized)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Unsupported device specification: {device!r}") from exc

    if not _is_device_available(selected):
        available = ", ".join(list_available_devices())
        raise ValueError(
            f"Requested device '{device}' is not available. "
            f"Available devices: {available}."
        )

    _activate_device(selected)
    _selected_device = selected
    return selected


def get_device() -> torch.device:
    """Return the configured device, or auto-detect one if not set."""
    global _selected_device
    if _selected_device is None:
        _selected_device = auto_detect_device()
        _activate_device(_selected_device)
    return _selected_device


def get_training_arguments_device_kwargs() -> dict[str, bool]:
    """Return TrainingArguments kwargs that respect the selected device."""
    device = get_device()
    if device.type == "cpu":
        return {"use_cpu": True}
    if device.type == "mps":
        return {"use_mps_device": True}
    return {}


def log_device() -> None:
    """Print the selected device and the detected runtime options."""
    device = get_device()
    available = ", ".join(list_available_devices())
    print(f"[device] Using {device}")
    print(f"[device] Available devices: {available}")
    if device.type == "cuda":
        try:
            current_device = torch.cuda.current_device()
            print(f"[device] GPU: {torch.cuda.get_device_name(current_device)}")
        except RuntimeError:
            pass
