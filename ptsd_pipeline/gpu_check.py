from typing import Tuple

import torch


def GPU_checker() -> Tuple[str, bool]:
    """
    Checks whether a CUDA-compatible GPU is available for PyTorch.

    The function verifies if CUDA is available on the current system.
    If a GPU is available, it returns the CUDA device name along with True.
    Otherwise, it returns "cpu" and False.

    :param: None
    :return: A tuple containing the device name and availability status.
    :rtype: Tuple[str, bool]
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"cuda - {device_name}", True
    return "cpu", False


def get_device() -> torch.device:
    """
    Returns the best available torch.device for use in model training/inference.
    Prints a one-line status message so the active device is always visible.

    Usage in any script::

        from gpu_check import get_device
        device = get_device()
        model = model.to(device)

    :return: torch.device ("cuda" or "cpu")
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[device] GPU detected — {gpu_name} (CUDA {torch.version.cuda})")
        return torch.device("cuda")
    print("[device] No GPU found — running on CPU.")
    return torch.device("cpu")


if __name__ == "__main__":
    device, has_gpu = GPU_checker()
    print(f"Device : {device}")
    print(f"GPU available: {has_gpu}")

    print()
    torch_device = get_device()
    print(f"torch.device : {torch_device}")
