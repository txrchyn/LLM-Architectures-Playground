import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch


def load_config_from_py_file(config_path: str) -> Any:
    """
    Load a configuration dataclass from a Python file.
    The file must expose a top-level variable named `config`.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("benchmark_config_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_config_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "config"):
        raise AttributeError(f"`config` variable not found inside {config_path}")

    return module.config


def resolve_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}'. Supported values: {list(mapping.keys())}")
    return mapping[key]


def resolve_device(device: str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
