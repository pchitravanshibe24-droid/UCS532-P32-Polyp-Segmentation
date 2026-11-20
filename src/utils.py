"""Utility functions for configuration, seeding, device selection, and logging."""

import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a YAML configuration file.
    
    Args:
        config_path: Path to the YAML file.
        
    Returns:
        Configuration dictionary.
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config format in {path}: expected a dictionary.")
        
    return config


def set_seed(seed: int = 42, *, strict: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic execution.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(preferred: str = "cpu") -> Any:
    """Select the execution device safely.
    
    Defaults to CPU as the guaranteed local path.
    Only selects CUDA if requested and PyTorch confirms CUDA is available.
    
    Args:
        preferred: 'cpu' or 'cuda'.
        
    Returns:
        torch.device instance.
    """
    import torch
    
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    return device


def print_environment_info(config: Optional[Dict[str, Any]] = None) -> None:
    """Print system, Python, PyTorch, and configuration details at startup."""
    import torch
    
    device = get_device(config.get("project", {}).get("device", "cpu") if config else "cpu")
    print("=" * 70)
    print("P32 — Polyp Segmentation with Explicit Pixel-Level Uncertainty")
    print("=" * 70)
    print(f"Python Version:   {sys.version.split()[0]}")
    print(f"PyTorch Version:  {torch.__version__}")
    print(f"Selected Device:  {device}")
    print(f"CPU: {get_cpu_name()} | Torch threads: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Hardware Note:    Running in CPU mode (local default)")
        
    if config:
        project_name = config.get("project", {}).get("name", "Unknown")
        seed = config.get("project", {}).get("seed", 42)
        print(f"Project Name:     {project_name}")
        print(f"Configured Seed:  {seed}")
    print("=" * 70)


def get_cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
