"""
utils.py — Shared utilities for the ORPO alignment pipeline.

Provides:
  - YAML config loading
  - Reproducibility (seed everything)
  - GPU diagnostics
  - Memory monitoring
  - Logging setup
  - Kaggle environment detection
"""

import os
import sys
import random
import logging
import platform
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary with all configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.info(f"[OK] Config loaded from: {config_path}")
    return config


def get_config_value(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    Retrieve a nested config value using dot notation.

    Example:
        get_config_value(config, "training.learning_rate", 5e-6)

    Args:
        config: The configuration dictionary.
        dotted_key: Dot-separated key path (e.g., "model.name").
        default: Default value if key is not found.

    Returns:
        The config value or the default.
    """
    keys = dotted_key.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# ═══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"[OK] Random seed set to: {seed}")


# ═══════════════════════════════════════════════════════════════════════════════
# GPU Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def print_gpu_info() -> Dict[str, Any]:
    """
    Print and return detailed GPU information.
    Validates compatibility with the pipeline requirements.

    Returns:
        Dictionary with GPU info (name, vram, compute_capability, etc.)
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
    }

    if not torch.cuda.is_available():
        logging.warning("[WARN] CUDA is NOT available. Training will fail.")
        return info

    info["device_count"] = torch.cuda.device_count()
    info["cuda_version"] = torch.version.cuda
    info["pytorch_version"] = torch.__version__

    print("=" * 60)
    print("GPU DIAGNOSTICS")
    print("=" * 60)
    print(f"  PyTorch version  : {torch.__version__}")
    print(f"  CUDA version     : {torch.version.cuda}")
    print(f"  Device count     : {torch.cuda.device_count()}")
    print("-" * 60)

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        # Use mem_get_info for cross-version compatibility (PyTorch 1.10+)
        _free_mem, _total_mem = torch.cuda.mem_get_info(i)
        vram_gb = _total_mem / (1024 ** 3)
        cc = f"{props.major}.{props.minor}"

        device_info = {
            "index": i,
            "name": props.name,
            "vram_gb": round(vram_gb, 1),
            "compute_capability": cc,
            "supports_bf16": props.major >= 8,
            "supports_fp16": True,
            "has_tensor_cores": props.major >= 7,
        }
        info["devices"].append(device_info)

        print(f"  GPU {i}: {props.name}")
        print(f"    VRAM           : {vram_gb:.1f} GB")
        print(f"    Compute Cap.   : {cc}")
        print(f"    Tensor Cores   : {'[YES]' if props.major >= 7 else '[NO]'}")
        print(f"    FP16 support   : [YES]")
        print(f"    BF16 support   : {'[YES]' if props.major >= 8 else '[NO] (use fp16)'}")

        # Compatibility warnings
        if props.major < 7:
            logging.warning(
                f"[WARN] GPU {i} ({props.name}) has SM {cc} -- Unsloth requires SM 7.0+. "
                "Training will likely fail. Switch to T4 or newer GPU."
            )
        if props.major >= 7 and props.major < 8:
            logging.info(
                f"[INFO] GPU {i} ({props.name}) supports fp16 but NOT bf16. "
                "Using fp16=True in training config."
            )

    print("=" * 60)
    return info


def get_vram_usage() -> Dict[str, float]:
    """
    Get current VRAM usage for the primary GPU.

    Returns:
        Dictionary with allocated, reserved, and free VRAM in GB.
    """
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "free_gb": 0.0}

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    # Use mem_get_info for cross-version compatibility
    _free_mem, _total_mem = torch.cuda.mem_get_info(0)
    total = _total_mem / (1024 ** 3)
    free = total - allocated

    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(free, 2),
    }


def print_vram_usage(label: str = "") -> None:
    """Print a one-line VRAM usage summary, optionally with a label."""
    usage = get_vram_usage()
    prefix = f"[{label}] " if label else ""
    print(
        f"  {prefix}VRAM: {usage['allocated_gb']:.2f} GB allocated / "
        f"{usage['total_gb']:.1f} GB total "
        f"({usage['free_gb']:.2f} GB free)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Kaggle Environment
# ═══════════════════════════════════════════════════════════════════════════════

def is_kaggle() -> bool:
    """Check if running inside a Kaggle notebook environment."""
    return os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def get_kaggle_secret(secret_name: str) -> Optional[str]:
    """
    Retrieve a Kaggle secret (e.g., HF_TOKEN, WANDB_API_KEY).

    Args:
        secret_name: Name of the Kaggle secret.

    Returns:
        The secret value, or None if not found / not on Kaggle.
    """
    if not is_kaggle():
        # Fall back to environment variable
        return os.environ.get(secret_name)

    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        secret = client.get_secret(secret_name)
        if secret:
            logging.info(f"[OK] Kaggle secret '{secret_name}' loaded successfully.")
        return secret
    except Exception as e:
        logging.warning(f"[WARN] Could not load Kaggle secret '{secret_name}': {e}")
        return os.environ.get(secret_name)


def setup_hf_token() -> None:
    """Configure HuggingFace token from Kaggle secrets or environment."""
    token = get_kaggle_secret("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        logging.info("[OK] HuggingFace token configured.")
    else:
        logging.warning(
            "[WARN] HF_TOKEN not found. Some gated models may not be accessible. "
            "Add it as a Kaggle Secret or set the HF_TOKEN env variable."
        )


def setup_wandb(config: Dict[str, Any]) -> None:
    """
    Configure WandB from config and Kaggle secrets.

    Args:
        config: The full pipeline config dictionary.
    """
    wandb_config = config.get("wandb", {})

    if not wandb_config.get("enabled", False):
        os.environ["WANDB_DISABLED"] = "true"
        logging.info("[INFO] WandB disabled via config.")
        return

    api_key = get_kaggle_secret("WANDB_API_KEY")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        logging.info("[OK] WandB API key configured.")
    else:
        logging.warning(
            "[WARN] WANDB_API_KEY not found. WandB will prompt for login or run offline. "
            "Add it as a Kaggle Secret to enable cloud logging."
        )
        os.environ["WANDB_MODE"] = "offline"


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for the pipeline.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to a file.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.info("[OK] Logging initialized.")


# ═══════════════════════════════════════════════════════════════════════════════
# Directory Management
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_dirs(config: Dict[str, Any]) -> None:
    """
    Create all output directories specified in the config.

    Args:
        config: The full pipeline config dictionary.
    """
    dirs_to_create = [
        config.get("data", {}).get("output_dir", "/kaggle/working/processed_data"),
        config.get("training", {}).get("output_dir", "/kaggle/working/orpo_checkpoints"),
        config.get("saving", {}).get("lora_output_dir", "/kaggle/working/orpo_lora_adapters"),
        config.get("saving", {}).get("merged_output_dir", "/kaggle/working/orpo_merged_model"),
        config.get("evaluation", {}).get("output_dir", "/kaggle/working/eval_results"),
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logging.debug(f"[OK] Ensured directory: {dir_path}")

    logging.info(f"[OK] Created {len(dirs_to_create)} output directories.")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Initialization (convenience wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_pipeline(config_path: str) -> Dict[str, Any]:
    """
    One-call initialization: load config, set seed, setup logging,
    configure tokens, print GPU info, create directories.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        The loaded configuration dictionary.
    """
    # 1. Load config
    config = load_config(config_path)

    # 2. Setup logging
    setup_logging(log_level="INFO")

    # 3. Set seed
    seed = get_config_value(config, "data.seed", 42)
    set_seed(seed)

    # 4. GPU diagnostics
    gpu_info = print_gpu_info()

    # 5. Validate GPU compatibility
    if gpu_info["cuda_available"] and gpu_info["devices"]:
        primary_gpu = gpu_info["devices"][0]
        if not primary_gpu["has_tensor_cores"]:
            logging.error(
                "[ERROR] Primary GPU lacks Tensor Cores. Unsloth will not work. "
                "Switch to T4 or newer GPU in Kaggle Settings > Accelerator."
            )
            raise RuntimeError("Incompatible GPU for Unsloth. Requires SM 7.0+.")

    # 6. Setup tokens
    setup_hf_token()
    setup_wandb(config)

    # 7. Create output directories
    ensure_dirs(config)

    # 8. Environment summary
    print("\n" + "=" * 60)
    print("PIPELINE INITIALIZED")
    print("=" * 60)
    print(f"  Environment      : {'Kaggle' if is_kaggle() else 'Local'}")
    print(f"  Python           : {platform.python_version()}")
    print(f"  Model            : {get_config_value(config, 'model.name')}")
    print(f"  LoRA rank        : {get_config_value(config, 'lora.r')}")
    print(f"  Precision        : {'fp16' if get_config_value(config, 'training.fp16') else 'fp32'}")
    print(f"  Batch size (eff) : {get_config_value(config, 'training.per_device_train_batch_size', 2)} × "
          f"{get_config_value(config, 'training.gradient_accumulation_steps', 4)} = "
          f"{get_config_value(config, 'training.per_device_train_batch_size', 2) * get_config_value(config, 'training.gradient_accumulation_steps', 4)}")
    print(f"  Max seq length   : {get_config_value(config, 'training.max_seq_length')}")
    print(f"  WandB            : {'enabled' if get_config_value(config, 'wandb.enabled') else 'disabled'}")
    print("=" * 60 + "\n")

    return config
