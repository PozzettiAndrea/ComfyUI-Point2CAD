# SPDX-License-Identifier: GPL-3.0-or-later
# Originally from ComfyUI-CADabra: https://github.com/PozzettiAndrea/ComfyUI-CADabra
# Copyright (C) 2025 ComfyUI-CADabra Contributors

"""
Model loader utilities for Point2CAD
Downloads and caches pretrained models from GitHub/Google Drive
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional, Dict

# Point2CAD model URLs and metadata
POINT2CAD_MODELS = {
    "parsenet_with_normals": {
        "url": "https://github.com/prs-eth/point2cad/raw/main/point2cad/logs/pretrained_models/parsenet.pth",
        "filename": "parsenet.pth",
        "description": "ParseNet model trained with normal information"
    },
    "parsenet_no_normals": {
        "url": "https://github.com/prs-eth/point2cad/raw/main/point2cad/logs/pretrained_models/parsenet_no_normals.pth",
        "filename": "parsenet_no_normals.pth",
        "description": "ParseNet model for raw point clouds without normals"
    },
    "hpnet": {
        "url": "https://drive.google.com/uc?export=download&id=1fj84kyD9CGT8j61IW-xSWZ5q4q5IpoYx",
        "filename": "hpnet_abc.pth",
        "description": "HPNet model (highest performance, pretrained on ABC dataset)"
    },
}


def get_models_dir() -> Path:
    """
    Get the models directory for Point2CAD models.
    Creates ComfyUI/models/cadrecon/point2cad/ if it doesn't exist.
    """
    # Navigate up from ComfyUI-Point2CAD/utils/ to ComfyUI/models/cadrecon/point2cad/
    current_dir = Path(__file__).parent.parent  # ComfyUI-Point2CAD/
    comfyui_dir = current_dir.parent.parent  # ComfyUI/custom_nodes/../ = ComfyUI/
    models_dir = comfyui_dir / "models" / "cadrecon" / "point2cad"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_model_path(model_name: str) -> Optional[Path]:
    """
    Get the path to a specific Point2CAD model.

    Args:
        model_name: Name of the model (parsenet_with_normals, parsenet_no_normals, hpnet)

    Returns:
        Path to the model file, or None if not found
    """
    if model_name not in POINT2CAD_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(POINT2CAD_MODELS.keys())}")

    models_dir = get_models_dir()
    filename = POINT2CAD_MODELS[model_name]["filename"]
    model_path = models_dir / filename

    if model_path.exists():
        return model_path
    else:
        return None


def download_file(url: str, destination: Path, description: str = "Downloading") -> bool:
    """Download a file with progress reporting."""
    try:
        print(f"[Download] {description}...")
        print(f"   URL: {url}")
        print(f"   Saving to: {destination}")

        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')

        urllib.request.urlretrieve(url, destination, reporthook)
        print()
        print(f"[OK] Download complete!")
        return True

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        if destination.exists():
            destination.unlink()
        return False


def download_point2cad_model(model_name: str, force_download: bool = False) -> Optional[Path]:
    """
    Download a Point2CAD model from GitHub if not already cached.

    Args:
        model_name: Name of the model to download
        force_download: If True, re-download even if file exists

    Returns:
        Path to the downloaded model file, or None if download failed
    """
    if model_name not in POINT2CAD_MODELS:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"   Available models: {list(POINT2CAD_MODELS.keys())}")
        return None

    model_info = POINT2CAD_MODELS[model_name]
    models_dir = get_models_dir()
    model_path = models_dir / model_info["filename"]

    if model_path.exists() and not force_download:
        print(f"[OK] Model already downloaded: {model_path}")
        return model_path

    print(f"Model not found locally, downloading {model_name}...")
    print(f"   Description: {model_info['description']}")

    success = download_file(
        url=model_info["url"],
        destination=model_path,
        description=f"Downloading {model_name}"
    )

    if success:
        return model_path
    else:
        return None


def list_available_models() -> Dict[str, bool]:
    """
    List all available Point2CAD models and their download status.

    Returns:
        Dictionary mapping model names to whether they are downloaded
    """
    models_status = {}
    for model_name in POINT2CAD_MODELS.keys():
        model_path = get_model_path(model_name)
        models_status[model_name] = model_path is not None
    return models_status
