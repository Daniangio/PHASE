"""Utilities for persisting descriptor dictionaries as NPZ files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


FeatureDict = Dict[str, np.ndarray]


def save_descriptor_npz(path: Path, features: FeatureDict) -> None:
    """
    Persists a residue feature dictionary into a compressed NPZ archive.

    Each residue key becomes an array entry inside the NPZ. Metadata such as
    residue mappings live next to the NPZ (e.g., system.json) to keep the file
    portable.
    """
    if not features:
        raise ValueError("Cannot persist an empty descriptor dictionary.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def load_descriptor_npz(path: Path) -> FeatureDict:
    """Loads a descriptor NPZ archive back into a residue dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Descriptor file '{path}' not found.")
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}

