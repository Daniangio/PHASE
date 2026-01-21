"""
Runner for static state-sensitivity analysis.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from phase.io.readers import MDAnalysisReader
from phase.features.extraction import FeatureExtractor
from phase.pipeline.builder import DatasetBuilder

from phase.analysis.static import StaticStateSensitivity


# ======================================================================
# Utility
# ======================================================================

def _safe_get(d: Dict, key: str, default: float = 0.0) -> float:
    """
    Safely retrieves a float value from a dictionary.
    Handles cases where the key is missing OR the value is None (from JSON null).
    """
    val = d.get(key) if isinstance(d, dict) else None
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def _load_descriptor_features(
    dataset_cfg: Dict[str, Any]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], int, int, Dict[str, str]]:
    descriptor_keys = dataset_cfg.get("descriptor_keys") or []
    if not descriptor_keys:
        raise ValueError("Descriptor keys missing from dataset configuration.")

    def _load(path: str) -> Dict[str, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in descriptor_keys}

    active_features = _load(dataset_cfg["active_descriptors"])
    inactive_features = _load(dataset_cfg["inactive_descriptors"])

    if not active_features or not inactive_features:
        raise ValueError("Descriptor files did not contain any residues.")

    n_frames_active = dataset_cfg.get("n_frames_active") or next(iter(active_features.values())).shape[0]
    n_frames_inactive = dataset_cfg.get("n_frames_inactive") or next(iter(inactive_features.values())).shape[0]
    residue_mapping = dataset_cfg.get("residue_mapping") or {}

    return (
        active_features,
        inactive_features,
        descriptor_keys,
        n_frames_active,
        n_frames_inactive,
        residue_mapping,
    )

def _combine_static_features(
    active_features: Dict[str, np.ndarray],
    inactive_features: Dict[str, np.ndarray],
    descriptor_keys: List[str],
    n_frames_active: int,
    n_frames_inactive: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    combined: Dict[str, np.ndarray] = {}
    for key in descriptor_keys:
        combined[key] = np.concatenate([active_features[key], inactive_features[key]], axis=0)

    labels = np.concatenate(
        [
            np.ones(n_frames_active, dtype=int),
            np.zeros(n_frames_inactive, dtype=int),
        ]
    )
    # shuffle_idx = np.random.permutation(labels.shape[0])
    # labels = labels[shuffle_idx]
    # for key in combined:
    #     combined[key] = combined[key][shuffle_idx]

    return combined, labels

# ======================================================================
# Runner
# ======================================================================

def run_analysis(
    analysis_type: str,
    file_paths: Dict[str, str],
    params: Dict[str, Any],
    residue_selections: Optional[Dict[str, str]] = None,
    progress_callback: Optional[callable] = None,
):

    # ------------------------------------------------------------
    # Helpers for printing / callback
    # ------------------------------------------------------------
    def report(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)
        else:
            print(f"[{pct}%] {msg}")


    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    use_descriptors = "active_descriptors" in file_paths

    reader = None
    extractor = None
    builder = None

    # Datasets for downstream steps
    features_static = None
    labels_Y = None
    mapping: Dict[str, str] = {}
    descriptor_keys: List[str] = []
    active_features_raw: Dict[str, np.ndarray] = {}
    inactive_features_raw: Dict[str, np.ndarray] = {}
    n_frames_active = None
    n_frames_inactive = None

    def _build_combined(static_act, static_inact, keys, n_act, n_inact):
        feats, labs = _combine_static_features(static_act, static_inact, keys, n_act, n_inact)
        if params.get("shuffle_static", True):
            shuffle_idx = np.random.permutation(labs.shape[0])
            labs = labs[shuffle_idx]
            for k in feats:
                feats[k] = feats[k][shuffle_idx]
        return feats, labs

    if use_descriptors:
        (
            active_features_raw,
            inactive_features_raw,
            descriptor_keys,
            n_frames_active,
            n_frames_inactive,
            mapping,
        ) = _load_descriptor_features(file_paths)
        features_static, labels_Y = _build_combined(
            active_features_raw,
            inactive_features_raw,
            descriptor_keys,
            n_frames_active,
            n_frames_inactive,
        )
    else:
        reader = MDAnalysisReader()
        extractor = FeatureExtractor(residue_selections)
        builder = DatasetBuilder(reader, extractor)
        report("Loading trajectories", 10)

        # Always extract per-state features; combine only when needed.
        active_features_raw, inactive_features_raw, mapping = builder.prepare_dynamic_analysis_data(
            file_paths["active_traj"], file_paths["active_topo"],
            file_paths["inactive_traj"], file_paths["inactive_topo"],
            active_slice=params.get("active_slice"),
            inactive_slice=params.get("inactive_slice"),
        )
        descriptor_keys = sorted(active_features_raw.keys())
        if not descriptor_keys:
            raise ValueError("No features found for analysis.")
        n_frames_active = next(iter(active_features_raw.values())).shape[0]
        n_frames_inactive = next(iter(inactive_features_raw.values())).shape[0]
        features_static, labels_Y = _build_combined(
            active_features_raw,
            inactive_features_raw,
            descriptor_keys,
            n_frames_active,
            n_frames_inactive,
        )

    # ==================================================================
    # Static state sensitivity
    # ==================================================================
    if analysis_type == "static":
        if features_static is None or labels_Y is None:
            raise ValueError("Static analysis dataset could not be prepared.")
        report("Running Static State Sensitivity", 20)

        static = StaticStateSensitivity()
        stats = static.run((features_static, labels_Y), **params)

        # Fail fast only if *all* state_score values are missing/non-finite.
        def _is_finite(val) -> bool:
            try:
                return np.isfinite(float(val))
            except Exception:
                return False
        invalid_scores = [k for k, v in stats.items() if not _is_finite(v.get("state_score"))]
        if stats and len(invalid_scores) == len(stats):
            preview = ", ".join(invalid_scores[:5])
            suffix = "…" if len(invalid_scores) > 5 else ""
            raise ValueError(f"Static analysis produced non-finite state_score for all residues: {preview}{suffix}")

        # Sort by state sensitivity
        sorted_keys = sorted(
            stats.keys(),
            key=lambda k: _safe_get(stats[k], "state_score"),
            reverse=True
        )
        final_static = {k: stats[k] for k in sorted_keys}

        report("Returning results", 100)
        return final_static, mapping

    raise ValueError("Only 'static' analysis is supported.")
