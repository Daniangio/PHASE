"""
Refactored runner for two-state QUBO analysis.

This implements:
  • QUBO_active   (Δ_act only)
  • QUBO_inactive (Δ_inact only)
  • QUBO_combined (Δ_avg, your original mode)

Classification uses all 3 QUBOs based on 9-label taxonomy:
  Global Hub, Active Hub, Inactive Hub,
  State-Switch Hub, Local Switch, Relay,
  Passive Scaffold, Redundant, Entropic Decoy.

We heavily comment classification logic for clarity.
"""

import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from alloskin.io.readers import MDAnalysisReader
from alloskin.features.extraction import FeatureExtractor
from alloskin.pipeline.builder import DatasetBuilder

from alloskin.analysis.static import StaticStateSensitivity
from alloskin.analysis.qubo import QUBOMaxCoverage   # Must be updated QUBO 2.0
from alloskin.analysis.dynamic import TransferEntropy


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
    # GOAL 1 — Static state sensitivity
    # ==================================================================
    if analysis_type == "static":
        if features_static is None or labels_Y is None:
            raise ValueError("Static analysis dataset could not be prepared.")
        report("Running Static State Sensitivity (Goal 1)", 20)

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

    # ==================================================================
    # GOAL 2 — QUBO
    # ==================================================================

    if analysis_type != "qubo":
        raise ValueError("Only 'static' and 'qubo' supported in this refactor.")

    report("Running QUBO (Goal 2)", 20)

    # ------------------------------------------------------------
    # 1. Gather static results (from file or new computation)
    # ------------------------------------------------------------
    static_results = None
    static_results_path = params.get("static_results_path")

    if static_results_path:
        report("Loading static analysis results from file", 30)
        with open(static_results_path, "r") as fh:
            static_results = json.load(fh)
    else:
        report("Running Static State Sensitivity (Goal 1)", 30)
        static = StaticStateSensitivity()
        if features_static is None or labels_Y is None:
            raise ValueError("Static analysis dataset could not be prepared for QUBO.")
        static_results = static.run((features_static, labels_Y), **params)

    # =============================================================
    # 2. RUN QUBOs per-state
    # =============================================================

    qubo = QUBOMaxCoverage()
    qubo_allowed_params = [
        "alpha", "beta_switch", "beta_hub",
        "gamma_redundancy", "ii_threshold",
        "ii_scale", "soft_threshold_power",
        "num_reads", "num_solutions", "seed",
        "use_per_state_imbalance",
        "filter_min_id", "filter_top_jsd", "filter_top_total",
        "static_results_path", "maxk",
    ]

    # Normalize param names to analyzer expectations
    qubo_params = dict(params)
    if "alpha_size" in qubo_params and "alpha" not in qubo_params:
        qubo_params["alpha"] = qubo_params.pop("alpha_size")
    final_qubo_params = {k: v for k, v in qubo_params.items() if k in qubo_allowed_params}

    # Prepare per-state feature dictionaries
    if not active_features_raw or not inactive_features_raw:
        raise ValueError("Active/inactive features missing for QUBO analysis.")
    features_act = {k: active_features_raw[k] for k in descriptor_keys if k in active_features_raw}
    features_inact = {k: inactive_features_raw[k] for k in descriptor_keys if k in inactive_features_raw}

    # A) ACTIVE only → use_per_state_imbalance=False (Δ_act only)
    report("Running QUBO_active (only active-state Δ)", 40)
    res_act = qubo.run(
        features_act,   # active only
        static_results=static_results,
        **final_qubo_params
    )

    # B) INACTIVE only
    report("Running QUBO_inactive (only inactive-state Δ)", 60)
    res_inact = qubo.run(
        features_inact,
        static_results=static_results,
        **final_qubo_params
    )

    # ==================================================================
    # 4. CLASSIFICATION USING ACTIVE/INACTIVE ONLY (no combined run)
    # ==================================================================
    report("Assigning taxonomy labels", 80)

    cand_act = res_act.get("matrix_indices", []) if isinstance(res_act, dict) else []
    cand_inact = res_inact.get("matrix_indices", []) if isinstance(res_inact, dict) else []
    candidates = sorted(set(cand_act) | set(cand_inact))

    if not candidates:
        return {
            "qubo_active": res_act,
            "qubo_inactive": res_inact,
            "classification": {},
            "mapping": mapping,
        }

    sel_act = set(res_act.get("solutions", [{}])[0].get("selected", [])) if "solutions" in res_act else set()
    sel_inact = set(res_inact.get("solutions", [{}])[0].get("selected", [])) if "solutions" in res_inact else set()

    hub_act = res_act.get("hub_scores", {}) if isinstance(res_act, dict) else {}
    hub_inact = res_inact.get("hub_scores", {}) if isinstance(res_inact, dict) else {}

    raw_scores_act = res_act.get("raw_state_scores", {}) if isinstance(res_act, dict) else {}
    raw_scores_inact = res_inact.get("raw_state_scores", {}) if isinstance(res_inact, dict) else {}
    reg_scores_act = res_act.get("regularized_state_scores", {}) if isinstance(res_act, dict) else {}
    reg_scores_inact = res_inact.get("regularized_state_scores", {}) if isinstance(res_inact, dict) else {}

    cov_act = np.array(res_act.get("coverage_weights", []), float) if isinstance(res_act, dict) and "coverage_weights" in res_act else np.empty((0, 0))
    cov_inact = np.array(res_inact.get("coverage_weights", []), float) if isinstance(res_inact, dict) and "coverage_weights" in res_inact else np.empty((0, 0))
    idx_act = {k: i for i, k in enumerate(cand_act)}
    idx_inact = {k: i for i, k in enumerate(cand_inact)}

    switch_high = params.get("taxonomy_switch_high", 0.8)
    switch_low = params.get("taxonomy_switch_low", 0.3)

    hub_act_vals = np.array([hub_act.get(k, 0.0) for k in candidates])
    hub_inact_vals = np.array([hub_inact.get(k, 0.0) for k in candidates])
    delta_vals = np.abs(hub_act_vals - hub_inact_vals)

    hub_high_act = np.percentile(hub_act_vals, params.get("taxonomy_hub_high_percentile", 80))
    hub_high_inact = np.percentile(hub_inact_vals, params.get("taxonomy_hub_high_percentile", 80))
    hub_low_act = np.percentile(hub_act_vals, params.get("taxonomy_hub_low_percentile", 50))
    hub_low_inact = np.percentile(hub_inact_vals, params.get("taxonomy_hub_low_percentile", 50))
    delta_high = np.percentile(delta_vals, params.get("taxonomy_delta_hub_high_percentile", 80))

    classification = {}

    for k in candidates:
        s_raw = max(float(raw_scores_act.get(k, 0.0)), float(raw_scores_inact.get(k, 0.0)))
        s_reg = max(reg_scores_act.get(k, s_raw), reg_scores_inact.get(k, s_raw))

        ha = float(hub_act.get(k, 0.0))
        hi = float(hub_inact.get(k, 0.0))
        dH = abs(ha - hi)

        in_act = k in sel_act
        in_inact = k in sel_inact

        # ------------------------------------------------------------
        # Determine taxonomy
        # ------------------------------------------------------------

        if in_act and in_inact:
            # Selected in both → structural core of both ensembles
            if ha >= hub_high_act and hi >= hub_high_inact:
                role = "Global Hub"
            else:
                role = "Global Hub [minor]"
        elif in_act and not in_inact:
            if ha >= hub_high_act and hi <= hub_low_inact:
                role = "Active Hub"
            elif s_reg >= switch_high and dH >= delta_high:
                role = "State-Switch Hub"
            else:
                role = "Relay"
        elif in_inact and not in_act:
            if hi >= hub_high_inact and ha <= hub_low_act:
                role = "Inactive Hub"
            elif s_reg >= switch_high and dH >= delta_high:
                role = "State-Switch Hub"
            else:
                role = "Relay"
        else:
            # Not selected in either active or inactive QUBO
            unique = 0.0
            if k in idx_act and cov_act.size:
                idx = idx_act[k]
                sel_idx = [idx_act[x] for x in sel_act if x in idx_act]
                if sel_idx:
                    union_cov = np.max(cov_act[sel_idx, :], axis=0)
                    unique = float(np.maximum(cov_act[idx] - union_cov, 0).sum())
                else:
                    unique = float(cov_act[idx].sum())
            elif k in idx_inact and cov_inact.size:
                idx = idx_inact[k]
                sel_idx = [idx_inact[x] for x in sel_inact if x in idx_inact]
                if sel_idx:
                    union_cov = np.max(cov_inact[sel_idx, :], axis=0)
                    unique = float(np.maximum(cov_inact[idx] - union_cov, 0).sum())
                else:
                    unique = float(cov_inact[idx].sum())

            if unique < 1e-3:
                role = "Redundant"
            else:
                if s_reg >= switch_high:
                    role = "Local Switch"
                elif s_reg < switch_low:
                    role = "Passive Scaffold"
                else:
                    role = "Entropic Decoy"

        classification[k] = {
            "label": role,
            "s_reg": float(s_reg),
            "hub_active": ha,
            "hub_inactive": hi,
            "delta_hub": dH,
            "selected_active": in_act,
            "selected_inactive": in_inact,
        }

    report("Returning results", 100)
    return {
        "qubo_active": res_act,
        "qubo_inactive": res_inact,
        "classification": classification,
        "mapping": mapping,
    }
