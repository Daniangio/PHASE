#!/usr/bin/env python3
"""
Assign per-residue cluster labels for a custom structure/trajectory.

This script is standalone: it only needs
  - numpy
  - MDAnalysis
  - dadapy
and a cluster folder containing:
  - cluster.npz
  - models/*.pkl (referenced by metadata_json in cluster.npz)
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import MDAnalysis as mda
    import MDAnalysis.analysis.dihedrals as mda_dihedrals
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(f"ERROR: MDAnalysis is required ({exc})")

try:
    from dadapy import Data
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(f"ERROR: dadapy is required ({exc})")


DIHEDRAL_KEYS = ("phi", "psi", "omega", "chi1", "chi2")

CHI2_DESCRIPTOR_INDEX = 4
SYMMETRIC_CHI2_RESNAMES = frozenset({"ASP", "LEU", "PHE", "TYR"})


def _fold_samples_for_model_symmetry(
    samples: np.ndarray,
    model: Any,
    *,
    residue_resname: Optional[str] = None,
) -> np.ndarray:
    meta = getattr(model, "phase_descriptor_symmetry", None)
    if not isinstance(meta, dict) or not bool(meta.get("enabled")):
        return np.asarray(samples, dtype=np.float64)
    descriptor = str(meta.get("descriptor") or "chi2").strip().lower()
    try:
        descriptor_index = int(meta.get("descriptor_index", CHI2_DESCRIPTOR_INDEX))
    except (TypeError, ValueError):
        return np.asarray(samples, dtype=np.float64)
    model_resname = str(meta.get("resname") or "").strip().upper()
    input_resname = str(residue_resname or "").strip().upper()
    if descriptor != "chi2" or descriptor_index != CHI2_DESCRIPTOR_INDEX:
        return np.asarray(samples, dtype=np.float64)
    if model_resname and model_resname not in SYMMETRIC_CHI2_RESNAMES:
        return np.asarray(samples, dtype=np.float64)
    if input_resname and input_resname not in SYMMETRIC_CHI2_RESNAMES:
        return np.asarray(samples, dtype=np.float64)
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] <= CHI2_DESCRIPTOR_INDEX:
        return arr
    folded = arr.copy()
    folded[:, CHI2_DESCRIPTOR_INDEX] = np.mod(2.0 * folded[:, CHI2_DESCRIPTOR_INDEX], 2.0 * np.pi)
    return folded

def _effective_angle_columns(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    keep: List[int] = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = np.isfinite(col)
        if not np.any(finite):
            continue
        finite_vals = col[finite]
        if finite_vals.size == 0:
            continue
        if np.all(np.abs(finite_vals) < 1e-12):
            continue
        keep.append(j)
    if keep:
        return np.asarray(keep, dtype=np.int32)
    finite_cols = [j for j in range(arr.shape[1]) if np.any(np.isfinite(arr[:, j]))]
    return np.asarray(finite_cols, dtype=np.int32)


def _angles_to_periodic(samples: np.ndarray, *, expected_dims: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if expected_dims is not None:
        exp_dims = max(0, int(expected_dims))
        if arr.shape[1] < exp_dims:
            arr = np.pad(arr, ((0, 0), (0, exp_dims - arr.shape[1])), constant_values=0.0)
        angles = arr[:, :exp_dims]
    else:
        cols = _effective_angle_columns(arr)
        if cols.size == 0:
            return np.zeros((arr.shape[0], 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        angles = arr[:, cols]
    two_pi = 2.0 * np.pi
    centered = np.mod(angles, two_pi)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    period = np.full(centered.shape[1], two_pi, dtype=np.float64)
    return centered, period


def _angles_to_circular_features(samples: np.ndarray) -> np.ndarray:
    centered, _ = _angles_to_periodic(samples)
    if centered.shape[1] == 0:
        return np.zeros((centered.shape[0], 0), dtype=np.float64)
    return np.concatenate([np.sin(centered), np.cos(centered)], axis=1).astype(np.float64, copy=False)


def _predict_cluster_adp(
    dp_data: Data,
    samples: np.ndarray,
    *,
    density_maxk: int,
    residue_resname: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    expected_dims: Optional[int] = None
    model_X = getattr(dp_data, "X", None)
    if model_X is not None:
        try:
            expected_dims = int(np.asarray(model_X).shape[1])
        except Exception:
            expected_dims = None

    samples = _fold_samples_for_model_symmetry(samples, dp_data, residue_resname=residue_resname)
    emb, _ = _angles_to_periodic(samples, expected_dims=expected_dims)
    if emb.shape[0] == 0:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty

    # For tiny query batches (single-pose/custom-PDB use cases), ADP's prediction path can be
    # disproportionately slow. Nearest training label in periodic angle space is usually the
    # same practical target and avoids long waits.
    if emb.shape[0] <= 4:
        train_X = np.asarray(getattr(dp_data, "X", None), dtype=np.float64)
        train_assigned = np.asarray(getattr(dp_data, "cluster_assignment", None), dtype=np.int32).reshape(-1)
        train_halo = np.asarray(getattr(dp_data, "cluster_assignment_halo", train_assigned), dtype=np.int32).reshape(-1)
        if train_X.ndim == 2 and train_X.shape[0] > 0 and train_X.shape[0] == train_assigned.shape[0]:
            two_pi = 2.0 * np.pi
            assigned_rows: List[int] = []
            halo_rows: List[int] = []
            for row in emb:
                delta = np.abs(train_X - row.reshape(1, -1))
                if delta.shape[1]:
                    delta = np.minimum(delta, two_pi - delta)
                nearest_idx = int(np.argmin(np.sum(delta * delta, axis=1)))
                assigned_rows.append(int(train_assigned[nearest_idx]))
                halo_rows.append(int(train_halo[nearest_idx]))
            return np.asarray(assigned_rows, dtype=np.int32), np.asarray(halo_rows, dtype=np.int32)

    squeeze_single = emb.shape[0] == 1
    emb_pred = np.concatenate([emb, emb], axis=0) if squeeze_single else emb
    maxk_val = max(1, min(int(density_maxk), emb_pred.shape[0] - 1))
    try:
        result = dp_data.predict_cluster_ADP(emb_pred, maxk=maxk_val, density_est="kstarNN", n_jobs=1)
    except ValueError as exc:
        if "Buffer has wrong number of dimensions" not in str(exc):
            raise
        train_X = np.asarray(getattr(dp_data, "X", None), dtype=np.float64)
        train_assigned = np.asarray(getattr(dp_data, "cluster_assignment", None), dtype=np.int32).reshape(-1)
        train_halo = np.asarray(getattr(dp_data, "cluster_assignment_halo", train_assigned), dtype=np.int32).reshape(-1)
        if train_X.ndim != 2 or train_X.shape[0] == 0 or train_X.shape[0] != train_assigned.shape[0]:
            raise
        two_pi = 2.0 * np.pi
        assigned_rows: List[int] = []
        halo_rows: List[int] = []
        for row in emb_pred:
            delta = np.abs(train_X - row.reshape(1, -1))
            if delta.shape[1]:
                delta = np.minimum(delta, two_pi - delta)
            nearest_idx = int(np.argmin(np.sum(delta * delta, axis=1)))
            assigned_rows.append(int(train_assigned[nearest_idx]))
            halo_rows.append(int(train_halo[nearest_idx]))
        result = (np.asarray(assigned_rows, dtype=np.int32), np.asarray(halo_rows, dtype=np.int32))

    labels_assigned = result[0] if isinstance(result, tuple) else result
    labels_halo = result[1] if isinstance(result, tuple) and len(result) > 1 else labels_assigned

    def _coerce_labels(labels: np.ndarray) -> np.ndarray:
        arr = np.asarray(labels, dtype=np.int32)
        if arr.ndim > 1:
            if arr.shape[1] == 0:
                return arr.reshape(-1)
            return arr[:, 0]
        return arr

    assigned = _coerce_labels(labels_assigned)
    halo = _coerce_labels(labels_halo)
    if squeeze_single:
        assigned = assigned[:1]
        halo = halo[:1]
    return assigned, halo


def _gaussian_logpdf_matrix(
    X: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    covariance_type: str,
    reg_covar: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    n, d = X.shape
    k = means.shape[0]
    out = np.full((n, k), -np.inf, dtype=np.float64)
    log2pi = d * np.log(2.0 * np.pi)
    reg = float(max(reg_covar, 1e-12))

    cov_kind = str(covariance_type or "full").lower()
    if cov_kind == "diag":
        for j in range(k):
            var = np.asarray(covariances[j], dtype=np.float64).reshape(-1)
            if var.size != d:
                var = np.resize(var, d)
            var = np.maximum(var, reg)
            diff = X - means[j]
            quad = np.sum((diff * diff) / var[None, :], axis=1)
            logdet = float(np.sum(np.log(var)))
            out[:, j] = -0.5 * (log2pi + logdet + quad)
        return out

    for j in range(k):
        cov = np.asarray(covariances[j], dtype=np.float64)
        if cov.shape != (d, d):
            cov = np.eye(d, dtype=np.float64) * reg
        cov = cov + np.eye(d, dtype=np.float64) * reg
        try:
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise np.linalg.LinAlgError("non-positive definite covariance")
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov = cov + np.eye(d, dtype=np.float64) * (reg * 10.0)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                continue
            inv = np.linalg.inv(cov)

        diff = X - means[j]
        quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
        out[:, j] = -0.5 * (log2pi + float(logdet) + quad)
    return out


def _predict_cluster_frozen_gmm(model: Dict[str, Any], samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = _angles_to_circular_features(samples)
    means = np.asarray(model.get("means", []), dtype=np.float64)
    covariances = np.asarray(model.get("covariances", []), dtype=np.float64)
    weights = np.asarray(model.get("weights", []), dtype=np.float64).reshape(-1)
    thresholds = np.asarray(model.get("thresholds_logpdf", []), dtype=np.float64).reshape(-1)
    covariance_type = str(model.get("covariance_type") or "full").lower()
    reg_covar = float(model.get("reg_covar", 1e-5))

    n = features.shape[0]
    if means.ndim != 2 or means.shape[0] == 0:
        empty = np.full((n,), -1, dtype=np.int32)
        return empty, empty

    k = means.shape[0]
    if weights.size != k:
        weights = np.full((k,), 1.0 / float(k), dtype=np.float64)
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    if thresholds.size != k:
        thresholds = np.full((k,), -np.inf, dtype=np.float64)

    logpdf = _gaussian_logpdf_matrix(
        features,
        means,
        covariances,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
    )
    logpost = logpdf + np.log(weights[None, :])
    assigned = np.argmax(logpost, axis=1).astype(np.int32, copy=False)
    halo = assigned.copy()
    chosen = logpdf[np.arange(n), assigned]
    halo[chosen < thresholds[assigned]] = -1
    return assigned, halo


def _predict_with_model(
    model: Any,
    samples: np.ndarray,
    *,
    density_maxk: int,
    residue_resname: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(model, dict):
        kind = str(model.get("kind") or "").lower()
        if kind == "frozen_gmm":
            return _predict_cluster_frozen_gmm(model, samples)
        if kind == "adp_legacy_models":
            model_assigned = model.get("model_assigned")
            model_halo = model.get("model_halo")
            assigned, _ = _predict_cluster_adp(
                model_assigned,
                samples,
                density_maxk=density_maxk,
                residue_resname=residue_resname,
            )
            _, halo = _predict_cluster_adp(
                model_halo,
                samples,
                density_maxk=density_maxk,
                residue_resname=residue_resname,
            )
            return assigned, halo
    if hasattr(model, "predict_cluster_ADP"):
        return _predict_cluster_adp(
            model,
            samples,
            density_maxk=density_maxk,
            residue_resname=residue_resname,
        )
    raise TypeError(f"Unsupported residue model type: {type(model)}")


def _safe_residue_dihedral_selection(
    residue: mda.core.groups.Residue,
    method_name: str,
) -> Optional[mda.core.groups.AtomGroup]:
    method = getattr(residue, method_name, None)
    if method is None:
        return None
    try:
        ag = method()
    except Exception:
        return None
    if ag is None:
        return None
    try:
        return ag if int(ag.n_atoms) == 4 else None
    except Exception:
        return None


def _chi2_selection(residue: mda.core.groups.Residue) -> Optional[mda.core.groups.AtomGroup]:
    try:
        atoms = residue.atoms
        ag1 = atoms.select_atoms("name CA")
        ag2 = atoms.select_atoms("name CB")
        ag3 = atoms.select_atoms("name CG CG1")
        ag4 = atoms.select_atoms("name CD CD1 OD1 ND1 SD")
    except Exception:
        return None
    if any(int(ag.n_atoms) != 1 for ag in (ag1, ag2, ag3, ag4)):
        return None
    return ag1 + ag2 + ag3 + ag4


def _calculate_dihedrals(
    atom_groups_list: Sequence[Optional[mda.core.groups.AtomGroup]],
    *,
    n_frames: int,
    start: Optional[int],
    stop: Optional[int],
    step: Optional[int],
) -> np.ndarray:
    mask = np.array([ag is not None for ag in atom_groups_list], dtype=bool)
    angles = np.full((n_frames, len(atom_groups_list)), 0.0, dtype=np.float32)
    valid_atom_groups = [ag for ag in atom_groups_list if ag is not None]
    if not valid_atom_groups:
        return angles
    analysis = mda_dihedrals.Dihedral(valid_atom_groups).run(start=start, stop=stop, step=step)
    raw = np.asarray(analysis.results.angles, dtype=np.float32)
    if raw.shape[0] != n_frames:
        raise ValueError(
            f"Dihedral analysis frame mismatch: got {raw.shape[0]} frames, expected {n_frames} (start={start}, stop={stop}, step={step})."
        )
    angles[:, mask] = raw
    return angles


def _extract_residue_dihedrals(
    universe: mda.Universe,
    residues: Sequence[mda.core.groups.Residue],
    *,
    start: Optional[int],
    stop: Optional[int],
    step: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    total = len(universe.trajectory)
    s = 0 if start is None else int(start)
    e = total if stop is None else int(stop)
    st = 1 if step is None else int(step)
    if st <= 0:
        raise ValueError("--step must be > 0")
    frame_indices = np.asarray(list(range(s, e, st)), dtype=np.int64)
    n_frames = int(frame_indices.size)
    if n_frames == 0:
        raise ValueError("Selected frame window has 0 frames.")

    selections: Dict[str, List[Optional[mda.core.groups.AtomGroup]]] = {name: [] for name in DIHEDRAL_KEYS}
    for residue in residues:
        selections["phi"].append(_safe_residue_dihedral_selection(residue, "phi_selection"))
        selections["psi"].append(_safe_residue_dihedral_selection(residue, "psi_selection"))
        selections["omega"].append(_safe_residue_dihedral_selection(residue, "omega_selection"))
        selections["chi1"].append(_safe_residue_dihedral_selection(residue, "chi1_selection"))
        selections["chi2"].append(_chi2_selection(residue))

    all_selections: List[Optional[mda.core.groups.AtomGroup]] = []
    offsets: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for name in DIHEDRAL_KEYS:
        block = selections[name]
        offsets[name] = (cursor, cursor + len(block))
        all_selections.extend(block)
        cursor += len(block)

    all_angles_deg = _calculate_dihedrals(all_selections, n_frames=n_frames, start=s, stop=e, step=st)
    n_res = len(residues)
    out_deg = np.zeros((n_frames, n_res, len(DIHEDRAL_KEYS)), dtype=np.float32)
    for i in range(n_res):
        for d_idx, name in enumerate(DIHEDRAL_KEYS):
            a, b = offsets[name]
            out_deg[:, i, d_idx] = all_angles_deg[:, a:b][:, i]
    out_rad = np.deg2rad(out_deg.astype(np.float64, copy=False)).astype(np.float32, copy=False)
    return out_rad, frame_indices


def _resolve_model_path(cluster_dir: Path, rel_or_abs: str) -> Path:
    p = Path(str(rel_or_abs))
    if p.is_absolute():
        return p
    system_dir = cluster_dir.parent.parent
    candidates = [
        cluster_dir / p,
        cluster_dir.parent / p,
        system_dir / p,
        cluster_dir / "models" / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return system_dir / p


def _load_models_from_metadata(
    cluster_dir: Path,
    metadata: Dict[str, Any],
    residue_keys: Sequence[str],
    *,
    selected_indices: Optional[Sequence[int]] = None,
) -> List[Any]:
    n_res = len(residue_keys)
    models: List[Any] = [None] * n_res
    selected_set = set(int(i) for i in selected_indices) if selected_indices is not None else None
    model_paths = metadata.get("model_paths") or []
    model_paths_halo = metadata.get("model_paths_halo") or []
    model_paths_assigned = metadata.get("model_paths_assigned") or []

    if isinstance(model_paths, list) and len(model_paths) == n_res:
        for i, rel in enumerate(model_paths):
            if selected_set is not None and i not in selected_set:
                continue
            if not rel:
                continue
            p = _resolve_model_path(cluster_dir, str(rel))
            if not p.exists():
                continue
            with open(p, "rb") as inp:
                models[i] = pickle.load(inp)
    elif (
        isinstance(model_paths_halo, list)
        and isinstance(model_paths_assigned, list)
        and len(model_paths_halo) == n_res
        and len(model_paths_assigned) == n_res
    ):
        for i, (rel_h, rel_a) in enumerate(zip(model_paths_halo, model_paths_assigned)):
            if selected_set is not None and i not in selected_set:
                continue
            if not rel_h or not rel_a:
                continue
            ph = _resolve_model_path(cluster_dir, str(rel_h))
            pa = _resolve_model_path(cluster_dir, str(rel_a))
            if not ph.exists() or not pa.exists():
                continue
            with open(ph, "rb") as inp:
                mh = pickle.load(inp)
            with open(pa, "rb") as inp:
                ma = pickle.load(inp)
            models[i] = {"kind": "adp_legacy_models", "model_halo": mh, "model_assigned": ma}

    overrides = metadata.get("residue_model_overrides") or {}
    if isinstance(overrides, dict):
        for key, spec in overrides.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if idx < 0 or idx >= n_res:
                continue
            if selected_set is not None and idx not in selected_set:
                continue
            if isinstance(spec, dict) and str(spec.get("kind") or "").lower() == "frozen_gmm":
                models[idx] = spec
    return models


def _parse_residue_subset(spec: Optional[str], residue_keys: Sequence[str]) -> List[int]:
    if not spec:
        return list(range(len(residue_keys)))
    key_to_idx = {str(k): i for i, k in enumerate(residue_keys)}
    out: List[int] = []
    tokens = [t.strip() for t in str(spec).split(",") if t.strip()]
    for tok in tokens:
        if tok in key_to_idx:
            out.append(key_to_idx[tok])
            continue
        m_key = re.fullmatch(r"res_(\d+)", tok)
        if m_key and tok in key_to_idx:
            out.append(key_to_idx[tok])
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            a_i = int(a)
            b_i = int(b)
            lo, hi = sorted((a_i, b_i))
            for resid in range(lo, hi + 1):
                k = f"res_{resid}"
                if k in key_to_idx:
                    out.append(key_to_idx[k])
            continue
        resid = int(tok)
        k = f"res_{resid}"
        if k not in key_to_idx:
            raise ValueError(f"Residue '{tok}' not found in cluster residue keys.")
        out.append(key_to_idx[k])
    dedup = sorted(set(int(i) for i in out))
    if not dedup:
        raise ValueError("No valid residues selected.")
    return dedup


def _extract_resid_from_key(residue_key: str) -> Optional[int]:
    m = re.search(r"(-?\d+)", str(residue_key))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _shift_resid_terms(selection: str, offset: int) -> str:
    if not selection:
        return selection

    def _replace(match: re.Match[str]) -> str:
        prefix = match.group(1)
        number = int(match.group(2))
        return f"{prefix}{number - int(offset)}"

    # Shift only "resid <N>" clauses.
    return re.sub(r"\b(resid\s+)(-?\d+)\b", _replace, str(selection))


def _pick_residue_for_key(
    universe: mda.Universe,
    *,
    residue_key: str,
    offset: int,
    residue_mapping_entry: Optional[str],
) -> mda.core.groups.Residue:
    queries: List[str] = []
    if residue_mapping_entry:
        shifted = _shift_resid_terms(str(residue_mapping_entry), offset)
        queries.append(f"protein and ({shifted})")
        queries.append(str(shifted))
    resid_key = _extract_resid_from_key(str(residue_key))
    if resid_key is not None:
        resid = int(resid_key) - offset
        queries.append(f"protein and resid {resid}")
        queries.append(f"resid {resid}")

    for q in queries:
        try:
            residues = universe.select_atoms(q).residues
        except Exception:
            continue
        if len(residues) == 1:
            return residues[0]
        if len(residues) > 1:
            return residues[0]
    raise ValueError(
        f"Could not map residue key '{residue_key}' to exactly one residue in the input structure."
    )


def _map_residues_for_offset(
    universe: mda.Universe,
    *,
    selected_keys: Sequence[str],
    residue_mapping: Dict[str, Any],
    offset: int,
) -> Tuple[List[mda.core.groups.Residue], List[str]]:
    residues: List[mda.core.groups.Residue] = []
    warnings: List[str] = []
    for key in selected_keys:
        map_entry = residue_mapping.get(str(key))
        picked = _pick_residue_for_key(
            universe,
            residue_key=str(key),
            offset=int(offset),
            residue_mapping_entry=str(map_entry) if map_entry else None,
        )
        residues.append(picked)

    # Consistency checks: key-based expected resid after offset vs picked residue resid.
    for key, residue in zip(selected_keys, residues):
        key_resid = _extract_resid_from_key(str(key))
        if key_resid is None:
            continue
        expected = int(key_resid) - int(offset)
        actual = int(getattr(residue, "resid", expected))
        if expected != actual:
            warnings.append(
                f"Residue key '{key}' expected resid {expected} after offset {offset}, "
                f"but mapped to resid {actual} ({residue.resname})."
            )
    return residues, warnings


def _candidate_offsets(
    universe: mda.Universe,
    selected_keys: Sequence[str],
    *,
    max_abs_offset: int,
) -> List[int]:
    key_resids = [v for v in (_extract_resid_from_key(k) for k in selected_keys) if v is not None]
    if not key_resids:
        return [0]
    first_key = int(key_resids[0])
    protein_resids = [int(r.resid) for r in universe.select_atoms("protein").residues]
    cands = {0}
    for resid in protein_resids:
        off = first_key - int(resid)
        if abs(off) <= int(max_abs_offset):
            cands.add(int(off))
    # Also include a dense local window around 0 for robustness.
    for off in range(-min(20, max_abs_offset), min(20, max_abs_offset) + 1):
        cands.add(int(off))
    return sorted(cands, key=lambda x: (abs(x), x))


def _auto_resolve_offset(
    universe: mda.Universe,
    *,
    selected_keys: Sequence[str],
    residue_mapping: Dict[str, Any],
    max_abs_offset: int,
) -> Tuple[int, List[mda.core.groups.Residue], List[str]]:
    errors: List[str] = []
    for off in _candidate_offsets(universe, selected_keys, max_abs_offset=max_abs_offset):
        try:
            residues, warnings = _map_residues_for_offset(
                universe,
                selected_keys=selected_keys,
                residue_mapping=residue_mapping,
                offset=off,
            )
            return int(off), residues, warnings
        except Exception as exc:
            errors.append(f"offset={off}: {exc}")
            continue
    preview = "; ".join(errors[:8])
    raise ValueError(
        f"Could not auto-resolve residue offset (tried offsets within ±{max_abs_offset}). "
        f"First errors: {preview}"
    )


def _load_cluster_inputs(cluster_dir: Path) -> tuple[List[str], Dict[str, Any], np.ndarray, int]:
    npz_path = cluster_dir / "cluster.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing cluster.npz in: {cluster_dir}")

    with np.load(npz_path, allow_pickle=False) as data:
        meta_raw = data.get("metadata_json")
        if meta_raw is None:
            raise ValueError("cluster.npz is missing metadata_json")
        metadata = json.loads(str(meta_raw.item() if hasattr(meta_raw, "item") else meta_raw))
        residue_keys = [str(v) for v in (metadata.get("residue_keys") or data.get("residue_keys", []))]
        if not residue_keys:
            raise ValueError("Could not recover residue_keys from cluster.npz")
        if "cluster_counts" in data:
            cluster_counts = np.asarray(data["cluster_counts"], dtype=np.int32)
        elif "merged__cluster_counts" in data:
            cluster_counts = np.asarray(data["merged__cluster_counts"], dtype=np.int32)
        else:
            cluster_counts = np.full((len(residue_keys),), -1, dtype=np.int32)

    params = metadata.get("cluster_params") or {}
    density_maxk = int(params.get("density_maxk", 100))
    return residue_keys, metadata, cluster_counts, density_maxk


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign cluster labels (per residue) to a custom structure/trajectory using an existing cluster folder.",
    )
    parser.add_argument("--cluster-dir", required=True, help="Path to a cluster folder containing cluster.npz.")
    parser.add_argument("--structure", required=True, help="Topology/structure file (PDB/mmCIF/etc.)")
    parser.add_argument("--trajectory", default="", help="Optional trajectory file (XTC/DCD/etc.)")
    parser.add_argument(
        "--residues",
        default="",
        help="Optional subset, e.g. 'res_10,res_11,50-60'. If omitted, all cluster residues are used.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help=(
            "Residue-number offset between cluster keys and input structure numbering. "
            "If omitted, script tries to infer it automatically."
        ),
    )
    parser.add_argument(
        "--max-auto-offset",
        type=int,
        default=300,
        help="Max |offset| considered during automatic offset search (default: 300).",
    )
    parser.add_argument("--start", type=int, default=None, help="Optional trajectory start frame.")
    parser.add_argument("--stop", type=int, default=None, help="Optional trajectory stop frame.")
    parser.add_argument("--step", type=int, default=1, help="Trajectory stride (default: 1).")
    parser.add_argument("--output", default="", help="Optional output NPZ path.")
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    parser.add_argument(
        "--label-mode",
        choices=["assigned", "halo", "both"],
        default="assigned",
        help="Which labels to print in terminal output (default: assigned).",
    )
    args = parser.parse_args()

    cluster_dir = Path(args.cluster_dir).expanduser().resolve()
    structure_path = Path(args.structure).expanduser().resolve()
    trajectory_path = Path(args.trajectory).expanduser().resolve() if args.trajectory else None
    if not cluster_dir.exists():
        raise SystemExit(f"ERROR: cluster dir not found: {cluster_dir}")
    if not structure_path.exists():
        raise SystemExit(f"ERROR: structure file not found: {structure_path}")
    if trajectory_path and not trajectory_path.exists():
        raise SystemExit(f"ERROR: trajectory file not found: {trajectory_path}")

    residue_keys, metadata, cluster_counts_all, density_maxk = _load_cluster_inputs(cluster_dir)
    selected_indices = _parse_residue_subset(args.residues, residue_keys)
    models_all = _load_models_from_metadata(
        cluster_dir,
        metadata,
        residue_keys,
        selected_indices=selected_indices,
    )
    if not any(models_all[i] is not None for i in selected_indices):
        raise SystemExit("ERROR: no residue models were found for the selected residues.")

    selected_keys = [residue_keys[i] for i in selected_indices]
    selected_models = [models_all[i] for i in selected_indices]
    selected_counts = np.asarray([cluster_counts_all[i] for i in selected_indices], dtype=np.int32)

    if trajectory_path:
        universe = mda.Universe(str(structure_path), str(trajectory_path))
    else:
        universe = mda.Universe(str(structure_path))

    residue_mapping = metadata.get("residue_mapping") or {}
    if args.offset is None:
        chosen_offset, selected_residues, mapping_warnings = _auto_resolve_offset(
            universe,
            selected_keys=selected_keys,
            residue_mapping=residue_mapping,
            max_abs_offset=max(0, int(args.max_auto_offset)),
        )
        print(f"[offset] Auto-detected offset: {chosen_offset}")
    else:
        chosen_offset = int(args.offset)
        selected_residues, mapping_warnings = _map_residues_for_offset(
            universe,
            selected_keys=selected_keys,
            residue_mapping=residue_mapping,
            offset=chosen_offset,
        )
        print(f"[offset] Using user-provided offset: {chosen_offset}")

    if mapping_warnings:
        print("[mapping] Sequence/numbering consistency warnings:", file=sys.stderr)
        for msg in mapping_warnings:
            print(f"  - {msg}", file=sys.stderr)

    angles_rad, frame_indices = _extract_residue_dihedrals(
        universe,
        selected_residues,
        start=args.start,
        stop=args.stop,
        step=args.step,
    )
    n_frames = int(angles_rad.shape[0])
    n_res = int(angles_rad.shape[1])

    labels_assigned = np.full((n_frames, n_res), -1, dtype=np.int32)
    labels_halo = np.full((n_frames, n_res), -1, dtype=np.int32)

    for j in range(n_res):
        model = selected_models[j]
        if model is None:
            print(f"WARNING: missing model for {selected_keys[j]} -> labels stay -1", file=sys.stderr)
            continue
        samples = np.asarray(angles_rad[:, j, :], dtype=np.float32)
        assigned, halo = _predict_with_model(
            model,
            samples,
            density_maxk=density_maxk,
            residue_resname=str(getattr(selected_residues[j], "resname", "") or ""),
        )
        labels_assigned[:, j] = np.asarray(assigned, dtype=np.int32)
        labels_halo[:, j] = np.asarray(halo, dtype=np.int32)

    if args.output:
        out_npz = Path(args.output).expanduser().resolve()
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_npz,
            residue_keys=np.asarray(selected_keys, dtype=object),
            residue_indices=np.asarray(selected_indices, dtype=np.int32),
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
            labels_assigned=labels_assigned,
            labels_halo=labels_halo,
            cluster_counts=selected_counts,
            descriptor_order=np.asarray(DIHEDRAL_KEYS, dtype=object),
            used_offset=np.asarray([chosen_offset], dtype=np.int32),
        )
        print(f"Saved NPZ: {out_npz}")

    if args.output_json:
        out_json = Path(args.output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cluster_dir": str(cluster_dir),
            "structure": str(structure_path),
            "trajectory": str(trajectory_path) if trajectory_path else None,
            "residue_keys": selected_keys,
            "residue_indices": selected_indices,
            "frame_indices": frame_indices.tolist(),
            "labels_assigned": labels_assigned.tolist(),
            "labels_halo": labels_halo.tolist(),
            "cluster_counts": selected_counts.tolist(),
            "descriptor_order": list(DIHEDRAL_KEYS),
            "used_offset": int(chosen_offset),
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON: {out_json}")

    print()
    print(f"Cluster dir: {cluster_dir}")
    print(f"Input: {structure_path}" + (f" + {trajectory_path}" if trajectory_path else ""))
    print(f"Offset used: {chosen_offset}")
    print(f"Frames evaluated: {n_frames}")
    print(f"Residues evaluated: {n_res}")
    print()

    mode = str(args.label_mode)
    if n_frames == 1:
        print("Per-residue labels (single frame):")
        for j, key in enumerate(selected_keys):
            if mode == "assigned":
                print(f"  {key:>10s} -> assigned={int(labels_assigned[0, j])}")
            elif mode == "halo":
                print(f"  {key:>10s} -> halo={int(labels_halo[0, j])}")
            else:
                print(
                    f"  {key:>10s} -> assigned={int(labels_assigned[0, j])}, halo={int(labels_halo[0, j])}"
                )
    else:
        print("Per-residue label summary across frames:")
        for j, key in enumerate(selected_keys):
            if mode == "assigned":
                uniq = np.unique(labels_assigned[:, j])
                print(f"  {key:>10s} -> assigned_unique={uniq.tolist()}")
            elif mode == "halo":
                uniq = np.unique(labels_halo[:, j])
                print(f"  {key:>10s} -> halo_unique={uniq.tolist()}")
            else:
                uniq_a = np.unique(labels_assigned[:, j]).tolist()
                uniq_h = np.unique(labels_halo[:, j]).tolist()
                print(f"  {key:>10s} -> assigned_unique={uniq_a}, halo_unique={uniq_h}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
