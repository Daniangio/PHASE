from __future__ import annotations

import json
import os
import re
import uuid
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from phase.potts.potts_model import load_potts_model, zero_sum_gauge_model
from phase.services.project_store import ProjectStore

ANALYSIS_METADATA_FILENAME = "analysis_metadata.json"
SINGLE_KIND = "hamiltonian_spectral_single"
PAIR_KIND = "hamiltonian_spectral_pair"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    value = str(value or "").strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "state"


def _convert_nan_to_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _convert_nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return [_convert_nan_to_none(v) for v in obj]
    if isinstance(obj, np.generic):
        return _convert_nan_to_none(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _state_names(store: ProjectStore, project_id: str, system_id: str) -> dict[str, str]:
    try:
        system = store.get_system(project_id, system_id)
    except Exception:
        return {}
    out: dict[str, str] = {}
    for sid, state in (system.states or {}).items():
        out[str(sid)] = str(getattr(state, "name", None) or sid)
    return out


def _resolve_model_path(system_dir: Path, model_entry: dict[str, Any]) -> Path:
    raw = str(model_entry.get("path") or "").strip()
    if not raw:
        raise FileNotFoundError(f"Potts model {model_entry.get('model_id')} has no path.")
    p = Path(raw)
    if not p.is_absolute():
        p = system_dir / p
    return p


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _states_from_pdbs(value: Any) -> list[str]:
    states: list[str] = []
    for item in _as_str_list(value):
        parts = Path(item).parts
        for idx, part in enumerate(parts[:-1]):
            if part == "states" and idx + 1 < len(parts):
                states.append(str(parts[idx + 1]))
                break
    seen: set[str] = set()
    return [s for s in states if not (s in seen or seen.add(s))]


def _model_has_full_hamiltonian(model: dict[str, Any]) -> bool:
    params = model.get("params") if isinstance(model.get("params"), dict) else {}
    name = str(model.get("name") or "").lower()
    path = str(model.get("path") or "").lower()
    delta_kind = str(params.get("delta_kind") or "").lower()
    # Delta-patch files are not complete endpoint Hamiltonians. Combined/model-patch files are.
    if "(delta)" in name or path.endswith("_delta.npz"):
        return False
    if delta_kind and "delta" in delta_kind and "combined" not in name and "combined" not in path and "model_patch" not in delta_kind:
        return False
    return True


def _candidate_states_for_model(model: dict[str, Any], known_states: set[str]) -> tuple[list[str], int]:
    params = model.get("params") if isinstance(model.get("params"), dict) else {}
    explicit = _as_str_list(params.get("state_ids") or model.get("state_ids") or model.get("fit_sample_state_ids"))
    explicit = [sid for sid in explicit if sid in known_states]
    if len(explicit) == 1:
        return explicit, 100

    pdb_states = [sid for sid in _states_from_pdbs(params.get("pdbs")) if sid in known_states]
    if len(pdb_states) == 1:
        return pdb_states, 90

    name = str(model.get("name") or "")
    path_stem = Path(str(model.get("path") or "")).stem
    text_candidates = {name, path_stem}
    matches: list[str] = []
    for sid in known_states:
        accepted = {sid, f"model_{sid}", f"potts_{sid}"}
        if any(txt in accepted for txt in text_candidates):
            matches.append(sid)
    if len(matches) == 1:
        return matches, 60
    return [], 0


def resolve_state_models(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_id: str,
    state_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Map selected state ids to one unambiguous full-Hamiltonian Potts model each."""
    requested = [str(s).strip() for s in state_ids if str(s).strip()]
    known_states = set(_state_names(store, project_id, system_id).keys()) | set(requested)
    models = store.list_potts_models(project_id, system_id, cluster_id)
    candidates: dict[str, list[tuple[int, str, dict[str, Any]]]] = {sid: [] for sid in requested}
    for model in models:
        if not _model_has_full_hamiltonian(model):
            continue
        mids = str(model.get("model_id") or "")
        states, score = _candidate_states_for_model(model, known_states)
        if len(states) != 1:
            continue
        sid = states[0]
        if sid not in candidates:
            continue
        params = model.get("params") if isinstance(model.get("params"), dict) else {}
        # Prefer standard single-state fits over combined delta endpoint models when both exist.
        fit_mode = str(params.get("fit_mode") or "").lower()
        if fit_mode == "standard" or not str(model.get("source") or "").startswith("offline_delta"):
            score += 10
        updated = str(model.get("updated_at") or model.get("created_at") or "")
        candidates[sid].append((score, updated, model))
    resolved: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for sid, rows in candidates.items():
        if not rows:
            skipped[sid] = "No unambiguous single-state full Potts model was found."
            continue
        rows = sorted(rows, key=lambda x: (x[0], x[1]), reverse=True)
        best_score = rows[0][0]
        best = [r for r in rows if r[0] == best_score]
        if len(best) > 1 and str(best[0][1]) == str(best[1][1]):
            skipped[sid] = "Multiple equally ranked Potts models matched this state."
            continue
        resolved[sid] = rows[0][2]
    return resolved, skipped


def frobenius_coupling_matrix(model: Any) -> np.ndarray:
    gauged = zero_sum_gauge_model(model)
    n_res = len(gauged.h)
    F = np.zeros((n_res, n_res), dtype=np.float64)
    for r, s in gauged.edges:
        mat = np.asarray(gauged.J[(int(r), int(s))], dtype=np.float64)
        val = float(np.sqrt(np.sum(mat * mat)))
        F[int(r), int(s)] = val
        F[int(s), int(r)] = val
    np.fill_diagonal(F, 0.0)
    return 0.5 * (F + F.T)


def spectral_decomposition(matrix: np.ndarray, *, top_k: int = 20, sort_mode: str = "desc") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M = np.asarray(matrix, dtype=np.float64)
    M = 0.5 * (M + M.T)
    values, vectors = np.linalg.eigh(M)
    if sort_mode == "abs":
        order = np.argsort(np.abs(values))[::-1]
    else:
        order = np.argsort(values)[::-1]
    values_sorted = values[order]
    vectors_sorted = vectors[:, order]
    top = max(1, min(int(top_k), int(vectors_sorted.shape[1])))
    top_values = values_sorted[:top]
    top_vectors = vectors_sorted[:, :top]
    # Make eigenvector sign deterministic for stable visualization.
    for k in range(top_vectors.shape[1]):
        col = top_vectors[:, k]
        idx = int(np.argmax(np.abs(col))) if col.size else 0
        if col.size and col[idx] < 0:
            top_vectors[:, k] = -col
    return values_sorted, vectors_sorted, top_values, top_vectors


def _load_residue_keys(cluster_dir: Path, n_res: int) -> np.ndarray:
    cluster_npz = cluster_dir / "cluster.npz"
    if cluster_npz.exists():
        try:
            with np.load(cluster_npz, allow_pickle=False) as data:
                if "residue_keys" in data:
                    arr = np.asarray(data["residue_keys"], dtype=str)
                    if arr.shape[0] == n_res:
                        return arr
        except Exception:
            pass
    return np.asarray([f"res_{i + 1}" for i in range(n_res)], dtype=str)


def _single_id(state_id: str) -> str:
    return f"single_{_safe_id(state_id)}"


def _pair_id(state_a_id: str, state_b_id: str) -> str:
    return f"pair_{_safe_id(state_a_id)}__{_safe_id(state_b_id)}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_convert_nan_to_none(payload), indent=2), encoding="utf-8")


def _analysis_exists(cluster_dir: Path, kind: str, analysis_id: str) -> bool:
    d = cluster_dir / "analyses" / kind / analysis_id
    return (d / ANALYSIS_METADATA_FILENAME).exists() and (d / "analysis.npz").exists()


def compute_single_spectral_analysis(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_id: str,
    state_id: str,
    model_entry: dict[str, Any],
    top_k: int = 20,
    overwrite: bool = False,
) -> dict[str, Any]:
    dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = dirs["system_dir"]
    cluster_dir = dirs["cluster_dir"]
    state_names = _state_names(store, project_id, system_id)
    analysis_id = _single_id(state_id)
    analysis_dir = cluster_dir / "analyses" / SINGLE_KIND / analysis_id
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
    npz_path = analysis_dir / "analysis.npz"
    if not overwrite and meta_path.exists() and npz_path.exists():
        return {"metadata": _read_json(meta_path), "analysis_npz": str(npz_path), "created": False}

    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_path = _resolve_model_path(system_dir, model_entry)
    model = load_potts_model(model_path)
    F = frobenius_coupling_matrix(model)
    eigenvalues, _, top_values, top_vectors = spectral_decomposition(F, top_k=top_k, sort_mode="desc")
    residue_keys = _load_residue_keys(cluster_dir, F.shape[0])
    strength = np.asarray(F.sum(axis=1), dtype=np.float32)
    np.savez_compressed(
        npz_path,
        analysis_format_version=np.asarray([1], dtype=np.int32),
        mode=np.asarray(["single"], dtype=str),
        state_id=np.asarray([state_id], dtype=str),
        state_name=np.asarray([state_names.get(state_id, state_id)], dtype=str),
        model_id=np.asarray([str(model_entry.get("model_id") or "")], dtype=str),
        model_name=np.asarray([str(model_entry.get("name") or model_entry.get("model_id") or "")], dtype=str),
        residue_keys=residue_keys,
        matrix=np.asarray(F, dtype=np.float32),
        residue_strength=strength,
        eigenvalues=np.asarray(eigenvalues, dtype=np.float32),
        top_eigenvalues=np.asarray(top_values, dtype=np.float32),
        top_eigenvectors=np.asarray(top_vectors.T, dtype=np.float32),
    )
    now = _utc_now()
    meta = {
        "analysis_id": analysis_id,
        "analysis_type": SINGLE_KIND,
        "mode": "single",
        "created_at": now,
        "updated_at": now,
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "state_id": state_id,
        "state_name": state_names.get(state_id, state_id),
        "model_id": str(model_entry.get("model_id") or ""),
        "model_name": str(model_entry.get("name") or model_entry.get("model_id") or ""),
        "model_path": str(model_entry.get("path") or ""),
        "summary": {
            "n_residues": int(F.shape[0]),
            "n_edges": int(np.count_nonzero(np.triu(F, 1))),
            "top_k": int(top_vectors.shape[1]),
            "largest_eigenvalue": float(top_values[0]) if top_values.size else None,
        },
    }
    _write_json(meta_path, meta)
    return {"metadata": meta, "analysis_npz": str(npz_path), "created": True}


def compute_pair_spectral_analysis(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_id: str,
    state_a_id: str,
    state_b_id: str,
    top_k: int = 20,
    overwrite: bool = False,
) -> dict[str, Any]:
    dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    cluster_dir = dirs["cluster_dir"]
    state_names = _state_names(store, project_id, system_id)
    analysis_id = _pair_id(state_a_id, state_b_id)
    analysis_dir = cluster_dir / "analyses" / PAIR_KIND / analysis_id
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
    npz_path = analysis_dir / "analysis.npz"
    if not overwrite and meta_path.exists() and npz_path.exists():
        return {"metadata": _read_json(meta_path), "analysis_npz": str(npz_path), "created": False}

    single_a = cluster_dir / "analyses" / SINGLE_KIND / _single_id(state_a_id) / "analysis.npz"
    single_b = cluster_dir / "analyses" / SINGLE_KIND / _single_id(state_b_id) / "analysis.npz"
    if not single_a.exists() or not single_b.exists():
        raise FileNotFoundError(f"Missing single spectral analysis for pair {state_a_id}, {state_b_id}.")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with np.load(single_a, allow_pickle=False) as da, np.load(single_b, allow_pickle=False) as db:
        F_a = np.asarray(da["matrix"], dtype=np.float64)
        F_b = np.asarray(db["matrix"], dtype=np.float64)
        residue_keys = np.asarray(da["residue_keys"], dtype=str)
    if F_a.shape != F_b.shape:
        raise ValueError(f"Single-state Frobenius matrices have different shapes for {state_a_id} and {state_b_id}.")
    dF = 0.5 * ((F_b - F_a) + (F_b - F_a).T)
    eigenvalues, _, top_values, top_vectors = spectral_decomposition(dF, top_k=top_k, sort_mode="abs")
    np.savez_compressed(
        npz_path,
        analysis_format_version=np.asarray([1], dtype=np.int32),
        mode=np.asarray(["pair"], dtype=str),
        state_a_id=np.asarray([state_a_id], dtype=str),
        state_b_id=np.asarray([state_b_id], dtype=str),
        state_a_name=np.asarray([state_names.get(state_a_id, state_a_id)], dtype=str),
        state_b_name=np.asarray([state_names.get(state_b_id, state_b_id)], dtype=str),
        residue_keys=residue_keys,
        matrix=np.asarray(dF, dtype=np.float32),
        residue_strength=np.asarray(np.sum(np.abs(dF), axis=1), dtype=np.float32),
        eigenvalues=np.asarray(eigenvalues, dtype=np.float32),
        top_eigenvalues=np.asarray(top_values, dtype=np.float32),
        top_eigenvectors=np.asarray(top_vectors.T, dtype=np.float32),
    )
    now = _utc_now()
    meta = {
        "analysis_id": analysis_id,
        "analysis_type": PAIR_KIND,
        "mode": "pair",
        "created_at": now,
        "updated_at": now,
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "state_a_id": state_a_id,
        "state_b_id": state_b_id,
        "state_a_name": state_names.get(state_a_id, state_a_id),
        "state_b_name": state_names.get(state_b_id, state_b_id),
        "summary": {
            "n_residues": int(dF.shape[0]),
            "n_changed_edges": int(np.count_nonzero(np.triu(np.abs(dF) > 0, 1))),
            "top_k": int(top_vectors.shape[1]),
            "principal_abs_eigenvalue": float(top_values[0]) if top_values.size else None,
        },
    }
    _write_json(meta_path, meta)
    return {"metadata": meta, "analysis_npz": str(npz_path), "created": True}


def upsert_hamiltonian_spectral_batch(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    state_ids: Sequence[str],
    top_k: int = 20,
    overwrite: bool = False,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    cluster_dir = dirs["cluster_dir"]
    requested: list[str] = []
    seen: set[str] = set()
    for raw in state_ids:
        sid = str(raw).strip()
        if sid and sid not in seen:
            seen.add(sid)
            requested.append(sid)
    if not requested:
        raise ValueError("Select at least one state.")

    state_models, skipped = resolve_state_models(
        store=store,
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        state_ids=requested,
    )
    total = len(state_models)
    if progress_callback:
        progress_callback("Computing single-state Hamiltonian spectra", 0, max(1, total))
    singles: list[dict[str, Any]] = []
    for idx, sid in enumerate(requested):
        model = state_models.get(sid)
        if not model:
            continue
        out = compute_single_spectral_analysis(
            store=store,
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            state_id=sid,
            model_entry=model,
            top_k=top_k,
            overwrite=overwrite,
        )
        singles.append(out["metadata"])
        if progress_callback:
            progress_callback("Computing single-state Hamiltonian spectra", len(singles), max(1, total))

    existing_single_ids: set[str] = set()
    single_root = cluster_dir / "analyses" / SINGLE_KIND
    if single_root.exists():
        for d in single_root.iterdir():
            meta = d / ANALYSIS_METADATA_FILENAME
            npz = d / "analysis.npz"
            if meta.exists() and npz.exists():
                try:
                    sid = str(_read_json(meta).get("state_id") or "")
                    if sid:
                        existing_single_ids.add(sid)
                except Exception:
                    pass
    pair_candidates = sorted(
        (a, b) for a, b in combinations(sorted(existing_single_ids), 2) if a in requested or b in requested
    )
    pair_total = len(pair_candidates)
    if progress_callback:
        progress_callback("Computing pair Hamiltonian spectra", 0, max(1, pair_total))
    pairs: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(pair_candidates):
        if not overwrite and _analysis_exists(cluster_dir, PAIR_KIND, _pair_id(a, b)):
            continue
        out = compute_pair_spectral_analysis(
            store=store,
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            state_a_id=a,
            state_b_id=b,
            top_k=top_k,
            overwrite=overwrite,
        )
        pairs.append(out["metadata"])
        if progress_callback:
            progress_callback("Computing pair Hamiltonian spectra", idx + 1, max(1, pair_total))

    return {
        "analysis_type": "hamiltonian_spectral_batch",
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "requested_state_ids": requested,
        "single_count": len(singles),
        "pair_count": len(pairs),
        "skipped_states": skipped,
        "single_analyses": singles,
        "pair_analyses": pairs,
    }
