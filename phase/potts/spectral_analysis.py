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
INTERSECTION_KIND = "hamiltonian_spectral_intersection"


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


def _orient_eigenvectors(vectors: np.ndarray) -> np.ndarray:
    out = np.asarray(vectors, dtype=np.float64).copy()
    for k in range(out.shape[1]):
        col = out[:, k]
        idx = int(np.argmax(np.abs(col))) if col.size else 0
        if col.size and col[idx] < 0:
            out[:, k] = -col
    return out


def spectral_decomposition(matrix: np.ndarray, *, top_k: int = 20, sort_mode: str = "desc") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M = np.asarray(matrix, dtype=np.float64)
    M = 0.5 * (M + M.T)
    values, vectors = np.linalg.eigh(M)
    if sort_mode == "abs":
        order = np.argsort(np.abs(values))[::-1]
    elif sort_mode == "asc":
        order = np.argsort(values)
    else:
        order = np.argsort(values)[::-1]
    values_sorted = values[order]
    vectors_sorted = _orient_eigenvectors(vectors[:, order])
    top = max(1, min(int(top_k), int(vectors_sorted.shape[1])))
    top_values = values_sorted[:top]
    top_vectors = vectors_sorted[:, :top]
    return values_sorted, vectors_sorted, top_values, top_vectors


def signed_normalized_laplacian(delta_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return signed normalized graph Laplacian using absolute degree D_i=sum_j |DeltaF_ij|."""
    dF = np.asarray(delta_matrix, dtype=np.float64)
    dF = 0.5 * (dF + dF.T)
    np.fill_diagonal(dF, 0.0)
    degree = np.sum(np.abs(dF), axis=1)
    L = np.zeros_like(dF, dtype=np.float64)
    active = degree > 0
    L[active, active] = 1.0
    denom = np.sqrt(np.outer(degree, degree))
    mask = (denom > 0) & (~np.eye(dF.shape[0], dtype=bool))
    L[mask] = -dF[mask] / denom[mask]
    return 0.5 * (L + L.T), degree


def normalized_laplacian(adjacency: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized graph Laplacian for a non-negative adjacency matrix.

    This is the v3 Laplacian used for both structural single-state communities
    (A=F) and functional pair communities (A=|DeltaF|).
    """
    A = np.asarray(adjacency, dtype=np.float64)
    A = np.abs(0.5 * (A + A.T))
    np.fill_diagonal(A, 0.0)
    degree = np.sum(A, axis=1)
    L = np.zeros_like(A, dtype=np.float64)
    active = degree > 0
    L[active, active] = 1.0
    denom = np.sqrt(np.outer(degree, degree))
    mask = (denom > 0) & (~np.eye(A.shape[0], dtype=bool))
    L[mask] = -A[mask] / denom[mask]
    return 0.5 * (L + L.T), degree


def laplacian_spectral_decomposition(
    laplacian: np.ndarray,
    *,
    top_k: int = 20,
    zero_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort complete spectrum ascending, but expose top vectors as smallest non-zero modes."""
    L = np.asarray(laplacian, dtype=np.float64)
    L = 0.5 * (L + L.T)
    values, vectors = np.linalg.eigh(L)
    order = np.argsort(values)
    values_sorted = values[order]
    vectors_sorted = _orient_eigenvectors(vectors[:, order])
    nonzero = np.where(np.abs(values_sorted) > float(zero_tol))[0]
    if nonzero.size == 0:
        selected = np.arange(min(max(1, int(top_k)), values_sorted.shape[0]))
    else:
        selected = nonzero[: min(int(top_k), int(nonzero.size))]
    top_values = values_sorted[selected]
    top_vectors = vectors_sorted[:, selected]
    return values_sorted, vectors_sorted, top_values, top_vectors, selected.astype(np.int32)


def _choose_laplacian_embedding_indices(
    values: np.ndarray,
    *,
    top_k: int = 20,
    zero_tol: float = 1e-10,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    nonzero = np.where(vals > float(zero_tol))[0]
    if nonzero.size == 0:
        return np.asarray([0], dtype=np.int32) if vals.size else np.asarray([], dtype=np.int32)
    candidates = nonzero[: min(max(1, int(top_k)), int(nonzero.size))]
    if candidates.size <= 2:
        return candidates.astype(np.int32)
    candidate_values = vals[candidates]
    gaps = np.diff(candidate_values)
    if gaps.size == 0 or not np.any(np.isfinite(gaps)):
        return candidates[:2].astype(np.int32)
    # Choose all modes before the largest eigengap, with at least two modes
    # when available. This follows the usual spectral-clustering heuristic.
    k = int(np.nanargmax(gaps)) + 1
    k = max(2, min(k, int(candidates.size)))
    return candidates[:k].astype(np.int32)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=np.float64)
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    out = np.zeros_like(arr, dtype=np.float64)
    np.divide(arr, denom, out=out, where=denom > 0)
    return out


def spectral_embedding_from_laplacian(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    *,
    top_k: int = 20,
    zero_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    selected = _choose_laplacian_embedding_indices(eigenvalues, top_k=top_k, zero_tol=zero_tol)
    if selected.size == 0:
        return np.zeros((0, 0), dtype=np.float64), selected
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    embedding = _row_normalize(vectors[:, selected])
    return embedding, selected


def _fallback_communities(embedding: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = int(np.asarray(embedding).shape[0])
    labels = np.ones(n, dtype=np.int32)
    return labels, labels.copy(), {
        "method": "fallback_single_community",
        "distance_metric": "cosine",
        "n_communities": 1 if n else 0,
        "warning": "DADApy clustering was unavailable or failed; assigned one community.",
    }


def dadapy_density_peak_communities(
    embedding: np.ndarray,
    *,
    maxk: int | None = None,
    density_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cluster row-normalized spectral embedding with DADApy on cosine distances."""
    Y = _row_normalize(np.asarray(embedding, dtype=np.float64))
    n = int(Y.shape[0])
    if n <= 1 or Y.shape[1] == 0:
        return _fallback_communities(Y)
    cosine = 1.0 - np.clip(Y @ Y.T, -1.0, 1.0)
    cosine = np.clip(0.5 * (cosine + cosine.T), 0.0, 2.0)
    np.fill_diagonal(cosine, 0.0)
    kmax = int(maxk) if maxk is not None else min(n - 1, max(10, int(np.ceil(np.sqrt(n))) * 2))
    kmax = max(1, min(kmax, n - 1))
    dk = int(density_k) if density_k is not None else min(10, kmax)
    dk = max(1, min(dk, kmax))
    try:
        from dadapy.data import Data  # type: ignore

        dp_data = Data(distances=cosine, maxk=kmax, verbose=False, n_jobs=1)
        dp_data.compute_density_kNN(k=dk)
        assigned, halo = dp_data.compute_clustering_ADP(Z=1.65)
        labels = np.asarray(assigned, dtype=np.int32) + 1
        halo_labels = np.asarray(halo, dtype=np.int32) + 1
        n_communities = int(len(set(labels.tolist())))
        return labels, halo_labels, {
            "method": "dadapy_density_peak_adp",
            "distance_metric": "cosine",
            "maxk": int(kmax),
            "density_k": int(dk),
            "n_communities": n_communities,
            "cluster_centers": [int(x) for x in getattr(dp_data, "cluster_centers", [])],
        }
    except Exception as exc:
        labels, halo, diagnostics = _fallback_communities(Y)
        diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        return labels, halo, diagnostics


def _community_sizes(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    unique, counts = np.unique(arr, return_counts=True)
    order = np.argsort(unique)
    return np.column_stack([unique[order], counts[order]]).astype(np.int32)


def _community_order(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels, dtype=np.int32)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int32)
    return np.lexsort((np.arange(arr.size), arr)).astype(np.int32)


def _community_interaction_matrix(source_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    M = np.asarray(source_matrix, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int32)
    unique = np.unique(labs)
    out = np.zeros((unique.size, unique.size), dtype=np.float64)
    index = {int(label): idx for idx, label in enumerate(unique.tolist())}
    for i in range(M.shape[0]):
        ci = index.get(int(labs[i]))
        if ci is None:
            continue
        for j in range(i + 1, M.shape[1]):
            cj = index.get(int(labs[j]))
            if cj is None:
                continue
            value = float(abs(M[i, j]))
            out[ci, cj] += value
            if ci != cj:
                out[cj, ci] += value
    return out


def _laplacian_community_bundle(source_matrix: np.ndarray, *, top_k: int = 20) -> dict[str, np.ndarray | dict[str, Any]]:
    laplacian, degree = normalized_laplacian(source_matrix)
    values, vectors, top_values, top_vectors, top_indices = laplacian_spectral_decomposition(laplacian, top_k=top_k)
    embedding, embedding_indices = spectral_embedding_from_laplacian(values, vectors, top_k=top_k)
    community_ids, community_halo_ids, diagnostics = dadapy_density_peak_communities(embedding)
    order = _community_order(community_ids)
    return {
        "laplacian_matrix": laplacian,
        "laplacian_degree": degree,
        "laplacian_eigenvalues": values,
        "laplacian_top_eigenvalues": top_values,
        "laplacian_top_eigenvectors": top_vectors,
        "laplacian_top_indices": top_indices,
        "laplacian_embedding": embedding,
        "laplacian_embedding_indices": embedding_indices,
        "community_ids": community_ids,
        "community_halo_ids": community_halo_ids,
        "community_sizes": _community_sizes(community_ids),
        "community_matrix_order": order,
        "community_interaction_matrix": _community_interaction_matrix(source_matrix, community_ids),
        "community_diagnostics": diagnostics,
    }


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


def _intersection_id(single_analysis_id: str, pair_analysis_id: str, min_group_size: int) -> str:
    return f"intersection_{_safe_id(single_analysis_id)}__{_safe_id(pair_analysis_id)}__min{int(min_group_size)}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_convert_nan_to_none(payload), indent=2), encoding="utf-8")


def _analysis_exists(cluster_dir: Path, kind: str, analysis_id: str) -> bool:
    d = cluster_dir / "analyses" / kind / analysis_id
    return (d / ANALYSIS_METADATA_FILENAME).exists() and (d / "analysis.npz").exists()


def _npz_has_fields(npz_path: Path, required: set[str]) -> bool:
    if not npz_path.exists():
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            return required.issubset(set(data.files))
    except Exception:
        return False


def _single_npz_has_v3_fields(npz_path: Path) -> bool:
    return _npz_has_fields(
        npz_path,
        {
            "laplacian_matrix",
            "laplacian_source_matrix",
            "laplacian_embedding",
            "laplacian_embedding_indices",
            "community_ids",
            "community_sizes",
            "community_matrix_order",
            "community_interaction_matrix",
        },
    )


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
    if not overwrite and meta_path.exists() and _single_npz_has_v3_fields(npz_path):
        return {"metadata": _read_json(meta_path), "analysis_npz": str(npz_path), "created": False}

    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_path = _resolve_model_path(system_dir, model_entry)
    model = load_potts_model(model_path)
    F = frobenius_coupling_matrix(model)
    eigenvalues, _, top_values, top_vectors = spectral_decomposition(F, top_k=top_k, sort_mode="desc")
    community = _laplacian_community_bundle(F, top_k=top_k)
    residue_keys = _load_residue_keys(cluster_dir, F.shape[0])
    strength = np.asarray(F.sum(axis=1), dtype=np.float32)
    community_diagnostics = community["community_diagnostics"]
    np.savez_compressed(
        npz_path,
        analysis_format_version=np.asarray([3], dtype=np.int32),
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
        laplacian_source_matrix=np.asarray(F, dtype=np.float32),
        laplacian_matrix=np.asarray(community["laplacian_matrix"], dtype=np.float32),
        laplacian_degree=np.asarray(community["laplacian_degree"], dtype=np.float32),
        laplacian_eigenvalues=np.asarray(community["laplacian_eigenvalues"], dtype=np.float32),
        laplacian_top_eigenvalues=np.asarray(community["laplacian_top_eigenvalues"], dtype=np.float32),
        laplacian_top_eigenvectors=np.asarray(community["laplacian_top_eigenvectors"], dtype=np.float32).T,
        laplacian_top_indices=np.asarray(community["laplacian_top_indices"], dtype=np.int32),
        laplacian_embedding=np.asarray(community["laplacian_embedding"], dtype=np.float32),
        laplacian_embedding_indices=np.asarray(community["laplacian_embedding_indices"], dtype=np.int32),
        community_ids=np.asarray(community["community_ids"], dtype=np.int32),
        community_halo_ids=np.asarray(community["community_halo_ids"], dtype=np.int32),
        community_sizes=np.asarray(community["community_sizes"], dtype=np.int32),
        community_matrix_order=np.asarray(community["community_matrix_order"], dtype=np.int32),
        community_interaction_matrix=np.asarray(community["community_interaction_matrix"], dtype=np.float32),
        community_diagnostics_json=np.asarray([json.dumps(_convert_nan_to_none(community_diagnostics))], dtype=str),
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
            "laplacian_top_k": int(np.asarray(community["laplacian_top_eigenvectors"]).shape[1]),
            "laplacian_embedding_k": int(np.asarray(community["laplacian_embedding"]).shape[1]),
            "n_communities": int(np.asarray(community["community_sizes"]).shape[0]),
            "community_method": str(community_diagnostics.get("method") if isinstance(community_diagnostics, dict) else ""),
        },
    }
    _write_json(meta_path, meta)
    return {"metadata": meta, "analysis_npz": str(npz_path), "created": True}


def _pair_npz_has_v3_fields(npz_path: Path) -> bool:
    return _npz_has_fields(
        npz_path,
        {
            "laplacian_matrix",
            "laplacian_source_matrix",
            "laplacian_degree",
            "laplacian_eigenvalues",
            "laplacian_top_eigenvalues",
            "laplacian_top_eigenvectors",
            "laplacian_top_indices",
            "laplacian_embedding",
            "laplacian_embedding_indices",
            "community_ids",
            "community_sizes",
            "community_matrix_order",
            "community_interaction_matrix",
        },
    )


def _load_analysis_npz_and_meta(cluster_dir: Path, kind: str, analysis_id: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    analysis_dir = cluster_dir / "analyses" / kind / analysis_id
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
    npz_path = analysis_dir / "analysis.npz"
    if not meta_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Analysis {kind}/{analysis_id} not found.")
    meta = _read_json(meta_path)
    with np.load(npz_path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    return payload, meta


def _community_size_map(labels: np.ndarray) -> dict[int, int]:
    arr = np.asarray(labels, dtype=np.int32)
    unique, counts = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts, strict=False)}


def _residue_label_list(residue_keys: np.ndarray, indices: list[int]) -> list[str]:
    keys = np.asarray(residue_keys, dtype=str)
    return [str(keys[i]) if 0 <= i < keys.shape[0] else f"res_{i + 1}" for i in indices]


def compute_spectral_intersection_analysis(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_id: str,
    single_analysis_id: str,
    pair_analysis_id: str,
    min_group_size: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Intersect structural and functional DADApy community labels in O(N)."""
    dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    cluster_dir = dirs["cluster_dir"]
    min_group_size = max(1, int(min_group_size))
    analysis_id = _intersection_id(single_analysis_id, pair_analysis_id, min_group_size)
    analysis_dir = cluster_dir / "analyses" / INTERSECTION_KIND / analysis_id
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
    npz_path = analysis_dir / "analysis.npz"
    if not overwrite and meta_path.exists() and npz_path.exists():
        return {"metadata": _read_json(meta_path), "analysis_npz": str(npz_path), "created": False}

    single, single_meta = _load_analysis_npz_and_meta(cluster_dir, SINGLE_KIND, single_analysis_id)
    pair, pair_meta = _load_analysis_npz_and_meta(cluster_dir, PAIR_KIND, pair_analysis_id)
    required = ("residue_keys", "community_ids")
    for key in required:
        if key not in single:
            raise ValueError(f"Single analysis {single_analysis_id} lacks required field {key}; rerun spectral analysis.")
        if key not in pair:
            raise ValueError(f"Pair analysis {pair_analysis_id} lacks required field {key}; rerun spectral analysis.")

    residue_keys = np.asarray(single["residue_keys"], dtype=str)
    pair_keys = np.asarray(pair["residue_keys"], dtype=str)
    if residue_keys.shape[0] != pair_keys.shape[0]:
        raise ValueError("Single and pair analyses have different residue counts.")
    if not np.array_equal(residue_keys.astype(str), pair_keys.astype(str)):
        raise ValueError("Single and pair analyses use different residue key orderings.")

    struct = np.asarray(single["community_ids"], dtype=np.int32)
    func = np.asarray(pair["community_ids"], dtype=np.int32)
    n_res = int(residue_keys.shape[0])
    if struct.shape[0] != n_res or func.shape[0] != n_res:
        raise ValueError("Community label arrays do not match residue count.")

    groups: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(zip(struct.tolist(), func.tolist(), strict=False)):
        groups.setdefault((int(key[0]), int(key[1])), []).append(int(idx))

    piston_groups = [(key, members) for key, members in groups.items() if len(members) >= min_group_size]
    piston_groups.sort(key=lambda row: (-len(row[1]), row[0][0], row[0][1]))
    piston_ids = np.zeros(n_res, dtype=np.int32)
    piston_group_ids: list[int] = []
    piston_struct_ids: list[int] = []
    piston_func_ids: list[int] = []
    piston_sizes: list[int] = []
    piston_members: list[dict[str, Any]] = []
    for gid, (key, members) in enumerate(piston_groups, start=1):
        for idx in members:
            piston_ids[idx] = gid
        piston_group_ids.append(gid)
        piston_struct_ids.append(int(key[0]))
        piston_func_ids.append(int(key[1]))
        piston_sizes.append(int(len(members)))
        piston_members.append(
            {
                "piston_id": gid,
                "structural_community_id": int(key[0]),
                "functional_community_id": int(key[1]),
                "size": int(len(members)),
                "residue_indices": [int(i) for i in members],
                "residue_keys": _residue_label_list(residue_keys, members),
                "tooltip": "A cohesive structural unit that rewires its correlations collectively. These residues form a solid mechanical gear that shifts during activation.",
            }
        )

    struct_sizes = _community_size_map(struct)
    func_sizes = _community_size_map(func)
    class_codes = np.zeros(n_res, dtype=np.int32)
    # 3=piston, 1=scaffold, 2=transient switch, 0=other
    class_codes[piston_ids > 0] = 3
    for idx in range(n_res):
        if class_codes[idx] == 3:
            continue
        s = int(struct[idx])
        f = int(func[idx])
        s_size = int(struct_sizes.get(s, 0))
        f_size = int(func_sizes.get(f, 0))
        if s > 0 and s_size >= min_group_size and (f <= 0 or len(groups.get((s, f), [])) < min_group_size):
            class_codes[idx] = 1
        elif (s <= 0 or s_size < min_group_size) and f > 0 and f_size >= min_group_size:
            class_codes[idx] = 2

    class_names = np.asarray(["other", "structural_scaffold", "transient_switch", "allosteric_piston"], dtype=str)
    class_counts = np.asarray(
        [[code, int(np.count_nonzero(class_codes == code))] for code in range(4)],
        dtype=np.int32,
    )
    combo_struct = np.asarray([int(key[0]) for key, _ in sorted(groups.items())], dtype=np.int32)
    combo_func = np.asarray([int(key[1]) for key, _ in sorted(groups.items())], dtype=np.int32)
    combo_sizes = np.asarray([len(members) for _, members in sorted(groups.items())], dtype=np.int32)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        analysis_format_version=np.asarray([1], dtype=np.int32),
        mode=np.asarray(["intersection"], dtype=str),
        residue_keys=residue_keys,
        structural_community_ids=struct,
        functional_community_ids=func,
        piston_ids=piston_ids,
        residue_class_codes=class_codes,
        residue_class_names=class_names,
        class_counts=class_counts,
        piston_group_ids=np.asarray(piston_group_ids, dtype=np.int32),
        piston_structural_community_ids=np.asarray(piston_struct_ids, dtype=np.int32),
        piston_functional_community_ids=np.asarray(piston_func_ids, dtype=np.int32),
        piston_sizes=np.asarray(piston_sizes, dtype=np.int32),
        composite_structural_community_ids=combo_struct,
        composite_functional_community_ids=combo_func,
        composite_group_sizes=combo_sizes,
        piston_members_json=np.asarray([json.dumps(_convert_nan_to_none(piston_members))], dtype=str),
    )
    now = _utc_now()
    meta = {
        "analysis_id": analysis_id,
        "analysis_type": INTERSECTION_KIND,
        "mode": "intersection",
        "created_at": now,
        "updated_at": now,
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "single_analysis_id": single_analysis_id,
        "pair_analysis_id": pair_analysis_id,
        "single_state_id": single_meta.get("state_id"),
        "single_state_name": single_meta.get("state_name"),
        "pair_state_a_id": pair_meta.get("state_a_id"),
        "pair_state_b_id": pair_meta.get("state_b_id"),
        "pair_state_a_name": pair_meta.get("state_a_name"),
        "pair_state_b_name": pair_meta.get("state_b_name"),
        "min_group_size": int(min_group_size),
        "summary": {
            "n_residues": n_res,
            "n_structural_communities": int(len(struct_sizes)),
            "n_functional_communities": int(len(func_sizes)),
            "n_composite_groups": int(len(groups)),
            "n_pistons": int(len(piston_groups)),
            "piston_residues": int(np.count_nonzero(piston_ids > 0)),
            "structural_scaffold_residues": int(np.count_nonzero(class_codes == 1)),
            "transient_switch_residues": int(np.count_nonzero(class_codes == 2)),
        },
    }
    _write_json(meta_path, meta)
    return {"metadata": meta, "analysis_npz": str(npz_path), "created": True}


def upsert_spectral_intersection_analysis(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    single_analysis_id: str,
    pair_analysis_id: str,
    min_group_size: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    out = compute_spectral_intersection_analysis(
        store=store,
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        single_analysis_id=single_analysis_id,
        pair_analysis_id=pair_analysis_id,
        min_group_size=min_group_size,
        overwrite=overwrite,
    )
    return {
        "analysis_type": INTERSECTION_KIND,
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "analysis": out["metadata"],
        "analysis_npz": out["analysis_npz"],
        "created": bool(out.get("created")),
    }


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
    if not overwrite and meta_path.exists() and _pair_npz_has_v3_fields(npz_path):
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
    laplacian_source = np.abs(dF)
    community = _laplacian_community_bundle(laplacian_source, top_k=top_k)
    community_diagnostics = community["community_diagnostics"]
    np.savez_compressed(
        npz_path,
        analysis_format_version=np.asarray([3], dtype=np.int32),
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
        laplacian_source_matrix=np.asarray(laplacian_source, dtype=np.float32),
        laplacian_matrix=np.asarray(community["laplacian_matrix"], dtype=np.float32),
        laplacian_degree=np.asarray(community["laplacian_degree"], dtype=np.float32),
        laplacian_eigenvalues=np.asarray(community["laplacian_eigenvalues"], dtype=np.float32),
        laplacian_top_eigenvalues=np.asarray(community["laplacian_top_eigenvalues"], dtype=np.float32),
        laplacian_top_eigenvectors=np.asarray(community["laplacian_top_eigenvectors"], dtype=np.float32).T,
        laplacian_top_indices=np.asarray(community["laplacian_top_indices"], dtype=np.int32),
        laplacian_embedding=np.asarray(community["laplacian_embedding"], dtype=np.float32),
        laplacian_embedding_indices=np.asarray(community["laplacian_embedding_indices"], dtype=np.int32),
        community_ids=np.asarray(community["community_ids"], dtype=np.int32),
        community_halo_ids=np.asarray(community["community_halo_ids"], dtype=np.int32),
        community_sizes=np.asarray(community["community_sizes"], dtype=np.int32),
        community_matrix_order=np.asarray(community["community_matrix_order"], dtype=np.int32),
        community_interaction_matrix=np.asarray(community["community_interaction_matrix"], dtype=np.float32),
        community_diagnostics_json=np.asarray([json.dumps(_convert_nan_to_none(community_diagnostics))], dtype=str),
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
            "laplacian_top_k": int(np.asarray(community["laplacian_top_eigenvectors"]).shape[1]),
            "laplacian_embedding_k": int(np.asarray(community["laplacian_embedding"]).shape[1]),
            "laplacian_first_nonzero_eigenvalue": float(np.asarray(community["laplacian_top_eigenvalues"])[0]) if np.asarray(community["laplacian_top_eigenvalues"]).size else None,
            "laplacian_zero_degree_residues": int(np.count_nonzero(np.asarray(community["laplacian_degree"]) <= 0)),
            "n_communities": int(np.asarray(community["community_sizes"]).shape[0]),
            "community_method": str(community_diagnostics.get("method") if isinstance(community_diagnostics, dict) else ""),
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
        if not overwrite and _pair_npz_has_v3_fields(cluster_dir / "analyses" / PAIR_KIND / _pair_id(a, b) / "analysis.npz"):
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
