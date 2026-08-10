from __future__ import annotations

import json
import hashlib
import os
import pickle
import re
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from phase.io.data import load_npz
from phase.potts.analysis_run import (
    ANALYSIS_METADATA_FILENAME,
    _compute_contact_edges_from_pdbs,
    _completion_cost_from_curve,
    _convert_nan_to_none,
    _delta_js_row_node_edge_values,
    _ensure_analysis_dir,
    _purge_matching_sampling_analyses,
    _resolve_model_for_analysis,
    compute_lambda_sweep_analysis,
    compute_md_vs_sample_metrics,
    compute_sample_energies,
    _gibbs_relax_worker,
    _normalize_js_filter_rules,
    _normalized_auc,
    _passes_js_filter_rules,
    _relativize,
    _single_frame_ligand_completion_worker,
    _utc_now,
)
from phase.potts.metrics import js_divergence, marginals, pairwise_joints_on_edges
from phase.potts.potts_model import add_potts_models, interpolate_potts_models, load_potts_model, zero_sum_gauge_model
from phase.potts.sample_io import SAMPLE_NPZ_FILENAME, load_sample_npz, save_sample_npz
from phase.services.project_store import ProjectStore


ProgressCallback = Optional[Callable[[str, int, int], None]]
_POTTS_ANALYSIS_SAMPLE_CACHE: dict[tuple[str, bool, str, bool], dict[str, Any]] = {}
_POTTS_ANALYSIS_MODEL_CACHE: dict[str, Any] = {}


def _load_cluster_residue_mapping(cluster_npz_path: Path) -> tuple[list[str], dict[str, str]]:
    residue_keys: list[str] = []
    residue_mapping: dict[str, str] = {}
    try:
        with np.load(cluster_npz_path, allow_pickle=True) as cluster_npz:
            if "residue_keys" in cluster_npz:
                residue_keys = [str(v) for v in np.asarray(cluster_npz["residue_keys"]).tolist()]
            raw_meta = cluster_npz.get("metadata_json")
            if raw_meta is not None:
                meta_json = raw_meta.item() if isinstance(raw_meta, np.ndarray) else raw_meta
                if isinstance(meta_json, bytes):
                    meta_json = meta_json.decode("utf-8", errors="ignore")
                meta = json.loads(str(meta_json)) if meta_json else {}
                if isinstance(meta, dict):
                    residue_mapping = {str(k): str(v) for k, v in (meta.get("residue_mapping") or {}).items()}
    except Exception:
        return residue_keys, residue_mapping
    return residue_keys, residue_mapping


def _resolve_analysis_contact_edges(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    md_samples: Sequence[dict[str, Any]],
    state_ids: Sequence[str] | None,
    pdbs: Sequence[str] | None,
    cluster_npz_path: Path,
    cutoff: float,
    atom_mode: str,
) -> list[tuple[int, int]]:
    residue_keys, residue_mapping = _load_cluster_residue_mapping(cluster_npz_path)
    if not residue_keys:
        return []

    system_meta = store.get_system(project_id, system_id)
    explicit_state_ids = [str(v).strip() for v in (state_ids or []) if str(v).strip()]
    inferred_state_ids: list[str] = []
    if not explicit_state_ids:
        inferred_state_ids = sorted(
            {
                str(sample.get("state_id") or "").strip()
                for sample in (md_samples or [])
                if str(sample.get("state_id") or "").strip()
            }
        )
    all_state_ids = explicit_state_ids or inferred_state_ids
    if not all_state_ids:
        all_state_ids = [
            str(state_id)
            for state_id, state in (system_meta.states or {}).items()
            if getattr(state, "pdb_file", None)
        ]

    pdb_paths: list[Path] = []
    for state_id in all_state_ids:
        state = (system_meta.states or {}).get(state_id)
        if state is None:
            continue
        pdb_rel = getattr(state, "pdb_file", None)
        if not pdb_rel:
            continue
        p = store.resolve_path(project_id, system_id, str(pdb_rel))
        if p.exists():
            pdb_paths.append(p)
    for value in (pdbs or []):
        raw = str(value).strip()
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = store.resolve_path(project_id, system_id, raw)
        if p.exists():
            pdb_paths.append(p)

    dedup_paths: list[Path] = []
    seen: set[str] = set()
    for p in pdb_paths:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        dedup_paths.append(Path(rp))
    if not dedup_paths:
        return []
    return _compute_contact_edges_from_pdbs(
        dedup_paths,
        residue_keys,
        residue_mapping,
        float(cutoff),
        str(atom_mode or "CA").upper(),
    )


def _derive_residue_display_labels(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_dir: Path,
    residue_keys: Sequence[str],
) -> list[str]:
    labels = [str(v) for v in residue_keys]
    if not labels:
        return labels

    residue_mapping: dict[str, str] = {}
    selected_state_ids: list[str] = []
    cluster_npz_path = cluster_dir / "cluster.npz"
    if cluster_npz_path.exists():
        try:
            with np.load(cluster_npz_path, allow_pickle=True) as cluster_npz:
                raw_meta = cluster_npz.get("metadata_json")
                if raw_meta is not None:
                    meta_json = raw_meta.item() if isinstance(raw_meta, np.ndarray) else raw_meta
                    if isinstance(meta_json, bytes):
                        meta_json = meta_json.decode("utf-8", errors="ignore")
                    meta = json.loads(str(meta_json)) if meta_json else {}
                    if isinstance(meta, dict):
                        residue_mapping = {str(k): str(v) for k, v in (meta.get("residue_mapping") or {}).items()}
                        selected_state_ids = [str(v) for v in (meta.get("selected_state_ids") or meta.get("selected_metastable_ids") or [])]
        except Exception:
            residue_mapping = {}
            selected_state_ids = []

    try:
        system_meta = store.get_system(project_id, system_id)
    except Exception:
        return labels

    state_pdb: Path | None = None
    candidate_state_ids = [sid for sid in selected_state_ids if sid in (system_meta.states or {})]
    if not candidate_state_ids:
        candidate_state_ids = [str(sid) for sid, st in (system_meta.states or {}).items() if getattr(st, "pdb_file", None)]
    for state_id in candidate_state_ids:
        state = (system_meta.states or {}).get(state_id)
        if state is None:
            continue
        if not residue_mapping and getattr(state, "residue_mapping", None):
            residue_mapping = {str(k): str(v) for k, v in (state.residue_mapping or {}).items()}
        pdb_rel = getattr(state, "pdb_file", None)
        if pdb_rel:
            candidate = store.resolve_path(project_id, system_id, str(pdb_rel))
            if candidate.exists():
                state_pdb = candidate
                break
    if state_pdb is None:
        return labels

    try:
        import MDAnalysis as mda  # type: ignore
        u = mda.Universe(str(state_pdb))
        resname_map = {int(res.resid): str(res.resname).strip() for res in u.residues}
    except Exception:
        return labels

    out: list[str] = []
    for key in labels:
        selection = str(residue_mapping.get(key) or key)
        resid_val: int | None = None
        match = re.search(r"(?:resid|resnum)\s+(-?\d+)", selection)
        if match:
            resid_val = int(match.group(1))
        else:
            tail = re.search(r"(\d+)$", key)
            if tail:
                resid_val = int(tail.group(1))
        if resid_val is not None and resid_val in resname_map:
            out.append(f"{resname_map[resid_val]}{resid_val}")
        elif resid_val is not None:
            out.append(f"res{resid_val}")
        else:
            out.append(key)
    return out


def atomic_pickle_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def pickle_load(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def orchestration_paths(analysis_dir: Path) -> dict[str, Path]:
    root = analysis_dir / "_orchestration"
    partials = root / "partials"
    errors = root / "errors"
    return {
        "root": root,
        "partials": partials,
        "errors": errors,
        "prepared": root / "prepared.pkl",
        "aggregate": root / "aggregate.pkl",
    }


def partial_result_path(analysis_dir: Path, row: int) -> Path:
    return orchestration_paths(analysis_dir)["partials"] / f"{int(row):06d}.pkl"


def partial_error_path(analysis_dir: Path, row: int) -> Path:
    return orchestration_paths(analysis_dir)["errors"] / f"{int(row):06d}.json"


def write_partial_result(analysis_dir: Path, row: int, payload: dict[str, Any]) -> Path:
    out = partial_result_path(analysis_dir, row)
    atomic_pickle_dump(out, payload)
    return out


def write_partial_error(analysis_dir: Path, row: int, error: str) -> Path:
    out = partial_error_path(analysis_dir, row)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"row": int(row), "error": str(error)}, indent=2), encoding="utf-8")
    return out


def load_partial_results(analysis_dir: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in range(int(expected_rows)):
        path = partial_result_path(analysis_dir, row)
        if not path.exists():
            raise FileNotFoundError(f"Missing partial result for row {row}: {path}")
        rows.append(pickle_load(path))
    return rows


def load_partial_errors(analysis_dir: Path) -> list[dict[str, Any]]:
    errors_dir = orchestration_paths(analysis_dir)["errors"]
    out: list[dict[str, Any]] = []
    if not errors_dir.exists():
        return out
    for path in sorted(errors_dir.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            out.append({"row": path.stem, "error": "Unreadable error payload."})
    return out


def cleanup_orchestration_dir(analysis_dir: Path) -> None:
    root = orchestration_paths(analysis_dir)["root"]
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        root.rmdir()
    except OSError:
        pass


def run_local_payload_batch(
    payloads: Sequence[dict[str, Any]],
    *,
    worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    max_workers: int = 1,
    progress_callback: ProgressCallback = None,
    progress_label: str = "Running batch",
) -> list[dict[str, Any]]:
    n_payloads = int(len(payloads))
    if n_payloads <= 0:
        return []
    workers = max(1, int(max_workers))
    out_rows: list[dict[str, Any] | None] = [None] * n_payloads
    if progress_callback:
        progress_callback(progress_label, 0, n_payloads)
    if workers <= 1:
        for row, payload in enumerate(payloads):
            out_rows[row] = worker_fn(payload)
            if progress_callback:
                progress_callback(progress_label, row + 1, n_payloads)
    else:
        workers = min(workers, n_payloads)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker_fn, payloads[row]): row for row in range(n_payloads)}
            done = 0
            for future in as_completed(futures):
                row = futures[future]
                out_rows[row] = future.result()
                done += 1
                if progress_callback:
                    progress_callback(progress_label, done, n_payloads)
    if any(v is None for v in out_rows):
        raise RuntimeError("Missing worker output while executing local payload batch.")
    return [row for row in out_rows if row is not None]


def _resolve_analysis_sample_npz_path(
    *,
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_dir: Path,
    sample_entry: dict[str, Any],
) -> Path | None:
    paths = sample_entry.get("paths") or {}
    rel = None
    if isinstance(paths, dict):
        rel = paths.get("summary_npz") or paths.get("path")
    rel = rel or sample_entry.get("path")
    if not rel:
        return None
    npz_path = Path(str(rel))
    if npz_path.is_absolute():
        return npz_path
    resolved = store.resolve_path(project_id, system_id, str(rel))
    if resolved.exists():
        return resolved
    alt = cluster_dir / str(rel)
    return alt if alt.exists() else resolved


def _load_potts_analysis_labels(
    npz_path: str | Path | None,
    *,
    md_mode: bool,
    md_label_mode: str,
    drop_invalid: bool,
) -> dict[str, Any]:
    if npz_path is None or not str(npz_path).strip():
        return {
            "labels": np.zeros((0, 0), dtype=int),
            "reason": "missing_path",
            "info": {"n_frames_total": 0, "n_frames_used": 0, "invalid_count": 0},
        }
    path = Path(str(npz_path)).resolve()
    mode = (md_label_mode or "assigned").strip().lower()
    cache_key = (str(path), bool(md_mode), mode, bool(drop_invalid))
    cached = _POTTS_ANALYSIS_SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        sample_npz = load_sample_npz(path)
    except Exception as exc:
        result = {
            "labels": np.zeros((0, 0), dtype=int),
            "reason": "load_failed",
            "info": {
                "n_frames_total": 0,
                "n_frames_used": 0,
                "invalid_count": 0,
                "error": str(exc),
            },
        }
        _POTTS_ANALYSIS_SAMPLE_CACHE[cache_key] = result
        return result
    if md_mode and mode in {"halo", "labels_halo"} and sample_npz.labels_halo is not None:
        labels = np.asarray(sample_npz.labels_halo, dtype=int)
    else:
        labels = np.asarray(sample_npz.labels, dtype=int)
    frame_indices = (
        np.asarray(sample_npz.frame_indices, dtype=np.int64)
        if sample_npz.frame_indices is not None and sample_npz.frame_indices.shape[0] == labels.shape[0]
        else np.arange(labels.shape[0], dtype=np.int64)
    )
    n_frames_total = int(labels.shape[0]) if labels.ndim == 2 else 0
    invalid_count = int(np.count_nonzero(sample_npz.invalid_mask)) if sample_npz.invalid_mask is not None else 0
    reason = None
    if bool(drop_invalid) and sample_npz.invalid_mask is not None:
        keep = ~np.asarray(sample_npz.invalid_mask, dtype=bool)
        if keep.shape[0] == labels.shape[0]:
            labels = labels[keep]
            frame_indices = frame_indices[keep]
            if labels.shape[0] == 0 and n_frames_total > 0:
                reason = "all_frames_invalid"
    if labels.size == 0 and reason is None:
        reason = "empty_labels"
    result = {
        "labels": np.asarray(labels, dtype=int),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "reason": reason,
        "info": {
            "n_frames_total": n_frames_total,
            "n_frames_used": int(labels.shape[0]) if labels.ndim == 2 else 0,
            "invalid_count": invalid_count,
        },
    }
    _POTTS_ANALYSIS_SAMPLE_CACHE[cache_key] = result
    return result


def _load_potts_analysis_model(model_path: str | Path) -> Any:
    key = str(Path(str(model_path)).resolve())
    cached = _POTTS_ANALYSIS_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = load_potts_model(key)
    _POTTS_ANALYSIS_MODEL_CACHE[key] = model
    return model


def _load_additive_potts_analysis_model(model_paths: Sequence[str | Path]) -> Any:
    paths = [str(Path(str(path)).resolve()) for path in model_paths if str(path or "").strip()]
    if not paths:
        raise ValueError("No Potts models were provided for energy evaluation.")
    key = "additive:" + "|".join(paths)
    cached = _POTTS_ANALYSIS_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = _load_potts_analysis_model(paths[0])
    for path in paths[1:]:
        model = add_potts_models(model, _load_potts_analysis_model(path))
    _POTTS_ANALYSIS_MODEL_CACHE[key] = model
    return model


def _skip_record_from_sample(sample_entry: dict[str, Any], *, stage: str, reason: str, info: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "sample_id": str(sample_entry.get("sample_id") or ""),
        "sample_name": sample_entry.get("name"),
        "sample_type": sample_entry.get("type"),
        "sample_method": sample_entry.get("method"),
        "stage": stage,
        "reason": str(reason or "empty_labels"),
    }
    payload.update(info or {})
    return payload


def _filter_potts_analysis_samples(
    samples: Sequence[dict[str, Any]],
    *,
    model_id: str | None = None,
    model_ids: Sequence[str] | None = None,
    requested_sample_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    requested = {str(value).strip() for value in (requested_sample_ids or []) if str(value).strip()}

    def model_ids_for_sample(sample: dict[str, Any]) -> set[str]:
        params = sample.get("params") if isinstance(sample.get("params"), dict) else {}
        raw = [sample.get("model_id"), params.get("model_id")]
        raw.extend(sample.get("model_ids") if isinstance(sample.get("model_ids"), list) else [])
        raw.extend(params.get("model_ids") if isinstance(params.get("model_ids"), list) else [])
        return {str(value).strip() for value in raw if str(value or "").strip()}

    if requested:
        return [sample for sample in samples if str(sample.get("sample_id") or "") in requested]
    selected_models = {str(value).strip() for value in (model_ids or []) if str(value).strip()}
    if len(selected_models) > 1:
        return [sample for sample in samples if model_ids_for_sample(sample) == selected_models]
    if selected_models:
        selected_model = next(iter(selected_models))
        return [sample for sample in samples if selected_model in model_ids_for_sample(sample)]
    if model_id:
        return [sample for sample in samples if str(model_id) in model_ids_for_sample(sample)]
    return list(samples)


def prepare_potts_analysis_batch(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_ref: str | None = None,
    model_refs: Sequence[str] | None = None,
    comparison_sample_ids: Sequence[str] | None = None,
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    n_workers: int | None = None,
    analysis_id: str | None = None,
    analysis_edge_mode: str | None = None,
    analysis_contact_cutoff: float | None = None,
    analysis_contact_atom_mode: str | None = None,
    analysis_contact_state_ids: Sequence[str] | None = None,
    analysis_contact_pdbs: Sequence[str] | None = None,
) -> dict[str, Any]:
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]
    cluster_path = cluster_dir / "cluster.npz"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster NPZ not found: {cluster_path}")

    ds = load_npz(str(cluster_path), unassigned_policy="drop_frames", allow_missing_edges=True)
    K = [int(v) for v in np.asarray(ds.cluster_counts, dtype=int).tolist()]
    cluster_edges = list(ds.edges or [])

    samples = store.list_samples(project_id, system_id, cluster_id)
    md_samples = [s for s in samples if str(s.get("type") or "").strip().lower() == "md_eval"]
    other_samples = [s for s in samples if str(s.get("type") or "").strip().lower() != "md_eval"]

    refs = list(dict.fromkeys(str(value).strip() for value in (model_refs or []) if str(value).strip()))
    if not refs and model_ref:
        refs = [str(model_ref).strip()]
    resolved_rows: list[tuple[Any, str | None, str, Path]] = []
    for ref in refs:
        resolved_model, resolved_id, resolved_name, resolved_path = _resolve_model_for_analysis(
            store=store,
            project_id=project_id,
            system_id=system_id,
            cluster_id=cluster_id,
            model_ref=ref,
        )
        path = Path(resolved_path).resolve()
        resolved_rows.append(
            (resolved_model, str(resolved_id) if resolved_id else None, str(resolved_name or resolved_id or path.stem), path)
        )
    resolved_rows.sort(key=lambda row: str(row[1] or row[3]))
    resolved_models = [row[0] for row in resolved_rows]
    resolved_model_ids = [str(row[1]) for row in resolved_rows if row[1]]
    resolved_model_names = [row[2] for row in resolved_rows]
    resolved_model_paths = [row[3] for row in resolved_rows]

    model = None
    if resolved_models:
        model = resolved_models[0]
        for extra_model in resolved_models[1:]:
            model = add_potts_models(model, extra_model)
    if len(resolved_model_paths) == 1:
        model_id = resolved_model_ids[0] if resolved_model_ids else None
        model_name = resolved_model_names[0]
    elif resolved_model_paths:
        if len(resolved_model_ids) == len(resolved_model_paths):
            model_id = "additive:" + "+".join(sorted(resolved_model_ids))
        else:
            digest = hashlib.sha256("|".join(sorted(map(str, resolved_model_paths))).encode("utf-8")).hexdigest()[:16]
            model_id = f"additive:{digest}"
        model_name = " + ".join(resolved_model_names)
    else:
        model_id = None
        model_name = None

    other_samples = _filter_potts_analysis_samples(
        other_samples,
        model_id=model_id,
        model_ids=resolved_model_ids,
        requested_sample_ids=comparison_sample_ids,
    )

    analysis_samples = [*md_samples, *other_samples]

    edge_mode = str(analysis_edge_mode or "").strip().lower()
    if edge_mode and edge_mode not in {"model", "cluster", "contact", "all_vs_all"}:
        raise ValueError("analysis_edge_mode must be one of: model, cluster, contact, all_vs_all.")
    if not edge_mode:
        edge_mode = "model" if model is not None else "cluster"
    contact_cutoff = float(analysis_contact_cutoff if analysis_contact_cutoff is not None else 10.0)
    if not np.isfinite(contact_cutoff) or contact_cutoff <= 0.0:
        raise ValueError("analysis_contact_cutoff must be > 0.")
    contact_atom_mode = str(analysis_contact_atom_mode or "CA").strip().upper()
    if contact_atom_mode not in {"CA", "CM"}:
        raise ValueError("analysis_contact_atom_mode must be one of: CA, CM.")

    if edge_mode == "model":
        edges_for_metrics = list((model.edges if model is not None else cluster_edges) or [])
    elif edge_mode == "cluster":
        edges_for_metrics = list(cluster_edges or [])
    elif edge_mode == "all_vs_all":
        n_res = int(len(K))
        edges_for_metrics = [(i, j) for i in range(n_res) for j in range(i + 1, n_res)]
    else:
        edges_for_metrics = _resolve_analysis_contact_edges(
            store=store,
            project_id=project_id,
            system_id=system_id,
            md_samples=md_samples,
            state_ids=analysis_contact_state_ids,
            pdbs=analysis_contact_pdbs,
            cluster_npz_path=cluster_path,
            cutoff=contact_cutoff,
            atom_mode=contact_atom_mode,
        )
        if not edges_for_metrics:
            raise ValueError("No contact edges found for analysis. Check states/PDBs and contact cutoff.")
    edges_for_metrics = sorted(
        {
            (min(int(r), int(s)), max(int(r), int(s)))
            for r, s in edges_for_metrics
            if int(r) != int(s)
        }
    )

    comparisons_root = _ensure_analysis_dir(cluster_dir, "md_vs_sample")
    energies_root = _ensure_analysis_dir(cluster_dir, "model_energy")
    _purge_matching_sampling_analyses(
        comparisons_root,
        model_id=model_id,
        model_name=model_name,
        md_label_mode=md_label_mode,
        drop_invalid=bool(drop_invalid),
    )
    _purge_matching_sampling_analyses(
        energies_root,
        model_id=model_id,
        model_name=model_name,
        md_label_mode=md_label_mode,
        drop_invalid=bool(drop_invalid),
    )

    payloads: list[dict[str, Any]] = []
    for md_entry in md_samples:
        md_npz_path = _resolve_analysis_sample_npz_path(
            store=store,
            project_id=project_id,
            system_id=system_id,
            cluster_dir=cluster_dir,
            sample_entry=md_entry,
        )
        md_payload = {
            "sample_id": str(md_entry.get("sample_id") or ""),
            "name": md_entry.get("name"),
            "type": md_entry.get("type"),
            "method": md_entry.get("method"),
            "npz_path": str(md_npz_path or ""),
        }
        for sample_entry in other_samples:
            sample_npz_path = _resolve_analysis_sample_npz_path(
                store=store,
                project_id=project_id,
                system_id=system_id,
                cluster_dir=cluster_dir,
                sample_entry=sample_entry,
            )
            payloads.append(
                {
                    "kind": "md_vs_sample",
                    "md_sample": md_payload,
                    "sample": {
                        "sample_id": str(sample_entry.get("sample_id") or ""),
                        "name": sample_entry.get("name"),
                        "type": sample_entry.get("type"),
                        "method": sample_entry.get("method"),
                        "npz_path": str(sample_npz_path or ""),
                    },
                    "K": list(K),
                    "edges": [tuple(map(int, edge)) for edge in edges_for_metrics],
                    "md_label_mode": str(md_label_mode or "assigned"),
                    "drop_invalid": bool(drop_invalid),
                }
            )

    if resolved_model_paths:
        for sample_entry in analysis_samples:
            sample_npz_path = _resolve_analysis_sample_npz_path(
                store=store,
                project_id=project_id,
                system_id=system_id,
                cluster_dir=cluster_dir,
                sample_entry=sample_entry,
            )
            payloads.append(
                {
                    "kind": "model_energy",
                    "sample": {
                        "sample_id": str(sample_entry.get("sample_id") or ""),
                        "name": sample_entry.get("name"),
                        "type": sample_entry.get("type"),
                        "method": sample_entry.get("method"),
                        "npz_path": str(sample_npz_path or ""),
                    },
                    "model_paths": [str(path) for path in resolved_model_paths],
                    "md_mode": str(sample_entry.get("type") or "").strip().lower() == "md_eval",
                    "md_label_mode": str(md_label_mode or "assigned"),
                    "drop_invalid": bool(drop_invalid),
                }
            )

    analysis_id = str(analysis_id or uuid.uuid4())
    batch_dir = cluster_dir / "_orchestration" / "potts_analysis" / analysis_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    requested_workers = int(n_workers) if n_workers is not None else 0
    prepared = {
        "analysis_id": analysis_id,
        "analysis_dir": str(batch_dir),
        "cluster_dir": str(cluster_dir),
        "system_dir": str(system_dir),
        "comparisons_root": str(comparisons_root),
        "energies_root": str(energies_root),
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "model_id": model_id,
        "model_ids": list(resolved_model_ids),
        "model_name": model_name,
        "model_names": list(resolved_model_names),
        "model_paths": [str(path) for path in resolved_model_paths],
        "md_label_mode": str(md_label_mode or "assigned"),
        "drop_invalid": bool(drop_invalid),
        "edges_for_metrics": [tuple(map(int, edge)) for edge in edges_for_metrics],
        "edge_selection": {
            "mode": edge_mode,
            "contact_cutoff": (float(contact_cutoff) if edge_mode == "contact" else None),
            "contact_atom_mode": (contact_atom_mode if edge_mode == "contact" else None),
            "contact_state_ids": [str(v) for v in (analysis_contact_state_ids or []) if str(v).strip()],
            "contact_pdbs": [str(v) for v in (analysis_contact_pdbs or []) if str(v).strip()],
        },
        "md_samples": len(md_samples),
        "other_samples": len(other_samples),
        "payloads": payloads,
        "requested_workers": requested_workers,
    }
    atomic_pickle_dump(orchestration_paths(batch_dir)["prepared"], prepared)
    return prepared


def _potts_analysis_payload_worker(payload: dict[str, Any]) -> dict[str, Any]:
    kind = str(payload.get("kind") or "").strip().lower()
    md_label_mode = str(payload.get("md_label_mode") or "assigned")
    drop_invalid = bool(payload.get("drop_invalid", True))
    if kind == "md_vs_sample":
        md_sample = payload.get("md_sample") or {}
        sample = payload.get("sample") or {}
        md_loaded = _load_potts_analysis_labels(
            md_sample.get("npz_path"),
            md_mode=True,
            md_label_mode=md_label_mode,
            drop_invalid=drop_invalid,
        )
        X_md = np.asarray(md_loaded["labels"], dtype=int)
        if X_md.size == 0:
            return {
                "kind": "md_vs_sample",
                "skip": _skip_record_from_sample(
                    md_sample,
                    stage="md_vs_sample",
                    reason=str(md_loaded.get("reason") or "empty_labels"),
                    info=dict(md_loaded.get("info") or {}),
                ),
            }
        sample_loaded = _load_potts_analysis_labels(
            sample.get("npz_path"),
            md_mode=False,
            md_label_mode=md_label_mode,
            drop_invalid=drop_invalid,
        )
        X_sample = np.asarray(sample_loaded["labels"], dtype=int)
        if X_sample.size == 0:
            return {
                "kind": "md_vs_sample",
                "skip": _skip_record_from_sample(
                    sample,
                    stage="md_vs_sample",
                    reason=str(sample_loaded.get("reason") or "empty_labels"),
                    info=dict(sample_loaded.get("info") or {}),
                ),
            }
        metrics = compute_md_vs_sample_metrics(
            X_md,
            X_sample,
            K=[int(v) for v in (payload.get("K") or [])],
            edges=[tuple(map(int, edge)) for edge in (payload.get("edges") or [])],
        )
        return {
            "kind": "md_vs_sample",
            "md_sample": md_sample,
            "sample": sample,
            "node_js": np.asarray(metrics["node_js"], dtype=float),
            "edge_js": np.asarray(metrics["edge_js"], dtype=float),
            "summary": {
                "node_js_mean": float(metrics["node_js_mean"]),
                "node_js_median": float(metrics["node_js_median"]),
                "node_js_max": float(metrics["node_js_max"]),
                "edge_js_mean": float(metrics["edge_js_mean"]),
                "edge_js_median": float(metrics["edge_js_median"]),
                "edge_js_max": float(metrics["edge_js_max"]),
                "combined_distance": float(metrics["combined_distance"]),
                "md_count": int(X_md.shape[0]),
                "sample_count": int(X_sample.shape[0]),
            },
        }
    if kind == "model_energy":
        sample = payload.get("sample") or {}
        loaded = _load_potts_analysis_labels(
            sample.get("npz_path"),
            md_mode=bool(payload.get("md_mode", False)),
            md_label_mode=md_label_mode,
            drop_invalid=drop_invalid,
        )
        X = np.asarray(loaded["labels"], dtype=int)
        if X.size == 0:
            return {
                "kind": "model_energy",
                "skip": _skip_record_from_sample(
                    sample,
                    stage="model_energy",
                    reason=str(loaded.get("reason") or "empty_labels"),
                    info=dict(loaded.get("info") or {}),
                ),
            }
        model = _load_additive_potts_analysis_model(payload.get("model_paths") or [])
        energy_payload = compute_sample_energies(model, X)
        return {
            "kind": "model_energy",
            "sample": sample,
            "energies": np.asarray(energy_payload["energies"], dtype=float),
            "frame_indices": np.asarray(loaded.get("frame_indices"), dtype=np.int64),
            "summary": {
                "energy_mean": float(energy_payload["energy_mean"]),
                "energy_median": float(energy_payload["energy_median"]),
                "energy_min": float(energy_payload["energy_min"]),
                "energy_max": float(energy_payload["energy_max"]),
                "count": int(X.shape[0]),
            },
        }
    raise ValueError(f"Unsupported Potts analysis payload kind: {kind}")


def aggregate_potts_analysis_batch(
    prepared: dict[str, Any],
    out_rows: Sequence[dict[str, Any]],
    *,
    workers_used: int,
) -> dict[str, Any]:
    cluster_dir = Path(str(prepared["cluster_dir"]))
    system_dir = Path(str(prepared["system_dir"]))
    comparisons_root = Path(str(prepared["comparisons_root"]))
    energies_root = Path(str(prepared["energies_root"]))
    model_id = prepared.get("model_id")
    model_ids = [str(value) for value in (prepared.get("model_ids") or [])]
    model_name = prepared.get("model_name")
    model_names = [str(value) for value in (prepared.get("model_names") or [])]
    md_label_mode = str(prepared.get("md_label_mode") or "assigned")
    drop_invalid = bool(prepared.get("drop_invalid", True))
    edges_for_metrics = [tuple(map(int, edge)) for edge in (prepared.get("edges_for_metrics") or [])]
    edge_selection = dict(prepared.get("edge_selection") or {})

    written_comparisons: list[dict[str, Any]] = []
    written_energies: list[dict[str, Any]] = []
    skipped_samples: list[dict[str, Any]] = []
    skipped_seen: set[tuple[str, str, str]] = set()

    for row in out_rows:
        skip_entry = row.get("skip") if isinstance(row, dict) else None
        if skip_entry:
            sample_id = str(skip_entry.get("sample_id") or "")
            seen_key = (sample_id, str(skip_entry.get("stage") or ""), str(skip_entry.get("reason") or ""))
            if seen_key not in skipped_seen:
                skipped_seen.add(seen_key)
                skipped_samples.append(skip_entry)
            continue
        kind = str(row.get("kind") or "").strip().lower()
        if kind == "md_vs_sample":
            analysis_id = str(uuid.uuid4())
            analysis_dir = comparisons_root / analysis_id
            analysis_dir.mkdir(parents=True, exist_ok=True)
            npz_path = analysis_dir / "analysis.npz"
            np.savez_compressed(
                npz_path,
                node_js=np.asarray(row.get("node_js"), dtype=float),
                edge_js=np.asarray(row.get("edge_js"), dtype=float),
                edges=np.asarray(edges_for_metrics, dtype=int),
            )
            md_sample = row.get("md_sample") or {}
            sample = row.get("sample") or {}
            meta = {
                "analysis_id": analysis_id,
                "analysis_type": "md_vs_sample",
                "created_at": _utc_now(),
                "project_id": prepared["project_id"],
                "system_id": prepared["system_id"],
                "cluster_id": prepared["cluster_id"],
                "md_sample_id": md_sample.get("sample_id"),
                "md_sample_name": md_sample.get("name"),
                "sample_id": sample.get("sample_id"),
                "sample_name": sample.get("name"),
                "sample_type": sample.get("type"),
                "sample_method": sample.get("method"),
                "model_id": model_id,
                "model_ids": model_ids,
                "model_name": model_name,
                "model_names": model_names,
                "drop_invalid": bool(drop_invalid),
                "md_label_mode": md_label_mode,
                "paths": {
                    "analysis_npz": _relativize(npz_path, system_dir),
                },
                "summary": row.get("summary") or {},
                "edge_selection": edge_selection,
            }
            (analysis_dir / ANALYSIS_METADATA_FILENAME).write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
            written_comparisons.append(meta)
        elif kind == "model_energy":
            analysis_id = str(uuid.uuid4())
            analysis_dir = energies_root / analysis_id
            analysis_dir.mkdir(parents=True, exist_ok=True)
            npz_path = analysis_dir / "analysis.npz"
            np.savez_compressed(
                npz_path,
                energies=np.asarray(row.get("energies"), dtype=float),
                frame_indices=np.asarray(row.get("frame_indices"), dtype=np.int64),
            )
            sample = row.get("sample") or {}
            meta = {
                "analysis_id": analysis_id,
                "analysis_type": "model_energy",
                "created_at": _utc_now(),
                "project_id": prepared["project_id"],
                "system_id": prepared["system_id"],
                "cluster_id": prepared["cluster_id"],
                "model_id": model_id,
                "model_ids": model_ids,
                "model_name": model_name,
                "model_names": model_names,
                "sample_id": sample.get("sample_id"),
                "sample_name": sample.get("name"),
                "sample_type": sample.get("type"),
                "sample_method": sample.get("method"),
                "drop_invalid": bool(drop_invalid),
                "md_label_mode": md_label_mode,
                "paths": {
                    "analysis_npz": _relativize(npz_path, system_dir),
                },
                "summary": row.get("summary") or {},
                "edge_selection": edge_selection,
            }
            (analysis_dir / ANALYSIS_METADATA_FILENAME).write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
            written_energies.append(meta)

    summary = {
        "cluster_id": prepared["cluster_id"],
        "md_samples": int(prepared.get("md_samples") or 0),
        "other_samples": int(prepared.get("other_samples") or 0),
        "comparisons_written": len(written_comparisons),
        "energies_written": len(written_energies),
        "model_id": model_id,
        "model_ids": model_ids,
        "model_name": model_name,
        "model_names": model_names,
        "edge_selection": edge_selection,
        "skipped_samples": skipped_samples,
        "workers_used": int(max(1, workers_used)),
    }
    result = {
        "analysis_id": str(prepared["analysis_id"]),
        "analysis_dir": str(Path(str(prepared["analysis_dir"])).resolve()),
        "analyses_root": str((cluster_dir / "analyses").resolve()),
        "metadata": summary,
    }
    atomic_pickle_dump(orchestration_paths(Path(str(prepared["analysis_dir"])))["aggregate"], result)
    return result


def run_potts_analysis_local(
    *,
    progress_callback: ProgressCallback = None,
    **kwargs: Any,
) -> dict[str, Any]:
    prepared = prepare_potts_analysis_batch(**kwargs)
    payloads = prepared.get("payloads") or []
    requested = int(kwargs.get("n_workers") or prepared.get("requested_workers") or 0)
    if requested > 0:
        workers = requested
    else:
        workers = max(1, min(int(os.cpu_count() or 1), max(1, len(payloads))))
    out_rows = run_local_payload_batch(
        payloads,
        worker_fn=_potts_analysis_payload_worker,
        max_workers=workers,
        progress_callback=progress_callback,
        progress_label="Running Sampling Explorer analysis",
    )
    workers_used = 1 if not payloads else max(1, min(workers, len(payloads)))
    return aggregate_potts_analysis_batch(prepared, out_rows, workers_used=workers_used)


def prepare_gibbs_relaxation_batch(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    start_sample_id: str,
    model_ref: str,
    beta: float = 1.0,
    n_start_frames: int = 100,
    gibbs_sweeps: int = 1000,
    seed: int = 0,
    start_label_mode: str = "assigned",
    drop_invalid: bool = True,
    n_workers: int | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    mode = (start_label_mode or "assigned").strip().lower()
    if mode not in {"assigned", "halo"}:
        raise ValueError("start_label_mode must be 'assigned' or 'halo'.")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be > 0.")
    n_start_frames = int(n_start_frames)
    gibbs_sweeps = int(gibbs_sweeps)
    seed = int(seed)
    if n_start_frames < 1:
        raise ValueError("n_start_frames must be >= 1.")
    if gibbs_sweeps < 1:
        raise ValueError("gibbs_sweeps must be >= 1.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]

    model_id = None
    model_name = None
    model_path = Path(str(model_ref))
    if not model_path.suffix:
        model_id = str(model_ref)
        models = store.list_potts_models(project_id, system_id, cluster_id)
        entry = next((m for m in models if m.get("model_id") == model_id), None)
        if not entry or not entry.get("path"):
            raise FileNotFoundError(f"Potts model_id not found on this cluster: {model_id}")
        model_name = str(entry.get("name") or model_id)
        model_path = store.resolve_path(project_id, system_id, str(entry.get("path")))
    else:
        if not model_path.is_absolute():
            model_path = store.resolve_path(project_id, system_id, str(model_path))
        model_name = model_path.stem
    if not model_path.exists():
        raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")

    model = load_potts_model(str(model_path))
    N = int(len(model.h))
    if N <= 0:
        raise ValueError("Model has no residues.")
    K_list = [int(k) for k in model.K_list()]

    samples = store.list_samples(project_id, system_id, cluster_id)
    sample_entry = next((s for s in samples if str(s.get("sample_id")) == str(start_sample_id)), None)
    if not sample_entry:
        raise FileNotFoundError(f"Sample not found on this cluster: {start_sample_id}")

    def _resolve_sample_path(entry: dict[str, Any]) -> Path:
        paths = entry.get("paths") or {}
        rel = None
        if isinstance(paths, dict):
            rel = paths.get("summary_npz") or paths.get("path")
        rel = rel or entry.get("path")
        if not rel:
            raise FileNotFoundError("Sample entry missing path.")
        p = Path(str(rel))
        if not p.is_absolute():
            resolved = store.resolve_path(project_id, system_id, str(rel))
            if not resolved.exists():
                alt = cluster_dir / str(rel)
                p = alt if alt.exists() else resolved
            else:
                p = resolved
        return p

    sample_npz = load_sample_npz(_resolve_sample_path(sample_entry))
    X = sample_npz.labels
    if mode in {"halo", "labels_halo"} and sample_npz.labels_halo is not None:
        X = sample_npz.labels_halo
    X = np.asarray(X, dtype=np.int32)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("Starting sample contains no labels.")
    if int(X.shape[1]) != N:
        raise ValueError(f"Starting sample has N={X.shape[1]}, model expects N={N}.")

    frame_indices = (
        np.asarray(sample_npz.frame_indices, dtype=np.int64)
        if sample_npz.frame_indices is not None and sample_npz.frame_indices.shape[0] == X.shape[0]
        else np.arange(X.shape[0], dtype=np.int64)
    )
    frame_state_ids = (
        np.asarray(sample_npz.frame_state_ids, dtype=str)
        if sample_npz.frame_state_ids is not None and sample_npz.frame_state_ids.shape[0] == X.shape[0]
        else np.full((X.shape[0],), "", dtype=str)
    )

    if drop_invalid and sample_npz.invalid_mask is not None:
        keep = ~np.asarray(sample_npz.invalid_mask, dtype=bool).ravel()
        if keep.shape[0] == X.shape[0]:
            X = X[keep]
            frame_indices = frame_indices[keep]
            frame_state_ids = frame_state_ids[keep]

    valid = np.all(X >= 0, axis=1)
    for i, k in enumerate(K_list):
        valid &= X[:, i] < int(k)
    if not np.any(valid):
        raise ValueError("No valid starting frames after filtering (invalid/out-of-range labels).")
    X = X[valid]
    frame_indices = frame_indices[valid]
    frame_state_ids = frame_state_ids[valid]

    n_select = min(n_start_frames, int(X.shape[0]))
    rng = np.random.default_rng(seed)
    selected_local = np.asarray(rng.choice(X.shape[0], size=n_select, replace=False), dtype=np.int64)
    selected_starts = np.asarray(X[selected_local], dtype=np.int32)
    selected_frame_indices = np.asarray(frame_indices[selected_local], dtype=np.int64)
    selected_frame_state_ids = np.asarray(frame_state_ids[selected_local], dtype=str)

    residue_keys: list[str] = []
    cluster_npz_path = cluster_dir / "cluster.npz"
    if cluster_npz_path.exists():
        try:
            with np.load(cluster_npz_path, allow_pickle=True) as cnpz:
                if "metadata_json" in cnpz:
                    meta = json.loads(cnpz["metadata_json"].item())
                    if isinstance(meta, dict):
                        residue_keys = [str(v) for v in (meta.get("residue_keys") or [])]
        except Exception:
            residue_keys = []
    if len(residue_keys) != N:
        residue_keys = [f"res_{i}" for i in range(N)]

    requested_workers = os.cpu_count() or 1 if n_workers is None or int(n_workers) <= 0 else int(n_workers)
    requested_workers = max(1, min(int(requested_workers), n_select))

    payloads = [
        {
            "model_path": str(model_path),
            "x0": selected_starts[row],
            "n_sweeps": int(gibbs_sweeps),
            "beta": float(beta),
            "seed": int(seed) + row,
        }
        for row in range(n_select)
    ]

    analysis_id = str(analysis_id or uuid.uuid4())
    analysis_root = _ensure_analysis_dir(cluster_dir, "gibbs_relaxation")
    analysis_dir = analysis_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    npz_path = analysis_dir / "analysis.npz"
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME

    prepared = {
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "system_dir": str(system_dir),
        "cluster_dir": str(cluster_dir),
        "analysis_id": analysis_id,
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(npz_path),
        "analysis_metadata": str(meta_path),
        "payloads": payloads,
        "residue_keys": list(residue_keys),
        "start_sample_id": str(start_sample_id),
        "start_sample_name": sample_entry.get("name"),
        "start_sample_type": sample_entry.get("type"),
        "model_id": model_id,
        "model_name": model_name,
        "model_path": str(model_path),
        "start_label_mode": mode,
        "drop_invalid": bool(drop_invalid),
        "beta": float(beta),
        "n_start_frames_requested": int(n_start_frames),
        "n_start_frames_used": int(n_select),
        "gibbs_sweeps": int(gibbs_sweeps),
        "seed": int(seed),
        "requested_workers": int(requested_workers),
        "selected_frame_indices": np.asarray(selected_frame_indices, dtype=np.int64),
        "selected_frame_state_ids": np.asarray(selected_frame_state_ids, dtype=str),
        "n_residues": int(N),
    }
    atomic_pickle_dump(orchestration_paths(analysis_dir)["prepared"], prepared)
    return prepared


def aggregate_gibbs_relaxation_batch(
    prepared: dict[str, Any],
    out_rows: Sequence[dict[str, Any]],
    *,
    workers_used: int,
) -> dict[str, Any]:
    if any(v is None for v in out_rows):
        raise RuntimeError("Missing worker output while aggregating Gibbs relaxation analysis.")

    n_select = int(prepared["n_start_frames_used"])
    gibbs_sweeps = int(prepared["gibbs_sweeps"])
    N = int(prepared["n_residues"])

    first_flip = np.zeros((n_select, N), dtype=np.int32)
    flip_counts_sum = np.zeros((gibbs_sweeps, N), dtype=np.uint32)
    energy_traces = np.zeros((n_select, gibbs_sweeps), dtype=np.float32)
    for row, out in enumerate(out_rows):
        first_flip[row] = np.asarray(out["first_flip"], dtype=np.int32)
        flip_counts_sum += np.asarray(out["flip_counts"], dtype=np.uint32)
        energy_traces[row] = np.asarray(out["energy_trace"], dtype=np.float32)

    mean_first = np.mean(first_flip, axis=0).astype(np.float32)
    median_first = np.median(first_flip, axis=0).astype(np.float32)
    q25_first = np.quantile(first_flip, 0.25, axis=0).astype(np.float32)
    q75_first = np.quantile(first_flip, 0.75, axis=0).astype(np.float32)

    order = np.argsort(mean_first, kind="mergesort")
    pct_fast = np.zeros((N,), dtype=np.float32)
    if N == 1:
        pct_fast[0] = 1.0
    else:
        pct_fast[order] = 1.0 - (np.arange(N, dtype=np.float32) / np.float32(N - 1))
    pct_slow = (1.0 - pct_fast).astype(np.float32)

    flip_prob_time = (flip_counts_sum.astype(np.float32) / float(n_select)).astype(np.float32)
    mean_flip_fraction_by_step = np.mean(flip_prob_time, axis=1).astype(np.float32)
    ever_flip_rate = np.mean(first_flip <= int(gibbs_sweeps), axis=0).astype(np.float32)
    early_cutoff = max(1, int(round(0.25 * float(gibbs_sweeps))))
    early_flip_rate = np.mean(first_flip <= int(early_cutoff), axis=0).astype(np.float32)

    energy_mean = np.mean(energy_traces, axis=0).astype(np.float32)
    energy_std = np.std(energy_traces, axis=0).astype(np.float32)

    top_k = min(20, N)
    top_fast_idx = np.argsort(mean_first)[:top_k].astype(np.int32)
    top_slow_idx = np.argsort(mean_first)[::-1][:top_k].astype(np.int32)

    analysis_dir = Path(str(prepared["analysis_dir"]))
    system_dir = Path(str(prepared["system_dir"]))
    npz_path = Path(str(prepared["analysis_npz"]))
    meta_path = Path(str(prepared["analysis_metadata"]))
    residue_keys = [str(v) for v in prepared["residue_keys"]]

    np.savez_compressed(
        npz_path,
        residue_keys=np.asarray(residue_keys, dtype=str),
        start_frame_indices=np.asarray(prepared["selected_frame_indices"], dtype=np.int64),
        start_frame_state_ids=np.asarray(prepared["selected_frame_state_ids"], dtype=str),
        first_flip_steps=np.asarray(first_flip, dtype=np.int32),
        mean_first_flip_steps=np.asarray(mean_first, dtype=np.float32),
        median_first_flip_steps=np.asarray(median_first, dtype=np.float32),
        q25_first_flip_steps=np.asarray(q25_first, dtype=np.float32),
        q75_first_flip_steps=np.asarray(q75_first, dtype=np.float32),
        flip_percentile_fast=np.asarray(pct_fast, dtype=np.float32),
        flip_percentile_slow=np.asarray(pct_slow, dtype=np.float32),
        ever_flip_rate=np.asarray(ever_flip_rate, dtype=np.float32),
        early_flip_rate=np.asarray(early_flip_rate, dtype=np.float32),
        flip_prob_time=np.asarray(flip_prob_time, dtype=np.float32),
        mean_flip_fraction_by_step=np.asarray(mean_flip_fraction_by_step, dtype=np.float32),
        energy_traces=np.asarray(energy_traces, dtype=np.float32),
        energy_mean=np.asarray(energy_mean, dtype=np.float32),
        energy_std=np.asarray(energy_std, dtype=np.float32),
        top_fast_indices=np.asarray(top_fast_idx, dtype=np.int32),
        top_slow_indices=np.asarray(top_slow_idx, dtype=np.int32),
        beta=np.asarray([float(prepared["beta"])], dtype=np.float32),
        gibbs_sweeps=np.asarray([gibbs_sweeps], dtype=np.int32),
        n_start_frames=np.asarray([n_select], dtype=np.int32),
    )

    now = _utc_now()
    meta = {
        "analysis_id": str(prepared["analysis_id"]),
        "analysis_type": "gibbs_relaxation",
        "created_at": now,
        "updated_at": now,
        "project_id": str(prepared["project_id"]),
        "system_id": str(prepared["system_id"]),
        "cluster_id": str(prepared["cluster_id"]),
        "start_sample_id": str(prepared["start_sample_id"]),
        "start_sample_name": prepared["start_sample_name"],
        "start_sample_type": prepared["start_sample_type"],
        "model_id": prepared["model_id"],
        "model_name": prepared["model_name"],
        "model_path": _relativize(Path(str(prepared["model_path"])), system_dir),
        "start_label_mode": str(prepared["start_label_mode"]),
        "drop_invalid": bool(prepared["drop_invalid"]),
        "beta": float(prepared["beta"]),
        "n_start_frames_requested": int(prepared["n_start_frames_requested"]),
        "n_start_frames_used": int(prepared["n_start_frames_used"]),
        "gibbs_sweeps": int(gibbs_sweeps),
        "seed": int(prepared["seed"]),
        "workers": int(max(1, workers_used)),
        "paths": {"analysis_npz": _relativize(npz_path, system_dir)},
        "summary": {
            "n_residues": int(N),
            "mean_first_flip_min": float(np.min(mean_first)) if mean_first.size else None,
            "mean_first_flip_median": float(np.median(mean_first)) if mean_first.size else None,
            "mean_first_flip_max": float(np.max(mean_first)) if mean_first.size else None,
            "early_cutoff_step": int(early_cutoff),
        },
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
    result = {"metadata": _convert_nan_to_none(meta), "analysis_npz": str(npz_path), "analysis_dir": str(analysis_dir)}
    atomic_pickle_dump(orchestration_paths(analysis_dir)["aggregate"], result)
    return result


def run_gibbs_relaxation_local(
    *,
    progress_callback: ProgressCallback = None,
    **kwargs: Any,
) -> dict[str, Any]:
    prepared = prepare_gibbs_relaxation_batch(**kwargs)
    payloads = prepared["payloads"]
    workers = max(1, int(prepared.get("requested_workers", 1)))
    out_rows = run_local_payload_batch(
        payloads,
        worker_fn=_gibbs_relax_worker,
        max_workers=workers,
        progress_callback=progress_callback,
        progress_label="Running Gibbs relaxations",
    )
    workers_used = 1 if not payloads else max(1, min(workers, len(payloads)))
    return aggregate_gibbs_relaxation_batch(prepared, out_rows, workers_used=workers_used)


def prepare_ligand_completion_batch(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_a_ref: str,
    model_b_ref: str,
    md_sample_id: str,
    constrained_residues: Sequence[str | int],
    reference_sample_id_a: str | None = None,
    reference_sample_id_b: str | None = None,
    sampler: str = "sa",
    lambda_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    n_start_frames: int = 100,
    n_samples_per_frame: int = 100,
    n_steps: int = 1000,
    tail_steps: int = 200,
    target_window_size: int = 11,
    target_pseudocount: float = 1e-3,
    epsilon_logpenalty: float = 1e-8,
    constraint_weight_mode: str = "uniform",
    constraint_weights: Sequence[float] | None = None,
    constraint_weight_min: float = 0.0,
    constraint_weight_max: float = 1.0,
    constraint_source_mode: str = "manual",
    constraint_delta_js_analysis_id: str | None = None,
    constraint_delta_js_sample_id: str | None = None,
    constraint_auto_top_k: int = 12,
    constraint_auto_edge_alpha: float = 0.3,
    constraint_auto_exclude_success: bool = True,
    gibbs_beta: float = 1.0,
    sa_beta_hot: float = 0.8,
    sa_beta_cold: float = 50.0,
    sa_schedule: str = "geom",
    md_label_mode: str = "assigned",
    drop_invalid: bool = True,
    success_metric_mode: str = "deltae",
    delta_js_experiment_id: str | None = None,
    delta_js_analysis_id: str | None = None,
    delta_js_filter_setup_id: str | None = None,
    delta_js_filter_edge_alpha: float = 0.75,
    delta_js_d_residue_min: float = 0.0,
    delta_js_d_residue_max: float | None = None,
    delta_js_d_edge_min: float = 0.0,
    delta_js_d_edge_max: float | None = None,
    delta_js_node_edge_alpha: float | None = None,
    js_success_threshold: float = 0.15,
    js_success_margin: float = 0.02,
    deltae_margin: float = 0.0,
    completion_target_success: float = 0.7,
    completion_cost_if_unreached: float | None = None,
    n_workers: int = 1,
    seed: int = 0,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    sampler = str(sampler or "sa").strip().lower()
    if sampler not in {"sa", "gibbs"}:
        raise ValueError("sampler must be 'sa' or 'gibbs'.")
    mode = str(md_label_mode or "assigned").strip().lower()
    if mode not in {"assigned", "halo"}:
        raise ValueError("md_label_mode must be 'assigned' or 'halo'.")
    success_mode = str(success_metric_mode or "deltae").strip().lower()
    if success_mode not in {"deltae", "delta_js_edge"}:
        raise ValueError("success_metric_mode must be one of: deltae, delta_js_edge.")
    shared_delta_js_id = str(delta_js_experiment_id or "").strip()
    if success_mode == "delta_js_edge" and not (str(delta_js_analysis_id or "").strip() or shared_delta_js_id):
        raise ValueError("delta_js_analysis_id (or delta_js_experiment_id) is required when success_metric_mode='delta_js_edge'.")
    if float(js_success_threshold) < 0:
        raise ValueError("js_success_threshold must be >= 0.")
    constraint_mode = str(constraint_source_mode or "manual").strip().lower()
    if constraint_mode not in {"manual", "delta_js_auto"}:
        raise ValueError("constraint_source_mode must be one of: manual, delta_js_auto.")
    if constraint_mode == "delta_js_auto" and not (
        str(constraint_delta_js_analysis_id or "").strip() or shared_delta_js_id
    ):
        raise ValueError("constraint_delta_js_analysis_id (or delta_js_experiment_id) is required when constraint_source_mode='delta_js_auto'.")
    if int(constraint_auto_top_k) < 1:
        raise ValueError("constraint_auto_top_k must be >= 1.")
    if not np.isfinite(float(delta_js_filter_edge_alpha)) or not (0.0 <= float(delta_js_filter_edge_alpha) <= 1.0):
        raise ValueError("delta_js_filter_edge_alpha must be in [0,1].")

    n_start_frames = int(n_start_frames)
    n_samples_per_frame = int(n_samples_per_frame)
    n_steps = int(n_steps)
    tail_steps = int(tail_steps)
    target_window_size = int(target_window_size)
    n_workers = int(n_workers)
    seed = int(seed)
    if n_start_frames < 1 or n_samples_per_frame < 1 or n_steps < 1 or tail_steps < 1 or target_window_size < 1:
        raise ValueError("n_start_frames, n_samples_per_frame, n_steps, tail_steps and target_window_size must be >= 1.")

    lambdas = np.asarray([float(v) for v in lambda_values], dtype=float).ravel()
    if lambdas.size < 2:
        raise ValueError("Provide at least 2 lambda values.")
    if not np.all(np.isfinite(lambdas)):
        raise ValueError("lambda_values must be finite.")
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    if np.unique(lambdas).size != lambdas.size:
        raise ValueError("lambda_values must be unique.")
    if float(np.min(lambdas)) < 0.0:
        raise ValueError("lambda_values must be >= 0.")
    if float(target_pseudocount) < 0:
        raise ValueError("target_pseudocount must be >= 0.")
    if float(epsilon_logpenalty) <= 0:
        raise ValueError("epsilon_logpenalty must be > 0.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]
    resolved_success_delta_js_id = str(delta_js_analysis_id or "").strip() or shared_delta_js_id
    resolved_constraint_delta_js_id = str(constraint_delta_js_analysis_id or "").strip() or shared_delta_js_id

    delta_js_filter_rules: list[dict[str, float]] = []
    delta_js_filter_setup_id_resolved = str(delta_js_filter_setup_id or "").strip()
    if delta_js_filter_setup_id_resolved:
        setup_path = cluster_dir / "ui_setups" / f"{delta_js_filter_setup_id_resolved}.json"
        if not setup_path.exists():
            raise FileNotFoundError(f"Delta-JS filter setup not found: {delta_js_filter_setup_id_resolved}")
        try:
            setup_obj = json.loads(setup_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid Delta-JS filter setup JSON: {delta_js_filter_setup_id_resolved}") from exc
        payload_obj = setup_obj.get("payload") if isinstance(setup_obj, dict) else None
        rules_raw = payload_obj.get("rules") if isinstance(payload_obj, dict) else None
        delta_js_filter_rules = _normalize_js_filter_rules(rules_raw)
        if not delta_js_filter_rules:
            raise ValueError(
                f"Delta-JS filter setup '{delta_js_filter_setup_id_resolved}' has no valid rules payload."
            )

    def _resolve_model(ref: str) -> tuple[Any, dict[str, Any], Path, str | None, str]:
        model_id: str | None = None
        model_path = Path(str(ref))
        model_meta: dict[str, Any] | None = None
        model_name = ""
        if not model_path.suffix:
            model_id = str(ref).strip()
            models = store.list_potts_models(project_id, system_id, cluster_id)
            model_meta = next((m for m in models if str(m.get("model_id")) == model_id), None)
            if not model_meta or not model_meta.get("path"):
                raise FileNotFoundError(f"Potts model_id not found on this cluster: {model_id}")
            model_name = str(model_meta.get("name") or model_id)
            model_path = store.resolve_path(project_id, system_id, str(model_meta.get("path")))
        else:
            if not model_path.is_absolute():
                model_path = store.resolve_path(project_id, system_id, str(model_path))
            model_name = model_path.stem
            model_meta = {"model_id": None, "name": model_name, "path": str(model_path)}
        if not model_path.exists():
            raise FileNotFoundError(f"Potts model NPZ not found: {model_path}")
        model = zero_sum_gauge_model(load_potts_model(str(model_path)))
        return model, dict(model_meta or {}), model_path.resolve(), model_id, model_name

    def _resolve_sample_path(entry: dict[str, Any]) -> Path:
        paths = entry.get("paths") or {}
        rel = None
        if isinstance(paths, dict):
            rel = paths.get("summary_npz") or paths.get("path")
        rel = rel or entry.get("path")
        if not rel:
            raise FileNotFoundError("Sample entry missing path.")
        p = Path(str(rel))
        if not p.is_absolute():
            resolved = store.resolve_path(project_id, system_id, str(rel))
            if not resolved.exists():
                alt = cluster_dir / str(rel)
                p = alt if alt.exists() else resolved
            else:
                p = resolved
        return p

    def _load_labels(entry: dict[str, Any], *, md_mode: bool) -> tuple[np.ndarray, np.ndarray]:
        p = _resolve_sample_path(entry)
        s = load_sample_npz(p)
        X = s.labels
        if md_mode and mode in {"halo", "labels_halo"} and s.labels_halo is not None:
            X = s.labels_halo
        X = np.asarray(X, dtype=np.int32)
        if X.ndim != 2 or X.size == 0:
            raise ValueError(f"Sample has empty labels: {entry.get('sample_id')}")
        frame_idx = (
            np.asarray(s.frame_indices, dtype=np.int64)
            if s.frame_indices is not None and s.frame_indices.shape[0] == X.shape[0]
            else np.arange(X.shape[0], dtype=np.int64)
        )
        if drop_invalid and s.invalid_mask is not None:
            keep = ~np.asarray(s.invalid_mask, dtype=bool).ravel()
            if keep.shape[0] == X.shape[0]:
                X = X[keep]
                frame_idx = frame_idx[keep]
        return X, frame_idx

    model_a, model_a_meta, model_a_path, model_a_id, model_a_name = _resolve_model(model_a_ref)
    model_b, model_b_meta, model_b_path, model_b_id, model_b_name = _resolve_model(model_b_ref)
    if len(model_a.h) != len(model_b.h):
        raise ValueError("Model sizes do not match.")
    K_list = [int(k) for k in model_a.K_list()]
    if K_list != [int(k) for k in model_b.K_list()]:
        raise ValueError("Model state cardinalities (K) do not match.")
    N = int(len(K_list))

    samples = store.list_samples(project_id, system_id, cluster_id)
    sample_by_id = {str(s.get("sample_id")): s for s in samples if isinstance(s, dict) and s.get("sample_id")}
    start_entry = sample_by_id.get(str(md_sample_id))
    if not start_entry:
        raise FileNotFoundError(f"Start MD sample not found: {md_sample_id}")
    if str(start_entry.get("type") or "") != "md_eval":
        raise ValueError(f"Start sample must be an md_eval sample: {md_sample_id}")
    X_start, frame_idx_start = _load_labels(start_entry, md_mode=True)
    valid = np.all(X_start >= 0, axis=1)
    for i, Ki in enumerate(K_list):
        valid &= X_start[:, i] < int(Ki)
    if not np.any(valid):
        raise ValueError("No valid MD frames after filtering invalid/out-of-range labels.")
    X_start = X_start[valid]
    frame_idx_start = frame_idx_start[valid]

    def _auto_ref_sample_id(model_meta: dict[str, Any], fallback_label: str) -> str:
        params = model_meta.get("params") or {}
        candidates: list[str] = []
        raw_state_ids = params.get("state_ids")
        if isinstance(raw_state_ids, list):
            candidates.extend([str(v) for v in raw_state_ids if str(v).strip()])
        for key in ("active_state_id", "inactive_state_id"):
            val = params.get(key)
            if val is not None and str(val).strip():
                candidates.append(str(val))
        md_entries = [s for s in samples if str(s.get("type") or "") == "md_eval"]
        for sid in candidates:
            hit = next((s for s in md_entries if str(s.get("state_id") or "") == sid), None)
            if hit:
                return str(hit.get("sample_id"))
        raise ValueError(
            f"Could not infer reference MD sample for model '{fallback_label}'. "
            "Provide reference_sample_id_a/reference_sample_id_b explicitly."
        )

    ref_id_a = str(reference_sample_id_a or "").strip()
    ref_id_b = str(reference_sample_id_b or "").strip()
    if not ref_id_a:
        ref_id_a = _auto_ref_sample_id(model_a_meta, str(model_a_name or model_a_id or "A"))
    if not ref_id_b:
        ref_id_b = _auto_ref_sample_id(model_b_meta, str(model_b_name or model_b_id or "B"))
    ref_entry_a = sample_by_id.get(ref_id_a)
    ref_entry_b = sample_by_id.get(ref_id_b)
    if not ref_entry_a or not ref_entry_b:
        raise FileNotFoundError("Reference sample ids not found on this cluster.")
    if str(ref_entry_a.get("type") or "") != "md_eval":
        raise ValueError(f"Reference sample A must be md_eval: {ref_id_a}")
    if str(ref_entry_b.get("type") or "") != "md_eval":
        raise ValueError(f"Reference sample B must be md_eval: {ref_id_b}")
    X_ref_a, _ = _load_labels(ref_entry_a, md_mode=True)
    X_ref_b, _ = _load_labels(ref_entry_b, md_mode=True)
    for Xr, name in ((X_ref_a, "A"), (X_ref_b, "B")):
        mask = np.all(Xr >= 0, axis=1)
        for i, Ki in enumerate(K_list):
            mask &= Xr[:, i] < int(Ki)
        if not np.any(mask):
            raise ValueError(f"Reference sample {name} has no valid frames.")
        if name == "A":
            X_ref_a = Xr[mask]
        else:
            X_ref_b = Xr[mask]

    p_ref_a = marginals(X_ref_a, K_list)
    p_ref_b = marginals(X_ref_b, K_list)
    max_k = int(max(K_list)) if K_list else 0
    p_ref_a_pad = np.zeros((N, max_k), dtype=np.float32)
    p_ref_b_pad = np.zeros((N, max_k), dtype=np.float32)
    for i, Ki in enumerate(K_list):
        p_ref_a_pad[i, :Ki] = np.asarray(p_ref_a[i], dtype=np.float32)
        p_ref_b_pad[i, :Ki] = np.asarray(p_ref_b[i], dtype=np.float32)

    residue_keys: list[str] = []
    cluster_npz_path = cluster_dir / "cluster.npz"
    if cluster_npz_path.exists():
        try:
            with np.load(cluster_npz_path, allow_pickle=True) as cnpz:
                if "residue_keys" in cnpz:
                    residue_keys = [str(v) for v in np.asarray(cnpz["residue_keys"]).tolist()]
        except Exception:
            residue_keys = []
    if len(residue_keys) != N:
        residue_keys = [f"res_{i}" for i in range(N)]
    key_to_index = {str(k): i for i, k in enumerate(residue_keys)}
    resid_to_index: dict[int, int] = {}
    for i, key in enumerate(residue_keys):
        match = re.search(r"(?:res[_-]?)(\d+)$", str(key), flags=re.IGNORECASE)
        if match:
            resid_to_index[int(match.group(1))] = i

    delta_js_eval_spec: dict[str, Any] = {}
    delta_js_selected_residue_indices = np.zeros((0,), dtype=np.int32)
    delta_js_selected_edge_indices = np.zeros((0,), dtype=np.int32)
    delta_js_selected_residue_weights = np.zeros((0,), dtype=np.float32)
    delta_js_selected_edge_weights = np.zeros((0,), dtype=np.float32)
    delta_js_selected_edges = np.zeros((0, 2), dtype=np.int32)
    delta_js_effective_alpha = float(np.nan)
    delta_js_filter_success_set_a = np.zeros((0,), dtype=np.int32)
    delta_js_filter_success_set_b = np.zeros((0,), dtype=np.int32)
    delta_js_filter_success_union = np.zeros((0,), dtype=np.int32)
    delta_js_filter_target_candidates = np.zeros((0,), dtype=np.int32)
    if success_mode == "delta_js_edge":
        raw_ref = str(resolved_success_delta_js_id or "").strip()
        analysis_npz_path: Path | None = None
        sample_ids_djs: list[str] = []
        js_node_a_all = np.zeros((0, 0), dtype=float)
        js_node_b_all = np.zeros((0, 0), dtype=float)
        js_edge_a_all = np.zeros((0, 0), dtype=float)
        js_edge_b_all = np.zeros((0, 0), dtype=float)
        ref_path = Path(raw_ref)
        if ref_path.suffix:
            if not ref_path.is_absolute():
                ref_path = store.resolve_path(project_id, system_id, str(ref_path))
            analysis_npz_path = ref_path
        else:
            candidate = cluster_dir / "analyses" / "delta_js" / raw_ref / "analysis.npz"
            if candidate.exists():
                analysis_npz_path = candidate
        if analysis_npz_path is None or not analysis_npz_path.exists():
            raise FileNotFoundError(f"Delta-JS analysis NPZ not found: {raw_ref}")

        with np.load(analysis_npz_path, allow_pickle=False) as djs:
            if "D_residue" not in djs:
                raise ValueError("Invalid delta_js analysis: missing D_residue.")
            D_residue = np.asarray(djs["D_residue"], dtype=float).ravel()
            if D_residue.shape[0] != N:
                raise ValueError(f"Delta-JS residue size mismatch: expected {N}, got {D_residue.shape[0]}.")
            edges_all = np.asarray(djs.get("edges", np.zeros((0, 2), dtype=int)), dtype=np.int32).reshape(-1, 2)
            D_edge = np.asarray(djs.get("D_edge", np.zeros((edges_all.shape[0],), dtype=float)), dtype=float).ravel()
            if D_edge.shape[0] != edges_all.shape[0]:
                raise ValueError("Invalid delta_js analysis: D_edge size mismatch.")
            top_res_idx = np.asarray(djs.get("top_residue_indices", np.arange(N, dtype=int)), dtype=np.int32).ravel()
            if top_res_idx.size == 0:
                top_res_idx = np.arange(N, dtype=np.int32)
            top_res_idx = top_res_idx[(top_res_idx >= 0) & (top_res_idx < N)]
            if top_res_idx.size == 0:
                raise ValueError("Delta-JS top_residue_indices are invalid/empty.")
            top_edge_idx = np.asarray(djs.get("top_edge_indices", np.arange(edges_all.shape[0], dtype=int)), dtype=np.int32).ravel()
            if edges_all.shape[0] == 0:
                top_edge_idx = np.zeros((0,), dtype=np.int32)
            elif top_edge_idx.size == 0:
                top_edge_idx = np.arange(edges_all.shape[0], dtype=np.int32)
            top_edge_idx = top_edge_idx[(top_edge_idx >= 0) & (top_edge_idx < edges_all.shape[0])]
            alpha_from_analysis = float(np.asarray(djs.get("node_edge_alpha", np.asarray([0.5], dtype=float))).ravel()[0])
            sample_ids_djs = [str(v) for v in np.asarray(djs.get("sample_ids", np.asarray([], dtype=str)), dtype=str).tolist()]
            js_node_a_all = np.asarray(djs.get("js_node_a", np.zeros((0, 0), dtype=float)), dtype=float)
            js_node_b_all = np.asarray(djs.get("js_node_b", np.zeros((0, 0), dtype=float)), dtype=float)
            js_edge_a_all = np.asarray(djs.get("js_edge_a", np.zeros((0, 0), dtype=float)), dtype=float)
            js_edge_b_all = np.asarray(djs.get("js_edge_b", np.zeros((0, 0), dtype=float)), dtype=float)

        rmin = float(delta_js_d_residue_min)
        rmax = float(delta_js_d_residue_max) if delta_js_d_residue_max is not None else float("inf")
        if rmax < rmin:
            rmin, rmax = rmax, rmin
        emn = float(delta_js_d_edge_min)
        emx = float(delta_js_d_edge_max) if delta_js_d_edge_max is not None else float("inf")
        if emx < emn:
            emn, emx = emx, emn
        keep_res: list[int] = []
        if delta_js_filter_rules:
            if not sample_ids_djs:
                raise ValueError("Delta-JS analysis has no sample_ids for filter-driven selection.")
            try:
                row_a = sample_ids_djs.index(str(ref_id_a))
                row_b = sample_ids_djs.index(str(ref_id_b))
            except Exception as exc:
                raise ValueError(
                    "Filter setup requires reference sample rows in selected Delta-JS analysis "
                    f"('{ref_id_a}', '{ref_id_b}')."
                ) from exc
            filt_a1, filt_a2 = _delta_js_row_node_edge_values(
                row=row_a,
                N=N,
                js_node_a=js_node_a_all,
                js_node_b=js_node_b_all,
                js_edge_a=js_edge_a_all,
                js_edge_b=js_edge_b_all,
                edges_all=edges_all,
                top_edge_indices=top_edge_idx,
                D_edge=D_edge,
                edge_alpha=float(delta_js_filter_edge_alpha),
            )
            filt_b1, filt_b2 = _delta_js_row_node_edge_values(
                row=row_b,
                N=N,
                js_node_a=js_node_a_all,
                js_node_b=js_node_b_all,
                js_edge_a=js_edge_a_all,
                js_edge_b=js_edge_b_all,
                edges_all=edges_all,
                top_edge_indices=top_edge_idx,
                D_edge=D_edge,
                edge_alpha=float(delta_js_filter_edge_alpha),
            )
            set_a = [i for i in range(N) if _passes_js_filter_rules(float(filt_a1[i]), float(filt_a2[i]), delta_js_filter_rules)]
            set_b = [i for i in range(N) if _passes_js_filter_rules(float(filt_b1[i]), float(filt_b2[i]), delta_js_filter_rules)]
            keep_res = sorted(set([int(i) for i in set_a] + [int(i) for i in set_b]))
            delta_js_filter_success_set_a = np.asarray(sorted(set(set_a)), dtype=np.int32)
            delta_js_filter_success_set_b = np.asarray(sorted(set(set_b)), dtype=np.int32)
            delta_js_filter_success_union = np.asarray(keep_res, dtype=np.int32)
            if not keep_res:
                raise ValueError("Delta-JS filter setup selected zero residues on both reference MD samples.")
        else:
            for idx in top_res_idx.tolist():
                d = float(D_residue[int(idx)])
                if np.isfinite(d) and d >= rmin and d <= rmax:
                    keep_res.append(int(idx))
            if not keep_res:
                raise ValueError(
                    "Delta-JS residue filter selected zero residues. Relax delta_js_d_residue_min/max or choose a different delta_js_analysis_id."
                )
        delta_js_selected_residue_indices = np.asarray(sorted(set(keep_res)), dtype=np.int32)
        delta_js_selected_residue_weights = np.asarray(
            [float(max(0.0, D_residue[int(i)])) for i in delta_js_selected_residue_indices.tolist()],
            dtype=np.float32,
        )
        if not np.any(delta_js_selected_residue_weights > 0):
            delta_js_selected_residue_weights = np.ones_like(delta_js_selected_residue_weights, dtype=np.float32)
        selected_res_set = {int(i) for i in delta_js_selected_residue_indices.tolist()}
        keep_edges: list[int] = []
        for eidx in top_edge_idx.tolist():
            r, s = int(edges_all[int(eidx), 0]), int(edges_all[int(eidx), 1])
            if r not in selected_res_set or s not in selected_res_set:
                continue
            if delta_js_filter_rules:
                keep_edges.append(int(eidx))
            else:
                d = float(D_edge[int(eidx)])
                if np.isfinite(d) and d >= emn and d <= emx:
                    keep_edges.append(int(eidx))
        if keep_edges:
            delta_js_selected_edge_indices = np.asarray(keep_edges, dtype=np.int32)
            delta_js_selected_edges = np.asarray(edges_all[delta_js_selected_edge_indices], dtype=np.int32)
            delta_js_selected_edge_weights = np.asarray(
                [float(max(0.0, D_edge[int(i)])) for i in delta_js_selected_edge_indices.tolist()],
                dtype=np.float32,
            )
            if not np.any(delta_js_selected_edge_weights > 0):
                delta_js_selected_edge_weights = np.ones_like(delta_js_selected_edge_weights, dtype=np.float32)
        edge_list = [(int(r), int(s)) for (r, s) in delta_js_selected_edges.tolist()]
        if edge_list:
            p_ref_edge_a_dict = pairwise_joints_on_edges(X_ref_a, K_list, edge_list)
            p_ref_edge_b_dict = pairwise_joints_on_edges(X_ref_b, K_list, edge_list)
            ref_edge_a = [np.asarray(p_ref_edge_a_dict[e], dtype=np.float32) for e in edge_list]
            ref_edge_b = [np.asarray(p_ref_edge_b_dict[e], dtype=np.float32) for e in edge_list]
        else:
            ref_edge_a = []
            ref_edge_b = []
        alpha = float(delta_js_node_edge_alpha) if delta_js_node_edge_alpha is not None else float(alpha_from_analysis)
        if not np.isfinite(alpha):
            alpha = 0.5
        alpha = max(0.0, min(1.0, alpha))
        if not edge_list:
            alpha = 0.0
        delta_js_effective_alpha = float(alpha)
        delta_js_eval_spec = {
            "residue_indices": np.asarray(delta_js_selected_residue_indices, dtype=np.int32),
            "residue_weights": np.asarray(delta_js_selected_residue_weights, dtype=np.float32),
            "edges": np.asarray(delta_js_selected_edges, dtype=np.int32),
            "edge_weights": np.asarray(delta_js_selected_edge_weights, dtype=np.float32),
            "ref_edge_a": ref_edge_a,
            "ref_edge_b": ref_edge_b,
            "node_edge_alpha": float(delta_js_effective_alpha),
        }

    parsed_constraints: list[int] = []
    constraint_auto_impact = np.zeros((N,), dtype=np.float32)
    constraint_auto_ranked_indices: list[int] = []
    constraint_auto_row_sample_id: str | None = None
    if constraint_mode == "manual":
        for raw in constrained_residues or []:
            idx: int | None = None
            if isinstance(raw, (int, np.integer)):
                value = int(raw)
                if value in resid_to_index:
                    idx = resid_to_index[value]
                elif 0 <= value < N:
                    idx = value
            else:
                token = str(raw).strip()
                if not token:
                    continue
                if token in key_to_index:
                    idx = key_to_index[token]
                else:
                    match = re.search(r"(\d+)", token)
                    if match:
                        value = int(match.group(1))
                        if value in resid_to_index:
                            idx = resid_to_index[value]
                        elif 0 <= value < N:
                            idx = value
            if idx is None:
                raise ValueError(f"Unable to resolve constrained residue: {raw!r}")
            if idx not in parsed_constraints:
                parsed_constraints.append(int(idx))
        if not parsed_constraints:
            raise ValueError("No constrained residues resolved.")
    else:
        raw_ref = str(resolved_constraint_delta_js_id or "").strip()
        c_djs_npz_path: Path | None = None
        ref_path = Path(raw_ref)
        if ref_path.suffix:
            if not ref_path.is_absolute():
                ref_path = store.resolve_path(project_id, system_id, str(ref_path))
            c_djs_npz_path = ref_path
        else:
            candidate = cluster_dir / "analyses" / "delta_js" / raw_ref / "analysis.npz"
            if candidate.exists():
                c_djs_npz_path = candidate
        if c_djs_npz_path is None or not c_djs_npz_path.exists():
            raise FileNotFoundError(f"Constraint delta-js analysis NPZ not found: {raw_ref}")
        target_sample_id = str(constraint_delta_js_sample_id or md_sample_id).strip()
        if not target_sample_id:
            raise ValueError("constraint_delta_js_sample_id resolution failed.")
        constraint_auto_row_sample_id = target_sample_id
        with np.load(c_djs_npz_path, allow_pickle=False) as djs:
            sample_ids = [str(v) for v in np.asarray(djs.get("sample_ids", np.asarray([], dtype=str)), dtype=str).tolist()]
            if not sample_ids:
                raise ValueError("Constraint delta-js analysis has no sample_ids.")
            try:
                row = sample_ids.index(target_sample_id)
            except Exception:
                raise ValueError(f"Sample '{target_sample_id}' not found in constraint delta-js analysis '{raw_ref}'.")
            D_residue = np.asarray(djs.get("D_residue", np.zeros((0,), dtype=float)), dtype=float).ravel()
            if D_residue.shape[0] != N:
                raise ValueError(f"Constraint delta-js residue size mismatch: expected {N}, got {D_residue.shape[0]}.")
            js_node_a = np.asarray(djs.get("js_node_a", np.zeros((0, 0), dtype=float)), dtype=float)
            js_node_b = np.asarray(djs.get("js_node_b", np.zeros((0, 0), dtype=float)), dtype=float)
            if js_node_a.ndim != 2 or js_node_b.ndim != 2 or js_node_a.shape != js_node_b.shape:
                raise ValueError("Constraint delta-js analysis has invalid js_node arrays.")
            if row >= js_node_a.shape[0] or js_node_a.shape[1] != N:
                raise ValueError("Constraint delta-js js_node dimensions mismatch.")
            node_term = np.asarray(D_residue, dtype=float) * np.abs(js_node_a[row] - js_node_b[row])
            node_term = np.where(np.isfinite(node_term), node_term, 0.0)
            edge_alpha = float(constraint_auto_edge_alpha)
            if not np.isfinite(edge_alpha):
                edge_alpha = 0.3
            edge_alpha = max(0.0, min(1.0, edge_alpha))
            edges_all = np.asarray(djs.get("edges", np.zeros((0, 2), dtype=int)), dtype=np.int32).reshape(-1, 2)
            D_edge = np.asarray(djs.get("D_edge", np.zeros((edges_all.shape[0],), dtype=float)), dtype=float).ravel()
            top_edge_indices = np.asarray(djs.get("top_edge_indices", np.arange(edges_all.shape[0], dtype=int)), dtype=np.int32).ravel()
            js_edge_a = np.asarray(djs.get("js_edge_a", np.zeros((0, 0), dtype=float)), dtype=float)
            js_edge_b = np.asarray(djs.get("js_edge_b", np.zeros((0, 0), dtype=float)), dtype=float)
            edge_term = np.zeros((N,), dtype=float)
            edge_counts = np.zeros((N,), dtype=float)
            if (
                edge_alpha > 0.0
                and edges_all.shape[0] > 0
                and top_edge_indices.size > 0
                and js_edge_a.ndim == 2
                and js_edge_b.ndim == 2
                and js_edge_a.shape == js_edge_b.shape
                and row < js_edge_a.shape[0]
            ):
                n_cols = min(js_edge_a.shape[1], top_edge_indices.shape[0])
                for col in range(n_cols):
                    eidx = int(top_edge_indices[col])
                    if eidx < 0 or eidx >= edges_all.shape[0] or eidx >= D_edge.shape[0]:
                        continue
                    r, s = int(edges_all[eidx, 0]), int(edges_all[eidx, 1])
                    if r < 0 or s < 0 or r >= N or s >= N:
                        continue
                    d = float(D_edge[eidx])
                    if not np.isfinite(d):
                        continue
                    value = float(abs(js_edge_a[row, col] - js_edge_b[row, col]))
                    if not np.isfinite(value):
                        continue
                    contrib = max(0.0, d) * value
                    edge_term[r] += contrib
                    edge_term[s] += contrib
                    edge_counts[r] += 1.0
                    edge_counts[s] += 1.0
            edge_term = edge_term / np.where(edge_counts > 0, edge_counts, 1.0)
            impact = (1.0 - edge_alpha) * node_term + edge_alpha * edge_term
            impact = np.where(np.isfinite(impact), impact, 0.0)
            impact = np.maximum(impact, 0.0)
            constraint_auto_impact = np.asarray(impact, dtype=np.float32)
            if delta_js_filter_rules:
                filt_a1, filt_a2 = _delta_js_row_node_edge_values(
                    row=row,
                    N=N,
                    js_node_a=js_node_a,
                    js_node_b=js_node_b,
                    js_edge_a=js_edge_a,
                    js_edge_b=js_edge_b,
                    edges_all=edges_all,
                    top_edge_indices=top_edge_indices,
                    D_edge=D_edge,
                    edge_alpha=float(delta_js_filter_edge_alpha),
                )
                target_candidates = [
                    int(i) for i in range(N)
                    if _passes_js_filter_rules(float(filt_a1[i]), float(filt_a2[i]), delta_js_filter_rules)
                ]
                delta_js_filter_target_candidates = np.asarray(sorted(set(target_candidates)), dtype=np.int32)
                if delta_js_filter_success_union.size == 0:
                    try:
                        row_ref_a = sample_ids.index(str(ref_id_a))
                        row_ref_b = sample_ids.index(str(ref_id_b))
                    except Exception as exc:
                        raise ValueError(
                            "Filter setup requires reference sample rows in selected Delta-JS analysis "
                            f"('{ref_id_a}', '{ref_id_b}')."
                        ) from exc
                    ref_a1, ref_a2 = _delta_js_row_node_edge_values(
                        row=row_ref_a,
                        N=N,
                        js_node_a=js_node_a,
                        js_node_b=js_node_b,
                        js_edge_a=js_edge_a,
                        js_edge_b=js_edge_b,
                        edges_all=edges_all,
                        top_edge_indices=top_edge_indices,
                        D_edge=D_edge,
                        edge_alpha=float(delta_js_filter_edge_alpha),
                    )
                    ref_b1, ref_b2 = _delta_js_row_node_edge_values(
                        row=row_ref_b,
                        N=N,
                        js_node_a=js_node_a,
                        js_node_b=js_node_b,
                        js_edge_a=js_edge_a,
                        js_edge_b=js_edge_b,
                        edges_all=edges_all,
                        top_edge_indices=top_edge_indices,
                        D_edge=D_edge,
                        edge_alpha=float(delta_js_filter_edge_alpha),
                    )
                    set_a = [int(i) for i in range(N) if _passes_js_filter_rules(float(ref_a1[i]), float(ref_a2[i]), delta_js_filter_rules)]
                    set_b = [int(i) for i in range(N) if _passes_js_filter_rules(float(ref_b1[i]), float(ref_b2[i]), delta_js_filter_rules)]
                    delta_js_filter_success_set_a = np.asarray(sorted(set(set_a)), dtype=np.int32)
                    delta_js_filter_success_set_b = np.asarray(sorted(set(set_b)), dtype=np.int32)
                    delta_js_filter_success_union = np.asarray(sorted(set(set_a + set_b)), dtype=np.int32)
        exclude_set: set[int] = set()
        if bool(constraint_auto_exclude_success):
            exclude_set = {int(i) for i in delta_js_selected_residue_indices.tolist()}
        if delta_js_filter_success_union.size:
            exclude_set |= {int(i) for i in delta_js_filter_success_union.tolist()}
        ranked = np.argsort(constraint_auto_impact)[::-1].tolist()
        if exclude_set:
            ranked = [int(i) for i in ranked if int(i) not in exclude_set]
        if delta_js_filter_rules:
            allowed_local = {int(i) for i in delta_js_filter_target_candidates.tolist()} - {
                int(i) for i in delta_js_filter_success_union.tolist()
            }
            ranked = [int(i) for i in ranked if int(i) in allowed_local]
        ranked = [int(i) for i in ranked if float(constraint_auto_impact[int(i)]) > 0.0]
        if not ranked:
            raise ValueError(
                "No residues available for constraint auto-selection after exclusions/filters. Adjust settings or disable exclude_success."
            )
        top_k = min(int(constraint_auto_top_k), len(ranked))
        parsed_constraints = [int(i) for i in ranked[:top_k]]
        constraint_auto_ranked_indices = [int(i) for i in ranked]

    C = parsed_constraints
    cw_mode = str(constraint_weight_mode or "uniform").strip().lower()
    cmin = float(constraint_weight_min)
    cmax = float(constraint_weight_max)
    if not np.isfinite(cmin):
        cmin = 0.0
    if not np.isfinite(cmax):
        cmax = 1.0
    if cmax < cmin:
        cmin, cmax = cmax, cmin
    if constraint_weights is not None:
        arr = np.asarray([float(v) for v in constraint_weights], dtype=float).ravel()
        if arr.size != len(C):
            raise ValueError("constraint_weights must have the same length as constrained_residues.")
        c_weights = np.clip(arr, cmin, cmax)
        cw_mode = "custom"
    elif constraint_mode == "delta_js_auto":
        arr = np.asarray([float(constraint_auto_impact[int(i)]) for i in C], dtype=float)
        if arr.size and np.nanmax(arr) > np.nanmin(arr):
            arr = (arr - np.nanmin(arr)) / max(1e-12, float(np.nanmax(arr) - np.nanmin(arr)))
        else:
            arr = np.ones((len(C),), dtype=float)
        c_weights = np.clip(arr, cmin, cmax)
        cw_mode = "delta_js_auto"
    elif cw_mode == "js_abs":
        p_start = marginals(X_start, K_list)
        vals = []
        for i in C:
            js_a_i = float(js_divergence(np.asarray(p_start[i], dtype=float), np.asarray(p_ref_a[i], dtype=float)))
            js_b_i = float(js_divergence(np.asarray(p_start[i], dtype=float), np.asarray(p_ref_b[i], dtype=float)))
            vals.append(abs(js_b_i - js_a_i))
        arr = np.asarray(vals, dtype=float)
        if arr.size and np.nanmax(arr) > np.nanmin(arr):
            arr = (arr - np.nanmin(arr)) / max(1e-12, float(np.nanmax(arr) - np.nanmin(arr)))
        else:
            arr = np.ones((len(C),), dtype=float)
        c_weights = np.clip(arr, cmin, cmax)
    else:
        c_weights = np.clip(np.ones((len(C),), dtype=float), cmin, cmax)
        cw_mode = "uniform"

    n_select = min(int(n_start_frames), int(X_start.shape[0]))
    rng = np.random.default_rng(seed)
    selected_local = np.asarray(rng.choice(X_start.shape[0], size=n_select, replace=False), dtype=np.int64)
    X_sel = np.asarray(X_start[selected_local], dtype=np.int32)
    frame_idx_sel = np.asarray(frame_idx_start[selected_local], dtype=np.int64)
    half_w = int(max(0, target_window_size // 2))
    penalty_by_frame: list[list[np.ndarray]] = []
    for loc in selected_local.tolist():
        lo = max(0, int(loc) - half_w)
        hi = min(int(X_start.shape[0]), int(loc) + half_w + 1)
        window = np.asarray(X_start[lo:hi], dtype=np.int32)
        frame_phi: list[np.ndarray] = []
        for i in C:
            Ki = int(K_list[i])
            vals = window[:, i]
            vals = vals[(vals >= 0) & (vals < Ki)]
            counts = np.ones((Ki,), dtype=float) if vals.size == 0 else np.bincount(vals, minlength=Ki).astype(float)
            counts = counts + float(target_pseudocount)
            probs = counts / max(1e-12, float(np.sum(counts)))
            phi = -np.log(np.clip(probs, float(epsilon_logpenalty), None))
            frame_phi.append(np.asarray(phi, dtype=np.float32))
        penalty_by_frame.append(frame_phi)

    if completion_cost_if_unreached is None:
        completion_cost_if_unreached = float(np.max(lambdas) + 1.0)

    payloads: list[dict[str, Any]] = []
    for row in range(n_select):
        payloads.append(
            {
                "model_a_path": str(model_a_path),
                "model_b_path": str(model_b_path),
                "x0": np.asarray(X_sel[row], dtype=np.int32),
                "K_list": K_list,
                "lambda_values": np.asarray(lambdas, dtype=float),
                "constrained_indices": C,
                "constrained_weights": np.asarray(c_weights, dtype=float),
                "penalty_phi": penalty_by_frame[row],
                "sampler": sampler,
                "n_steps": int(n_steps),
                "tail_steps": int(tail_steps),
                "n_samples_per_frame": int(n_samples_per_frame),
                "gibbs_beta": float(gibbs_beta),
                "sa_beta_hot": float(sa_beta_hot),
                "sa_beta_cold": float(sa_beta_cold),
                "sa_schedule": str(sa_schedule),
                "success_metric_mode": str(success_mode),
                "js_success_threshold": float(js_success_threshold),
                "js_success_margin": float(js_success_margin),
                "delta_js_eval_spec": delta_js_eval_spec,
                "deltae_margin": float(deltae_margin),
                "p_ref_a": np.asarray(p_ref_a_pad, dtype=np.float32),
                "p_ref_b": np.asarray(p_ref_b_pad, dtype=np.float32),
                "seed": int(seed) + row * 131,
            }
        )

    analysis_id = str(analysis_id or uuid.uuid4())
    analysis_root = _ensure_analysis_dir(cluster_dir, "ligand_completion")
    analysis_dir = analysis_root / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    npz_path = analysis_dir / "analysis.npz"
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME

    prepared = {
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "system_dir": str(system_dir),
        "cluster_dir": str(cluster_dir),
        "analysis_id": analysis_id,
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(npz_path),
        "analysis_metadata": str(meta_path),
        "payloads": payloads,
        "frame_idx_sel": np.asarray(frame_idx_sel, dtype=np.int64),
        "residue_keys": list(residue_keys),
        "constrained_indices": [int(v) for v in C],
        "constraint_weights": np.asarray(c_weights, dtype=np.float32),
        "constraint_source_mode": str(constraint_mode),
        "constraint_auto_row_sample_id": str(constraint_auto_row_sample_id) if constraint_auto_row_sample_id else None,
        "constraint_auto_impact": np.asarray(constraint_auto_impact, dtype=np.float32),
        "constraint_auto_ranked_indices": [int(v) for v in constraint_auto_ranked_indices],
        "model_a_id": model_a_id,
        "model_a_name": model_a_name,
        "model_a_path": str(model_a_path),
        "model_b_id": model_b_id,
        "model_b_name": model_b_name,
        "model_b_path": str(model_b_path),
        "md_sample_id": str(md_sample_id),
        "md_sample_name": start_entry.get("name"),
        "reference_sample_id_a": str(ref_id_a),
        "reference_sample_name_a": ref_entry_a.get("name"),
        "reference_sample_id_b": str(ref_id_b),
        "reference_sample_name_b": ref_entry_b.get("name"),
        "sampler": sampler,
        "lambda_values": np.asarray(lambdas, dtype=np.float32),
        "n_start_frames_requested": int(n_start_frames),
        "n_start_frames_used": int(n_select),
        "n_samples_per_frame": int(n_samples_per_frame),
        "n_steps": int(n_steps),
        "tail_steps": int(tail_steps),
        "target_window_size": int(target_window_size),
        "target_pseudocount": float(target_pseudocount),
        "epsilon_logpenalty": float(epsilon_logpenalty),
        "constraint_delta_js_analysis_id": str(resolved_constraint_delta_js_id) if resolved_constraint_delta_js_id else None,
        "constraint_auto_top_k": int(constraint_auto_top_k),
        "constraint_auto_edge_alpha": float(constraint_auto_edge_alpha),
        "constraint_auto_exclude_success": bool(constraint_auto_exclude_success),
        "constraint_weight_mode": cw_mode,
        "constraint_weight_min": float(cmin),
        "constraint_weight_max": float(cmax),
        "gibbs_beta": float(gibbs_beta),
        "sa_beta_hot": float(sa_beta_hot),
        "sa_beta_cold": float(sa_beta_cold),
        "sa_schedule": str(sa_schedule),
        "md_label_mode": mode,
        "drop_invalid": bool(drop_invalid),
        "success_metric_mode": str(success_mode),
        "js_success_threshold": float(js_success_threshold),
        "js_success_margin": float(js_success_margin),
        "delta_js_experiment_id": str(shared_delta_js_id) if shared_delta_js_id else None,
        "delta_js_analysis_id": str(resolved_success_delta_js_id) if resolved_success_delta_js_id else None,
        "delta_js_filter_setup_id": str(delta_js_filter_setup_id_resolved) if delta_js_filter_setup_id_resolved else None,
        "delta_js_filter_edge_alpha": float(delta_js_filter_edge_alpha),
        "delta_js_d_residue_min": float(delta_js_d_residue_min),
        "delta_js_d_residue_max": float(delta_js_d_residue_max) if delta_js_d_residue_max is not None else None,
        "delta_js_d_edge_min": float(delta_js_d_edge_min),
        "delta_js_d_edge_max": float(delta_js_d_edge_max) if delta_js_d_edge_max is not None else None,
        "delta_js_node_edge_alpha": float(delta_js_effective_alpha) if np.isfinite(delta_js_effective_alpha) else None,
        "delta_js_selected_residue_indices": np.asarray(delta_js_selected_residue_indices, dtype=np.int32),
        "delta_js_selected_residue_weights": np.asarray(delta_js_selected_residue_weights, dtype=np.float32),
        "delta_js_selected_edge_indices": np.asarray(delta_js_selected_edge_indices, dtype=np.int32),
        "delta_js_selected_edges": np.asarray(delta_js_selected_edges, dtype=np.int32),
        "delta_js_selected_edge_weights": np.asarray(delta_js_selected_edge_weights, dtype=np.float32),
        "delta_js_filter_success_set_a": np.asarray(delta_js_filter_success_set_a, dtype=np.int32),
        "delta_js_filter_success_set_b": np.asarray(delta_js_filter_success_set_b, dtype=np.int32),
        "delta_js_filter_success_union": np.asarray(delta_js_filter_success_union, dtype=np.int32),
        "delta_js_filter_target_candidates": np.asarray(delta_js_filter_target_candidates, dtype=np.int32),
        "deltae_margin": float(deltae_margin),
        "completion_target_success": float(completion_target_success),
        "completion_cost_if_unreached": float(completion_cost_if_unreached),
        "seed": int(seed),
        "requested_workers": int(max(1, n_workers)),
    }
    atomic_pickle_dump(orchestration_paths(analysis_dir)["prepared"], prepared)
    return prepared


def aggregate_ligand_completion_batch(
    prepared: dict[str, Any],
    out_rows: Sequence[dict[str, Any]],
    *,
    workers_used: int,
) -> dict[str, Any]:
    if any(v is None for v in out_rows):
        raise RuntimeError("Missing worker output while aggregating ligand completion analysis.")

    lambdas = np.asarray(prepared["lambda_values"], dtype=np.float32)
    n_select = int(prepared["n_start_frames_used"])
    L = int(lambdas.shape[0])
    success_a = np.zeros((n_select, L), dtype=np.float32)
    success_b = np.zeros((n_select, L), dtype=np.float32)
    js_a_under_a = np.zeros((n_select, L), dtype=np.float32)
    js_b_under_a = np.zeros((n_select, L), dtype=np.float32)
    js_a_under_b = np.zeros((n_select, L), dtype=np.float32)
    js_b_under_b = np.zeros((n_select, L), dtype=np.float32)
    novelty_under_a = np.zeros((n_select, L), dtype=np.float32)
    novelty_under_b = np.zeros((n_select, L), dtype=np.float32)
    deltae_mean_under_a = np.zeros((n_select, L), dtype=np.float32)
    deltae_mean_under_b = np.zeros((n_select, L), dtype=np.float32)
    success_js_eval_under_a = np.zeros((n_select, L), dtype=np.float32)
    success_js_eval_under_b = np.zeros((n_select, L), dtype=np.float32)
    success_js_eval_node_under_a = np.zeros((n_select, L), dtype=np.float32)
    success_js_eval_node_under_b = np.zeros((n_select, L), dtype=np.float32)
    success_js_eval_edge_under_a = np.zeros((n_select, L), dtype=np.float32)
    success_js_eval_edge_under_b = np.zeros((n_select, L), dtype=np.float32)
    raw_deltae = np.zeros((n_select,), dtype=np.float32)
    raw_js_a = np.zeros((n_select,), dtype=np.float32)
    raw_js_b = np.zeros((n_select,), dtype=np.float32)
    for row, out in enumerate(out_rows):
        success_a[row] = np.asarray(out["success_a"], dtype=np.float32)
        success_b[row] = np.asarray(out["success_b"], dtype=np.float32)
        js_a_under_a[row] = np.asarray(out["js_a_under_a"], dtype=np.float32)
        js_b_under_a[row] = np.asarray(out["js_b_under_a"], dtype=np.float32)
        js_a_under_b[row] = np.asarray(out["js_a_under_b"], dtype=np.float32)
        js_b_under_b[row] = np.asarray(out["js_b_under_b"], dtype=np.float32)
        novelty_under_a[row] = np.asarray(out["novelty_under_a"], dtype=np.float32)
        novelty_under_b[row] = np.asarray(out["novelty_under_b"], dtype=np.float32)
        deltae_mean_under_a[row] = np.asarray(out["deltae_mean_under_a"], dtype=np.float32)
        deltae_mean_under_b[row] = np.asarray(out["deltae_mean_under_b"], dtype=np.float32)
        success_js_eval_under_a[row] = np.asarray(out["success_js_eval_under_a"], dtype=np.float32)
        success_js_eval_under_b[row] = np.asarray(out["success_js_eval_under_b"], dtype=np.float32)
        success_js_eval_node_under_a[row] = np.asarray(out["success_js_eval_node_under_a"], dtype=np.float32)
        success_js_eval_node_under_b[row] = np.asarray(out["success_js_eval_node_under_b"], dtype=np.float32)
        success_js_eval_edge_under_a[row] = np.asarray(out["success_js_eval_edge_under_a"], dtype=np.float32)
        success_js_eval_edge_under_b[row] = np.asarray(out["success_js_eval_edge_under_b"], dtype=np.float32)
        raw_deltae[row] = np.float32(out["raw_deltae"])
        raw_js_a[row] = np.float32(out["raw_js_a"])
        raw_js_b[row] = np.float32(out["raw_js_b"])

    auc_a = np.asarray([_normalized_auc(lambdas, success_a[i]) for i in range(n_select)], dtype=np.float32)
    auc_b = np.asarray([_normalized_auc(lambdas, success_b[i]) for i in range(n_select)], dtype=np.float32)
    auc_dir = np.asarray(auc_b - auc_a, dtype=np.float32)
    cost_a = np.asarray(
        [
            _completion_cost_from_curve(
                lambdas,
                success_a[i],
                target_success=float(prepared["completion_target_success"]),
                unreached_value=float(prepared["completion_cost_if_unreached"]),
            )
            for i in range(n_select)
        ],
        dtype=np.float32,
    )
    cost_b = np.asarray(
        [
            _completion_cost_from_curve(
                lambdas,
                success_b[i],
                target_success=float(prepared["completion_target_success"]),
                unreached_value=float(prepared["completion_cost_if_unreached"]),
            )
            for i in range(n_select)
        ],
        dtype=np.float32,
    )
    novelty_frame = np.asarray(
        0.5 * (np.mean(novelty_under_a, axis=1) + np.mean(novelty_under_b, axis=1)),
        dtype=np.float32,
    )
    lacs_component_completion = np.asarray(auc_dir, dtype=np.float32)
    lacs_component_raw = np.asarray(-raw_deltae, dtype=np.float32)
    lacs_component_novelty = np.asarray(novelty_frame, dtype=np.float32)
    success_a_mean = np.asarray(np.mean(success_a, axis=0), dtype=np.float32)
    success_b_mean = np.asarray(np.mean(success_b, axis=0), dtype=np.float32)
    success_a_std = np.asarray(np.std(success_a, axis=0), dtype=np.float32)
    success_b_std = np.asarray(np.std(success_b, axis=0), dtype=np.float32)

    analysis_dir = Path(str(prepared["analysis_dir"]))
    system_dir = Path(str(prepared["system_dir"]))
    npz_path = Path(str(prepared["analysis_npz"]))
    meta_path = Path(str(prepared["analysis_metadata"]))
    residue_keys = [str(v) for v in prepared["residue_keys"]]
    delta_js_selected_residue_indices = np.asarray(prepared["delta_js_selected_residue_indices"], dtype=np.int32)
    delta_js_selected_edge_indices = np.asarray(prepared["delta_js_selected_edge_indices"], dtype=np.int32)
    delta_js_selected_edges = np.asarray(prepared["delta_js_selected_edges"], dtype=np.int32)

    np.savez_compressed(
        npz_path,
        lambda_values=np.asarray(lambdas, dtype=np.float32),
        frame_indices=np.asarray(prepared["frame_idx_sel"], dtype=np.int64),
        constrained_indices=np.asarray(prepared["constrained_indices"], dtype=np.int32),
        constrained_keys=np.asarray([residue_keys[i] for i in prepared["constrained_indices"]], dtype=str),
        constraint_weights=np.asarray(prepared["constraint_weights"], dtype=np.float32),
        constraint_source_mode=np.asarray([str(prepared["constraint_source_mode"])], dtype=str),
        constraint_auto_row_sample_id=np.asarray(
            [str(prepared["constraint_auto_row_sample_id"])] if prepared["constraint_auto_row_sample_id"] else [],
            dtype=str,
        ),
        constraint_auto_impact=np.asarray(prepared["constraint_auto_impact"], dtype=np.float32),
        constraint_auto_ranked_indices=np.asarray(prepared["constraint_auto_ranked_indices"], dtype=np.int32),
        success_a=np.asarray(success_a, dtype=np.float32),
        success_b=np.asarray(success_b, dtype=np.float32),
        success_a_mean=np.asarray(success_a_mean, dtype=np.float32),
        success_b_mean=np.asarray(success_b_mean, dtype=np.float32),
        success_a_std=np.asarray(success_a_std, dtype=np.float32),
        success_b_std=np.asarray(success_b_std, dtype=np.float32),
        auc_a=np.asarray(auc_a, dtype=np.float32),
        auc_b=np.asarray(auc_b, dtype=np.float32),
        auc_dir=np.asarray(auc_dir, dtype=np.float32),
        cost_a=np.asarray(cost_a, dtype=np.float32),
        cost_b=np.asarray(cost_b, dtype=np.float32),
        raw_deltae=np.asarray(raw_deltae, dtype=np.float32),
        raw_js_a=np.asarray(raw_js_a, dtype=np.float32),
        raw_js_b=np.asarray(raw_js_b, dtype=np.float32),
        js_a_under_a=np.asarray(js_a_under_a, dtype=np.float32),
        js_b_under_a=np.asarray(js_b_under_a, dtype=np.float32),
        js_a_under_b=np.asarray(js_a_under_b, dtype=np.float32),
        js_b_under_b=np.asarray(js_b_under_b, dtype=np.float32),
        novelty_under_a=np.asarray(novelty_under_a, dtype=np.float32),
        novelty_under_b=np.asarray(novelty_under_b, dtype=np.float32),
        success_js_eval_under_a=np.asarray(success_js_eval_under_a, dtype=np.float32),
        success_js_eval_under_b=np.asarray(success_js_eval_under_b, dtype=np.float32),
        success_js_eval_node_under_a=np.asarray(success_js_eval_node_under_a, dtype=np.float32),
        success_js_eval_node_under_b=np.asarray(success_js_eval_node_under_b, dtype=np.float32),
        success_js_eval_edge_under_a=np.asarray(success_js_eval_edge_under_a, dtype=np.float32),
        success_js_eval_edge_under_b=np.asarray(success_js_eval_edge_under_b, dtype=np.float32),
        novelty_frame=np.asarray(novelty_frame, dtype=np.float32),
        deltae_mean_under_a=np.asarray(deltae_mean_under_a, dtype=np.float32),
        deltae_mean_under_b=np.asarray(deltae_mean_under_b, dtype=np.float32),
        lacs_component_completion=np.asarray(lacs_component_completion, dtype=np.float32),
        lacs_component_raw=np.asarray(lacs_component_raw, dtype=np.float32),
        lacs_component_novelty=np.asarray(lacs_component_novelty, dtype=np.float32),
        success_mode=np.asarray([str(prepared["success_metric_mode"])], dtype=str),
        js_success_threshold=np.asarray([float(prepared["js_success_threshold"])], dtype=np.float32),
        js_success_margin=np.asarray([float(prepared["js_success_margin"])], dtype=np.float32),
        delta_js_selected_residue_indices=np.asarray(delta_js_selected_residue_indices, dtype=np.int32),
        delta_js_selected_residue_keys=np.asarray([str(residue_keys[int(i)]) for i in delta_js_selected_residue_indices.tolist()], dtype=str),
        delta_js_selected_residue_weights=np.asarray(prepared["delta_js_selected_residue_weights"], dtype=np.float32),
        delta_js_selected_edge_indices=np.asarray(delta_js_selected_edge_indices, dtype=np.int32),
        delta_js_selected_edges=np.asarray(delta_js_selected_edges, dtype=np.int32),
        delta_js_selected_edge_weights=np.asarray(prepared["delta_js_selected_edge_weights"], dtype=np.float32),
        delta_js_node_edge_alpha=np.asarray(
            [float(prepared["delta_js_node_edge_alpha"])] if prepared["delta_js_node_edge_alpha"] is not None else [],
            dtype=np.float32,
        ),
        delta_js_filter_setup_id=np.asarray(
            [str(prepared["delta_js_filter_setup_id"])] if prepared["delta_js_filter_setup_id"] else [],
            dtype=str,
        ),
        delta_js_filter_edge_alpha=np.asarray([float(prepared["delta_js_filter_edge_alpha"])], dtype=np.float32),
        delta_js_filter_success_set_a=np.asarray(prepared["delta_js_filter_success_set_a"], dtype=np.int32),
        delta_js_filter_success_set_b=np.asarray(prepared["delta_js_filter_success_set_b"], dtype=np.int32),
        delta_js_filter_success_union=np.asarray(prepared["delta_js_filter_success_union"], dtype=np.int32),
        delta_js_filter_target_candidates=np.asarray(prepared["delta_js_filter_target_candidates"], dtype=np.int32),
    )

    default_lacs_weights = {"completion": 1.0, "raw_bias": 0.5, "novelty": 0.5}
    lacs_default = (
        default_lacs_weights["completion"] * float(np.median(lacs_component_completion))
        + default_lacs_weights["raw_bias"] * float(np.median(lacs_component_raw))
        - default_lacs_weights["novelty"] * float(np.median(lacs_component_novelty))
    )
    now = _utc_now()
    meta = {
        "analysis_id": str(prepared["analysis_id"]),
        "analysis_type": "ligand_completion",
        "created_at": now,
        "updated_at": now,
        "project_id": str(prepared["project_id"]),
        "system_id": str(prepared["system_id"]),
        "cluster_id": str(prepared["cluster_id"]),
        "model_a_id": prepared["model_a_id"],
        "model_a_name": prepared["model_a_name"],
        "model_a_path": _relativize(Path(str(prepared["model_a_path"])), system_dir),
        "model_b_id": prepared["model_b_id"],
        "model_b_name": prepared["model_b_name"],
        "model_b_path": _relativize(Path(str(prepared["model_b_path"])), system_dir),
        "md_sample_id": str(prepared["md_sample_id"]),
        "md_sample_name": prepared["md_sample_name"],
        "reference_sample_id_a": str(prepared["reference_sample_id_a"]),
        "reference_sample_name_a": prepared["reference_sample_name_a"],
        "reference_sample_id_b": str(prepared["reference_sample_id_b"]),
        "reference_sample_name_b": prepared["reference_sample_name_b"],
        "sampler": prepared["sampler"],
        "lambda_values": [float(v) for v in lambdas.tolist()],
        "n_start_frames_requested": int(prepared["n_start_frames_requested"]),
        "n_start_frames_used": int(prepared["n_start_frames_used"]),
        "n_samples_per_frame": int(prepared["n_samples_per_frame"]),
        "n_steps": int(prepared["n_steps"]),
        "tail_steps": int(prepared["tail_steps"]),
        "target_window_size": int(prepared["target_window_size"]),
        "target_pseudocount": float(prepared["target_pseudocount"]),
        "epsilon_logpenalty": float(prepared["epsilon_logpenalty"]),
        "constraint_source_mode": str(prepared["constraint_source_mode"]),
        "constraint_delta_js_analysis_id": prepared["constraint_delta_js_analysis_id"],
        "constraint_delta_js_sample_id": prepared["constraint_auto_row_sample_id"],
        "constraint_auto_top_k": int(prepared["constraint_auto_top_k"]),
        "constraint_auto_edge_alpha": float(prepared["constraint_auto_edge_alpha"]),
        "constraint_auto_exclude_success": bool(prepared["constraint_auto_exclude_success"]),
        "constraint_weight_mode": str(prepared["constraint_weight_mode"]),
        "constraint_weight_min": float(prepared["constraint_weight_min"]),
        "constraint_weight_max": float(prepared["constraint_weight_max"]),
        "constrained_indices": [int(v) for v in prepared["constrained_indices"]],
        "constrained_keys": [str(residue_keys[v]) for v in prepared["constrained_indices"]],
        "constraint_weights": [float(v) for v in np.asarray(prepared["constraint_weights"], dtype=float).tolist()],
        "gibbs_beta": float(prepared["gibbs_beta"]),
        "sa_beta_hot": float(prepared["sa_beta_hot"]),
        "sa_beta_cold": float(prepared["sa_beta_cold"]),
        "sa_schedule": str(prepared["sa_schedule"]),
        "md_label_mode": str(prepared["md_label_mode"]),
        "drop_invalid": bool(prepared["drop_invalid"]),
        "success_metric_mode": str(prepared["success_metric_mode"]),
        "js_success_threshold": float(prepared["js_success_threshold"]),
        "js_success_margin": float(prepared["js_success_margin"]),
        "delta_js_experiment_id": prepared["delta_js_experiment_id"],
        "delta_js_analysis_id": prepared["delta_js_analysis_id"],
        "delta_js_filter_setup_id": prepared["delta_js_filter_setup_id"],
        "delta_js_filter_edge_alpha": float(prepared["delta_js_filter_edge_alpha"]),
        "delta_js_d_residue_min": float(prepared["delta_js_d_residue_min"]),
        "delta_js_d_residue_max": prepared["delta_js_d_residue_max"],
        "delta_js_d_edge_min": float(prepared["delta_js_d_edge_min"]),
        "delta_js_d_edge_max": prepared["delta_js_d_edge_max"],
        "delta_js_node_edge_alpha": prepared["delta_js_node_edge_alpha"],
        "delta_js_selected_residue_indices": [int(v) for v in delta_js_selected_residue_indices.tolist()],
        "delta_js_selected_residue_keys": [str(residue_keys[int(v)]) for v in delta_js_selected_residue_indices.tolist()],
        "delta_js_selected_edge_indices": [int(v) for v in delta_js_selected_edge_indices.tolist()],
        "delta_js_selected_edges": [[int(r), int(s)] for (r, s) in delta_js_selected_edges.tolist()],
        "deltae_margin": float(prepared["deltae_margin"]),
        "completion_target_success": float(prepared["completion_target_success"]),
        "completion_cost_if_unreached": float(prepared["completion_cost_if_unreached"]),
        "seed": int(prepared["seed"]),
        "workers": int(max(1, workers_used)),
        "paths": {"analysis_npz": str(npz_path.relative_to(system_dir))},
        "lacs_default_weights": default_lacs_weights,
        "summary": {
            "n_residues": int(len(residue_keys)),
            "n_constrained": int(len(prepared["constrained_indices"])),
            "n_constraint_auto_candidates": int(len(prepared["constraint_auto_ranked_indices"])),
            "n_delta_js_residues": int(delta_js_selected_residue_indices.shape[0]),
            "n_delta_js_edges": int(delta_js_selected_edge_indices.shape[0]),
            "n_delta_js_filter_success_a": int(np.asarray(prepared["delta_js_filter_success_set_a"]).shape[0]),
            "n_delta_js_filter_success_b": int(np.asarray(prepared["delta_js_filter_success_set_b"]).shape[0]),
            "n_delta_js_filter_success_union": int(np.asarray(prepared["delta_js_filter_success_union"]).shape[0]),
            "n_delta_js_filter_target_candidates": int(np.asarray(prepared["delta_js_filter_target_candidates"]).shape[0]),
            "auc_a_median": float(np.median(auc_a)),
            "auc_b_median": float(np.median(auc_b)),
            "auc_dir_median": float(np.median(auc_dir)),
            "cost_a_median": float(np.median(cost_a)),
            "cost_b_median": float(np.median(cost_b)),
            "raw_deltae_median": float(np.median(raw_deltae)),
            "novelty_median": float(np.median(novelty_frame)),
            "lacs_default_median": float(lacs_default),
            "success_js_eval_under_a_median": float(np.nanmedian(success_js_eval_under_a))
            if str(prepared["success_metric_mode"]) == "delta_js_edge"
            else None,
            "success_js_eval_under_b_median": float(np.nanmedian(success_js_eval_under_b))
            if str(prepared["success_metric_mode"]) == "delta_js_edge"
            else None,
        },
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
    result = {"metadata": _convert_nan_to_none(meta), "analysis_npz": str(npz_path), "analysis_dir": str(analysis_dir)}
    atomic_pickle_dump(orchestration_paths(analysis_dir)["aggregate"], result)
    return result


def run_ligand_completion_local(
    *,
    progress_callback: ProgressCallback = None,
    **kwargs: Any,
) -> dict[str, Any]:
    prepared = prepare_ligand_completion_batch(**kwargs)
    payloads = prepared["payloads"]
    workers = max(1, int(kwargs.get("n_workers", 1)))
    out_rows = run_local_payload_batch(
        payloads,
        worker_fn=_single_frame_ligand_completion_worker,
        max_workers=workers,
        progress_callback=progress_callback,
        progress_label="Running conditional completion",
    )
    workers_used = 1 if not payloads else max(1, min(workers, len(payloads)))
    return aggregate_ligand_completion_batch(prepared, out_rows, workers_used=workers_used)


_COMBINED_MODEL_CACHE: dict[tuple[str, ...], Any] = {}
_GAUGED_MODEL_CACHE: dict[str, Any] = {}
_POTTS_NN_WORKER_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _load_combined_model_cached(model_paths: Sequence[str]) -> Any:
    from phase.potts.sampling_run import _load_combined_model

    key = tuple(str(v) for v in model_paths)
    if key not in _COMBINED_MODEL_CACHE:
        _COMBINED_MODEL_CACHE[key] = _load_combined_model(key)
    return _COMBINED_MODEL_CACHE[key]


def _load_gauged_model_cached(model_path: str) -> Any:
    key = str(model_path)
    if key not in _GAUGED_MODEL_CACHE:
        _GAUGED_MODEL_CACHE[key] = zero_sum_gauge_model(load_potts_model(key))
    return _GAUGED_MODEL_CACHE[key]


def _get_potts_nn_worker_cache(prepared: dict[str, Any]) -> dict[str, Any]:
    key = (str(prepared["prepared_path"]), str(prepared["model_path"]))
    cached = _POTTS_NN_WORKER_CACHE.get(key)
    if cached is not None:
        return cached

    model = _load_gauged_model_cached(str(prepared["model_path"]))
    md_unique = np.ascontiguousarray(np.asarray(prepared["md_unique_sequences"], dtype=np.int32))
    md_unique_t = np.ascontiguousarray(md_unique.T)
    h = [np.asarray(v, dtype=float) for v in model.h]
    edges = [(int(r), int(s)) for r, s in model.edges]
    max_gap_h = np.asarray(prepared["max_gap_h"], dtype=float)
    max_gap_J = np.asarray(prepared["max_gap_J"], dtype=float)
    z_node = float(prepared["z_node"])
    z_edge = float(prepared["z_edge"])
    z_edge_per_residue = np.asarray(prepared["z_edge_per_residue"], dtype=float)

    edge_mats = [np.asarray(model.J[(r, s)], dtype=float) for r, s in edges]
    edge_r_idx = np.asarray([int(r) for r, _ in edges], dtype=np.int32)
    edge_s_idx = np.asarray([int(s) for _, s in edges], dtype=np.int32)
    md_edge_r = md_unique_t[edge_r_idx] if edge_r_idx.size else np.zeros((0, md_unique.shape[0]), dtype=np.int32)
    md_edge_s = md_unique_t[edge_s_idx] if edge_s_idx.size else np.zeros((0, md_unique.shape[0]), dtype=np.int32)

    cached = {
        "model": model,
        "h": h,
        "edges": edges,
        "max_gap_h": max_gap_h,
        "max_gap_J": max_gap_J,
        "z_node": z_node,
        "z_edge": z_edge,
        "z_edge_per_residue": z_edge_per_residue,
        "md_unique": md_unique,
        "md_unique_t": md_unique_t,
        "edge_mats": edge_mats,
        "edge_r_idx": edge_r_idx,
        "edge_s_idx": edge_s_idx,
        "md_edge_r": md_edge_r,
        "md_edge_s": md_edge_s,
    }
    _POTTS_NN_WORKER_CACHE[key] = cached
    return cached


def run_sampling_chain_payload(payload: dict[str, Any]) -> dict[str, Any]:
    from phase.potts.sampling_run import (
        _run_gibbs_chain_worker,
        _run_rex_chain_worker,
        _run_sa_chain_worker,
        _run_sa_independent_worker,
    )

    worker_kind = str(payload.get("worker_kind") or "").strip().lower()
    if worker_kind == "gibbs_single":
        model = _load_combined_model_cached(payload.get("model_npz") or [])
        return _run_gibbs_chain_worker(
            {
                "model": model,
                "beta": float(payload["beta"]),
                "n_samples": int(payload["n_samples"]),
                "burn_in": int(payload["burn_in"]),
                "thinning": int(payload["thinning"]),
                "seed": int(payload["seed"]),
                "progress": bool(payload.get("progress", False)),
                "progress_mode": str(payload.get("progress_mode", "samples")),
                "progress_desc": str(payload.get("progress_desc") or "Gibbs samples"),
                "progress_position": payload.get("progress_position"),
            }
        )
    if worker_kind == "gibbs_rex":
        model = _load_combined_model_cached(payload.get("model_npz") or [])
        run = _run_rex_chain_worker(
            {
                "model": model,
                "betas": payload["betas"],
                "sweeps_per_round": int(payload["sweeps_per_round"]),
                "n_rounds": int(payload["n_rounds"]),
                "burn_in_rounds": int(payload["burn_in_rounds"]),
                "thinning_rounds": int(payload["thinning_rounds"]),
                "seed": int(payload["seed"]),
                "progress": bool(payload.get("progress", False)),
                "progress_mode": str(payload.get("progress_mode", "samples")),
                "progress_desc": str(payload.get("progress_desc") or "REX samples"),
                "progress_position": payload.get("progress_position"),
            }
        )
        samples_by_beta = run.get("samples_by_beta") if isinstance(run, dict) else None
        labels = None
        if isinstance(samples_by_beta, dict):
            labels = samples_by_beta.get(float(payload["target_beta"]))
        if not isinstance(labels, np.ndarray):
            labels = np.zeros((0, len(model.h)), dtype=np.int32)
        return {
            "labels": np.asarray(labels, dtype=np.int32),
            "burnin_clipped": bool(payload.get("burnin_clipped", False)),
        }
    if worker_kind == "sa_independent":
        return _run_sa_independent_worker(payload)
    if worker_kind == "sa_chain":
        return _run_sa_chain_worker(payload)
    raise ValueError(f"Unknown sampling worker_kind: {worker_kind!r}")


def prepare_sampling_batch(
    *,
    cluster_npz: str,
    sa_md_sample_npz: str | None = None,
    results_dir: str | Path,
    model_npz: Sequence[str],
    sampling_method: str,
    beta: float,
    seed: int,
    progress: bool = False,
    gibbs_method: str = "single",
    gibbs_samples: int = 500,
    gibbs_burnin: int = 50,
    gibbs_thin: int = 2,
    gibbs_chains: int = 1,
    rex_betas: str = "",
    rex_n_replicas: int = 8,
    rex_beta_min: float = 0.2,
    rex_beta_max: float = 1.0,
    rex_spacing: str = "geom",
    rex_rounds: int = 2000,
    rex_burnin_rounds: int = 50,
    rex_sweeps_per_round: int = 2,
    rex_thin_rounds: int = 1,
    rex_chains: int = 1,
    sa_reads: int = 2000,
    sa_chains: int = 1,
    sa_sweeps: int = 2000,
    sa_beta_hot: float = 0.0,
    sa_beta_cold: float = 0.0,
    sa_schedule_type: str = "geometric",
    sa_custom_beta_schedule: Sequence[float] | str | None = None,
    sa_num_sweeps_per_beta: int = 1,
    sa_randomize_order: bool = False,
    sa_acceptance_criteria: str = "Metropolis",
    sa_init: str = "md",
    sa_init_md_frame: int = -1,
    sa_restart: str = "independent",
    sa_restart_topk: int = 200,
    sa_md_sample_id: str = "",
    sa_md_state_ids: str = "",
    penalty_safety: float = 8.0,
    repair: str = "none",
) -> dict[str, Any]:
    from phase.potts.sampling_run import (
        _load_combined_model,
        _normalize_model_paths,
        _normalize_sa_restart,
        _normalize_sa_schedule_type,
        _parse_float_list,
        _parse_sa_custom_schedule,
    )
    from phase.potts.sampling import make_beta_ladder

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_path = results_dir / SAMPLE_NPZ_FILENAME
    model_paths = _normalize_model_paths(model_npz)
    model = _load_combined_model(model_paths)
    n_residues = int(len(model.h))
    method = (sampling_method or "gibbs").strip().lower()
    if method not in {"gibbs", "sa"}:
        raise ValueError("--sampling-method must be gibbs or sa.")

    payloads: list[dict[str, Any]] = []
    requested_workers = 1
    progress_label = "Sampling"
    if method == "gibbs":
        gm = (gibbs_method or "single").strip().lower()
        if gm not in {"single", "rex"}:
            raise ValueError("--gibbs-method must be single or rex.")
        if gm == "single":
            n_chains = max(1, int(gibbs_chains))
            total_samples = max(0, int(gibbs_samples))
            if total_samples > 0 and n_chains > total_samples:
                n_chains = total_samples
            base = total_samples // max(1, n_chains)
            extra = total_samples % max(1, n_chains)
            for idx in range(max(1, n_chains)):
                n_samples = base + (1 if idx < extra else 0)
                if n_samples <= 0:
                    continue
                payloads.append(
                    {
                        "worker_kind": "gibbs_single",
                        "chain_index": int(idx),
                        "model_npz": list(model_paths),
                        "beta": float(beta),
                        "n_samples": int(n_samples),
                        "burn_in": int(gibbs_burnin),
                        "thinning": int(gibbs_thin),
                        "seed": int(seed) + idx,
                        "progress": bool(progress),
                        "progress_mode": "samples",
                        "progress_desc": f"Gibbs chain {idx + 1}/{max(1, n_chains)} samples",
                        "progress_position": int(idx),
                    }
                )
            requested_workers = max(1, min(max(1, n_chains), max(1, len(payloads))))
            progress_label = "Running Gibbs chains"
        else:
            if rex_betas.strip():
                betas = _parse_float_list(rex_betas)
            else:
                betas = make_beta_ladder(
                    beta_min=float(rex_beta_min),
                    beta_max=float(rex_beta_max),
                    n_replicas=int(rex_n_replicas),
                    spacing=str(rex_spacing),
                )
            if all(abs(float(b) - float(beta)) > 1e-12 for b in betas):
                betas = sorted(set(list(betas) + [float(beta)]))
            total_rounds = max(1, int(rex_rounds))
            n_chains = max(1, int(rex_chains))
            if n_chains > total_rounds:
                n_chains = total_rounds
            base_rounds = total_rounds // max(1, n_chains)
            extra = total_rounds % max(1, n_chains)
            for idx in range(max(1, n_chains)):
                rounds = base_rounds + (1 if idx < extra else 0)
                if rounds <= 0:
                    continue
                burn_in = min(int(rex_burnin_rounds), max(0, rounds - 1))
                payloads.append(
                    {
                        "worker_kind": "gibbs_rex",
                        "chain_index": int(idx),
                        "model_npz": list(model_paths),
                        "betas": [float(v) for v in betas],
                        "target_beta": float(beta),
                        "sweeps_per_round": int(rex_sweeps_per_round),
                        "n_rounds": int(rounds),
                        "burn_in_rounds": int(burn_in),
                        "thinning_rounds": int(rex_thin_rounds),
                        "seed": int(seed) + idx,
                        "burnin_clipped": bool(burn_in != int(rex_burnin_rounds)),
                        "progress": bool(progress),
                        "progress_mode": "samples",
                        "progress_desc": f"REX chain {idx + 1}/{max(1, n_chains)} samples",
                        "progress_position": int(idx),
                    }
                )
            requested_workers = max(1, min(max(1, n_chains), max(1, len(payloads))))
            progress_label = "Running replica-exchange chains"
    else:
        if not str(sa_md_sample_npz or "").strip():
            raise ValueError("SA sampling requires sa_md_sample_npz.")
        schedule_type = _normalize_sa_schedule_type(sa_schedule_type or "geometric")
        custom_schedule = _parse_sa_custom_schedule(sa_custom_beta_schedule)
        sweeps_per_beta = int(sa_num_sweeps_per_beta)
        if sweeps_per_beta < 1:
            raise ValueError("--sa-num-sweeps-per-beta must be >= 1.")
        acceptance_raw = str(sa_acceptance_criteria or "Metropolis").strip().lower()
        if acceptance_raw not in {"metropolis", "gibbs"}:
            raise ValueError("--sa-acceptance-criteria must be Metropolis or Gibbs.")
        acceptance = "Metropolis" if acceptance_raw == "metropolis" else "Gibbs"
        if custom_schedule:
            if schedule_type not in {"custom", "geometric", "linear"}:
                raise ValueError("--sa-schedule-type must be geometric, linear, or custom.")
            schedule_type = "custom"
            if any((not np.isfinite(v)) or v < 0 for v in custom_schedule):
                raise ValueError("--sa-custom-beta-schedule values must be finite and >= 0.")
        elif schedule_type not in {"geometric", "linear"}:
            raise ValueError("--sa-schedule-type must be geometric or linear unless using a custom schedule.")
        if (sa_beta_hot and not sa_beta_cold) or (sa_beta_cold and not sa_beta_hot):
            raise ValueError("Provide both --sa-beta-hot and --sa-beta-cold, or neither.")
        beta_range = None
        if sa_beta_hot and sa_beta_cold:
            beta_range = (float(sa_beta_hot), float(sa_beta_cold))
        if custom_schedule and beta_range is not None:
            raise ValueError("Choose either --sa-custom-beta-schedule or --sa-beta-hot/--sa-beta-cold, not both.")
        restart = _normalize_sa_restart(sa_restart or "independent")
        if restart not in {"previous", "md", "independent"}:
            raise ValueError("--sa-restart must be one of: previous, md, independent.")
        n_chains = max(1, int(sa_chains))
        total_reads = max(0, int(sa_reads))
        if total_reads <= 0:
            payloads = []
            requested_workers = 1
        else:
            if n_chains > total_reads:
                n_chains = total_reads
            base = total_reads // max(1, n_chains)
            extra = total_reads % max(1, n_chains)
            worker_kind = "sa_independent" if restart == "independent" else "sa_chain"
            for idx in range(max(1, n_chains)):
                n_reads = base + (1 if idx < extra else 0)
                if n_reads <= 0:
                    continue
                payload: dict[str, Any] = {
                    "worker_kind": worker_kind,
                    "chain_index": int(idx),
                    "model_npz": list(model_paths),
                    "sa_md_sample_npz": str(sa_md_sample_npz or ""),
                    "beta": float(beta),
                    "penalty_safety": float(penalty_safety),
                    "sweeps": int(sa_sweeps),
                    "seed": int(seed) + idx,
                    "beta_range": beta_range,
                    "sa_schedule_type": schedule_type,
                    "sa_custom_beta_schedule": list(custom_schedule),
                    "sa_num_sweeps_per_beta": int(sweeps_per_beta),
                    "sa_randomize_order": bool(sa_randomize_order),
                    "sa_acceptance_criteria": str(acceptance),
                    "sa_init": str(sa_init),
                    "sa_init_md_frame": int(sa_init_md_frame),
                    "sa_md_sample_id": str(sa_md_sample_id),
                    "sa_md_state_ids": str(sa_md_state_ids),
                    "repair": str(repair),
                    "sa_restart_topk": int(sa_restart_topk),
                    "progress": bool(progress),
                    "progress_desc": f"SA chain {idx + 1}/{max(1, n_chains)} samples",
                    "progress_position": int(idx),
                }
                if worker_kind == "sa_independent":
                    payload["n_reads"] = int(n_reads)
                else:
                    payload["n_samples"] = int(n_reads)
                    payload["sa_restart"] = restart
                payloads.append(payload)
            requested_workers = max(1, min(max(1, n_chains), max(1, len(payloads))))
        progress_label = "Running SA chains"

    prepared = {
        "results_dir": str(results_dir),
        "sample_path": str(sample_path),
        "sampling_method": method,
        "gibbs_method": (gibbs_method or "single").strip().lower(),
        "sa_md_sample_npz": str(sa_md_sample_npz or ""),
        "sa_md_sample_id": str(sa_md_sample_id or ""),
        "n_residues": int(n_residues),
        "requested_workers": int(requested_workers),
        "progress_label": progress_label,
        "payloads": payloads,
        "progress": bool(progress),
    }
    atomic_pickle_dump(orchestration_paths(results_dir)["prepared"], prepared)
    return prepared


def aggregate_sampling_batch(
    prepared: dict[str, Any],
    out_rows: Sequence[dict[str, Any]],
    *,
    workers_used: int,
) -> dict[str, Any]:
    sample_path = Path(str(prepared["sample_path"]))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    n_residues = int(prepared["n_residues"])
    method = str(prepared["sampling_method"])

    parts_labels: list[np.ndarray] = []
    parts_invalid: list[np.ndarray] = []
    parts_valid_counts: list[np.ndarray] = []
    burnin_clipped = False

    for out in out_rows:
        labels = np.asarray(out.get("labels"), dtype=np.int32)
        if labels.size:
            parts_labels.append(labels)
        invalid_mask = out.get("invalid_mask")
        valid_counts = out.get("valid_counts")
        if invalid_mask is not None:
            parts_invalid.append(np.asarray(invalid_mask, dtype=bool))
        if valid_counts is not None:
            parts_valid_counts.append(np.asarray(valid_counts, dtype=np.int32))
        burnin_clipped = burnin_clipped or bool(out.get("burnin_clipped", False))

    labels = (
        np.concatenate(parts_labels, axis=0)
        if parts_labels
        else np.zeros((0, n_residues), dtype=np.int32)
    )
    if method == "sa":
        invalid_mask = (
            np.concatenate(parts_invalid, axis=0)
            if parts_invalid
            else np.zeros((labels.shape[0],), dtype=bool)
        )
        valid_counts = (
            np.concatenate(parts_valid_counts, axis=0)
            if parts_valid_counts
            else np.zeros((labels.shape[0],), dtype=np.int32)
        )
        save_sample_npz(sample_path, labels=labels, invalid_mask=invalid_mask, valid_counts=valid_counts)
    else:
        save_sample_npz(sample_path, labels=labels)

    result = {
        "sample_path": str(sample_path),
        "n_samples": int(labels.shape[0]),
        "n_residues": int(labels.shape[1]) if labels.ndim == 2 else int(n_residues),
        "workers_used": int(max(1, workers_used)),
        "burnin_clipped": bool(burnin_clipped),
    }
    atomic_pickle_dump(orchestration_paths(sample_path.parent)["aggregate"], result)
    return result


def run_sampling_local(
    *,
    progress_callback: ProgressCallback = None,
    max_workers_override: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    prepared = prepare_sampling_batch(**kwargs)
    payloads = prepared["payloads"]
    max_workers = prepared["requested_workers"] if max_workers_override is None else max(1, int(max_workers_override))
    out_rows = run_local_payload_batch(
        payloads,
        worker_fn=run_sampling_chain_payload,
        max_workers=max_workers,
        progress_callback=progress_callback,
        progress_label=str(prepared["progress_label"]),
    )
    workers_used = 1 if not payloads else max(1, min(max_workers, len(payloads)))
    return aggregate_sampling_batch(prepared, out_rows, workers_used=workers_used)


def _filter_lambda_sampling_params(raw: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "gibbs_method": "rex",
        "beta": 1.0,
        "gibbs_samples": 500,
        "gibbs_burnin": 50,
        "gibbs_thin": 2,
        "rex_spacing": "geom",
        "rex_rounds": 2000,
        "rex_burnin_rounds": 50,
        "rex_sweeps_per_round": 2,
        "rex_thin_rounds": 1,
        "rex_n_replicas": 8,
        "rex_beta_min": 0.2,
        "rex_beta_max": 1.0,
        "seed": 0,
    }

    out: dict[str, Any] = {"sampling_method": "gibbs"}

    def _maybe(key: str, value: Any) -> None:
        if value in (None, "", [], {}):
            return
        if key in defaults and defaults[key] == value:
            return
        out[key] = value

    gm = str(raw.get("gibbs_method") or "rex").lower()
    if gm not in {"single", "rex"}:
        gm = "rex"
    _maybe("gibbs_method", gm)
    _maybe("beta", float(raw.get("beta") or 1.0))

    if gm == "single":
        _maybe("gibbs_samples", int(raw.get("gibbs_samples") or defaults["gibbs_samples"]))
        _maybe("gibbs_burnin", int(raw.get("gibbs_burnin") or defaults["gibbs_burnin"]))
        _maybe("gibbs_thin", int(raw.get("gibbs_thin") or defaults["gibbs_thin"]))
        chains = raw.get("gibbs_chains")
        if chains is not None:
            _maybe("gibbs_chains", int(chains))
    else:
        rex_betas = raw.get("rex_betas")
        if isinstance(rex_betas, list):
            rex_betas = ",".join(str(v) for v in rex_betas)
        if isinstance(rex_betas, str) and rex_betas.strip():
            _maybe("rex_betas", str(rex_betas).strip())
        else:
            _maybe("rex_beta_min", float(raw.get("rex_beta_min") or defaults["rex_beta_min"]))
            _maybe("rex_beta_max", float(raw.get("rex_beta_max") or defaults["rex_beta_max"]))
            _maybe("rex_n_replicas", int(raw.get("rex_n_replicas") or defaults["rex_n_replicas"]))
            _maybe("rex_spacing", str(raw.get("rex_spacing") or defaults["rex_spacing"]))
        _maybe("rex_rounds", int(raw.get("rex_rounds") or defaults["rex_rounds"]))
        _maybe("rex_burnin_rounds", int(raw.get("rex_burnin_rounds") or defaults["rex_burnin_rounds"]))
        _maybe("rex_sweeps_per_round", int(raw.get("rex_sweeps_per_round") or defaults["rex_sweeps_per_round"]))
        _maybe("rex_thin_rounds", int(raw.get("rex_thin_rounds") or defaults["rex_thin_rounds"]))
        chains = raw.get("rex_chains")
        if chains is not None:
            _maybe("rex_chains", int(chains))
    seed = raw.get("seed")
    if seed is not None:
        _maybe("seed", int(seed))
    return out


def prepare_lambda_sweep_batch(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_a_id: str,
    model_b_id: str,
    reference_sample_id_a: str | None = None,
    reference_sample_id_b: str | None = None,
    comparison_sample_ids: Sequence[str] | str | None = None,
    md_sample_id_1: str | None = None,
    md_sample_id_2: str | None = None,
    md_sample_id_3: str | None = None,
    series_id: str | None = None,
    series_label: str | None = None,
    lambda_count: int = 11,
    alpha: float = 0.5,
    md_label_mode: str = "assigned",
    keep_invalid: bool = False,
    gibbs_method: str = "rex",
    beta: float = 1.0,
    seed: int = 0,
    gibbs_samples: int = 500,
    gibbs_burnin: int = 50,
    gibbs_thin: int = 2,
    gibbs_chains: int = 1,
    rex_betas: str | Sequence[float] | None = None,
    rex_beta_min: float = 0.2,
    rex_beta_max: float = 1.0,
    rex_spacing: str = "geom",
    rex_n_replicas: int = 8,
    rex_rounds: int = 2000,
    rex_burnin_rounds: int = 50,
    rex_sweeps_per_round: int = 2,
    rex_thin_rounds: int = 1,
    rex_chains: int = 1,
    n_workers: int | None = None,
) -> dict[str, Any]:
    from phase.potts.sampling_run import _parse_float_list
    from phase.potts.sampling import make_beta_ladder

    if int(lambda_count) < 2:
        raise ValueError("lambda_count must be >= 2.")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be in [0,1].")
    def _normalize_ids(values: Sequence[str] | str | None) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [v.strip() for v in values.split(",") if v.strip()]
        return [str(v).strip() for v in values if str(v).strip()]

    resolved_reference_a = str(reference_sample_id_a or md_sample_id_1 or "").strip()
    resolved_reference_b = str(reference_sample_id_b or md_sample_id_2 or "").strip()
    resolved_comparison_ids = _normalize_ids(comparison_sample_ids)
    if not resolved_comparison_ids:
        legacy_c = str(md_sample_id_3 or "").strip()
        if legacy_c:
            resolved_comparison_ids = [legacy_c]

    if not resolved_reference_a or not resolved_reference_b:
        raise ValueError("reference_sample_id_a and reference_sample_id_b are required.")
    if not resolved_comparison_ids:
        raise ValueError("comparison_sample_ids must contain at least one sample id.")
    reference_sample_ids = [resolved_reference_a, resolved_reference_b, *resolved_comparison_ids]
    if len(set(reference_sample_ids)) != len(reference_sample_ids):
        raise ValueError("Lambda sweep reference and comparison samples must be distinct.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    entry = store.get_cluster_entry(project_id, system_id, cluster_id)

    models_meta = entry.get("potts_models") or []
    model_a_meta = next((m for m in models_meta if isinstance(m, dict) and m.get("model_id") == model_a_id), None)
    model_b_meta = next((m for m in models_meta if isinstance(m, dict) and m.get("model_id") == model_b_id), None)
    if not model_a_meta or not model_a_meta.get("path"):
        raise FileNotFoundError(f"Endpoint model A not found on this cluster: {model_a_id}")
    if not model_b_meta or not model_b_meta.get("path"):
        raise FileNotFoundError(f"Endpoint model B not found on this cluster: {model_b_id}")

    for meta, label in ((model_a_meta, "model_a_id"), (model_b_meta, "model_b_id")):
        params = meta.get("params") or {}
        kind = str(params.get("delta_kind") or "")
        if kind.startswith("delta"):
            raise ValueError(f"{label} is a delta-only model ({kind}); choose a combined or standard endpoint.")

    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    samples_dir = cluster_dirs["samples_dir"]
    analyses_dir = cluster_dirs["cluster_dir"] / "analyses" / "lambda_sweep"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    existing_samples = store.list_samples(project_id, system_id, cluster_id)
    existing_sample_ids = {str(s.get("sample_id") or "") for s in existing_samples if isinstance(s, dict)}
    missing_samples = [sid for sid in reference_sample_ids if sid not in existing_sample_ids]
    if missing_samples:
        raise FileNotFoundError(f"Reference/comparison sample(s) not found on this cluster: {', '.join(missing_samples)}")

    model_a_name = str(model_a_meta.get("name") or model_a_id)
    model_b_name = str(model_b_meta.get("name") or model_b_id)
    model_a_path = store.resolve_path(project_id, system_id, str(model_a_meta.get("path")))
    model_b_path = store.resolve_path(project_id, system_id, str(model_b_meta.get("path")))
    if not model_a_path.exists():
        raise FileNotFoundError(f"Model A NPZ missing on disk: {model_a_path}")
    if not model_b_path.exists():
        raise FileNotFoundError(f"Model B NPZ missing on disk: {model_b_path}")

    lambda_values = np.linspace(0.0, 1.0, int(lambda_count)).astype(float)
    resolved_series_id = str(series_id or uuid.uuid4())
    resolved_series_label = str(series_label or "").strip() or f"Lambda sweep {time.strftime('%Y%m%d %H:%M')}"
    analysis_id = str(uuid.uuid4())
    analysis_dir = analyses_dir / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    gibbs_method_norm = str(gibbs_method or "rex").lower()
    if gibbs_method_norm not in {"single", "rex"}:
        gibbs_method_norm = "rex"

    chain_specs_by_lambda: list[list[dict[str, Any]]] = []
    payloads: list[dict[str, Any]] = []
    sampling_params = {
        "gibbs_method": gibbs_method_norm,
        "beta": float(beta),
        "seed": int(seed),
        "gibbs_samples": int(gibbs_samples),
        "gibbs_burnin": int(gibbs_burnin),
        "gibbs_thin": int(gibbs_thin),
        "gibbs_chains": int(gibbs_chains),
        "rex_betas": ",".join(str(float(v)) for v in rex_betas) if isinstance(rex_betas, (list, tuple)) else str(rex_betas or ""),
        "rex_n_replicas": int(rex_n_replicas),
        "rex_beta_min": float(rex_beta_min),
        "rex_beta_max": float(rex_beta_max),
        "rex_spacing": str(rex_spacing),
        "rex_rounds": int(rex_rounds),
        "rex_burnin_rounds": int(rex_burnin_rounds),
        "rex_sweeps_per_round": int(rex_sweeps_per_round),
        "rex_thin_rounds": int(rex_thin_rounds),
        "rex_chains": int(rex_chains),
    }

    sample_records: list[dict[str, Any]] = []
    requested_workers = int(n_workers) if n_workers is not None else 0
    if requested_workers < 0:
        requested_workers = 0

    for lambda_index, lam in enumerate(lambda_values.tolist()):
        sample_id = str(uuid.uuid4())
        sample_dir = samples_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        display_name = f"{resolved_series_label} λ={float(lam):.3f}"
        rel_summary = str((sample_dir / SAMPLE_NPZ_FILENAME).relative_to(system_dir))
        sample_records.append(
            {
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
                "sample_name": display_name,
                "lambda": float(lam),
                "lambda_index": int(lambda_index),
                "path": rel_summary,
            }
        )
        chain_specs: list[dict[str, Any]] = []
        if gibbs_method_norm == "single":
            n_chains = max(1, int(gibbs_chains))
            total_samples = max(0, int(gibbs_samples))
            if total_samples > 0 and n_chains > total_samples:
                n_chains = total_samples
            base = total_samples // max(1, n_chains)
            extra = total_samples % max(1, n_chains)
            for chain_index in range(max(1, n_chains)):
                n_samples = base + (1 if chain_index < extra else 0)
                if n_samples <= 0:
                    continue
                payload = {
                    "worker_kind": "lambda_gibbs_single",
                    "lambda_index": int(lambda_index),
                    "chain_index": int(chain_index),
                    "lambda_value": float(lam),
                    "model_a_path": str(model_a_path),
                    "model_b_path": str(model_b_path),
                    "beta": float(beta),
                    "n_samples": int(n_samples),
                    "burn_in": int(gibbs_burnin),
                    "thinning": int(gibbs_thin),
                    "seed": int(seed) + int(lambda_index) * 10000 + int(chain_index),
                }
                chain_specs.append(payload)
                payloads.append(payload)
        else:
            if isinstance(rex_betas, (list, tuple)):
                betas = [float(v) for v in rex_betas]
            elif isinstance(rex_betas, str) and rex_betas.strip():
                betas = _parse_float_list(rex_betas)
            else:
                betas = make_beta_ladder(
                    beta_min=float(rex_beta_min),
                    beta_max=float(rex_beta_max),
                    n_replicas=int(rex_n_replicas),
                    spacing=str(rex_spacing),
                )
            if all(abs(float(v) - float(beta)) > 1e-12 for v in betas):
                betas = sorted(set(list(betas) + [float(beta)]))
            total_rounds = max(1, int(rex_rounds))
            n_chains = max(1, int(rex_chains))
            if n_chains > total_rounds:
                n_chains = total_rounds
            base_rounds = total_rounds // max(1, n_chains)
            extra = total_rounds % max(1, n_chains)
            for chain_index in range(max(1, n_chains)):
                rounds = base_rounds + (1 if chain_index < extra else 0)
                if rounds <= 0:
                    continue
                burn_in = min(int(rex_burnin_rounds), max(0, rounds - 1))
                payload = {
                    "worker_kind": "lambda_gibbs_rex",
                    "lambda_index": int(lambda_index),
                    "chain_index": int(chain_index),
                    "lambda_value": float(lam),
                    "model_a_path": str(model_a_path),
                    "model_b_path": str(model_b_path),
                    "target_beta": float(beta),
                    "betas": [float(v) for v in betas],
                    "sweeps_per_round": int(rex_sweeps_per_round),
                    "n_rounds": int(rounds),
                    "burn_in_rounds": int(burn_in),
                    "thinning_rounds": int(rex_thin_rounds),
                    "seed": int(seed) + int(lambda_index) * 10000 + int(chain_index),
                    "burnin_clipped": bool(burn_in != int(rex_burnin_rounds)),
                }
                chain_specs.append(payload)
                payloads.append(payload)
        chain_specs_by_lambda.append(chain_specs)
        if requested_workers <= 0:
            requested_workers = max(requested_workers, len(chain_specs))

    if n_workers is None:
        requested_workers = max(1, len(payloads)) if payloads else 1

    npz_path = analysis_dir / "analysis.npz"
    meta_path = analysis_dir / ANALYSIS_METADATA_FILENAME
    prepared = {
        "analysis_id": analysis_id,
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(npz_path),
        "analysis_metadata": str(meta_path),
        "project_id": str(project_id),
        "system_id": str(system_id),
        "cluster_id": str(cluster_id),
        "system_dir": str(system_dir),
        "model_a_id": str(model_a_id),
        "model_b_id": str(model_b_id),
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "model_a_path": str(model_a_path),
        "model_b_path": str(model_b_path),
        "reference_sample_id_a": resolved_reference_a,
        "reference_sample_id_b": resolved_reference_b,
        "comparison_sample_ids": resolved_comparison_ids,
        "reference_sample_ids": reference_sample_ids,
        "series_id": resolved_series_id,
        "series_label": resolved_series_label,
        "lambda_values": np.asarray(lambda_values, dtype=np.float32),
        "sample_records": sample_records,
        "sampling_params": sampling_params,
        "gibbs_method": gibbs_method_norm,
        "md_label_mode": str(md_label_mode or "assigned"),
        "drop_invalid": bool(not keep_invalid),
        "alpha": float(alpha),
        "payloads": payloads,
        "requested_workers": int(max(1, requested_workers)) if payloads else 1,
    }
    atomic_pickle_dump(orchestration_paths(analysis_dir)["prepared"], prepared)
    return prepared


def run_lambda_sweep_payload(payload: dict[str, Any]) -> dict[str, Any]:
    worker_kind = str(payload.get("worker_kind") or "").strip().lower()
    lam = float(payload["lambda_value"])
    endpoint_b = _load_gauged_model_cached(str(payload["model_b_path"]))
    endpoint_a = _load_gauged_model_cached(str(payload["model_a_path"]))
    model = interpolate_potts_models(endpoint_b, endpoint_a, lam)
    if worker_kind == "lambda_gibbs_single":
        from phase.potts.sampling_run import _run_gibbs_chain_worker

        return _run_gibbs_chain_worker(
            {
                "model": model,
                "beta": float(payload["beta"]),
                "n_samples": int(payload["n_samples"]),
                "burn_in": int(payload["burn_in"]),
                "thinning": int(payload["thinning"]),
                "seed": int(payload["seed"]),
                "progress": False,
                "progress_mode": "samples",
                "progress_desc": f"λ={lam:.3f} Gibbs",
            }
        )
    if worker_kind == "lambda_gibbs_rex":
        from phase.potts.sampling_run import _run_rex_chain_worker

        run = _run_rex_chain_worker(
            {
                "model": model,
                "betas": payload["betas"],
                "sweeps_per_round": int(payload["sweeps_per_round"]),
                "n_rounds": int(payload["n_rounds"]),
                "burn_in_rounds": int(payload["burn_in_rounds"]),
                "thinning_rounds": int(payload["thinning_rounds"]),
                "seed": int(payload["seed"]),
                "progress": False,
                "progress_mode": "samples",
                "progress_desc": f"λ={lam:.3f} REX",
            }
        )
        samples_by_beta = run.get("samples_by_beta") if isinstance(run, dict) else None
        labels = None
        if isinstance(samples_by_beta, dict):
            labels = samples_by_beta.get(float(payload["target_beta"]))
        if not isinstance(labels, np.ndarray):
            labels = np.zeros((0, len(model.h)), dtype=np.int32)
        return {"labels": np.asarray(labels, dtype=np.int32), "burnin_clipped": bool(payload.get("burnin_clipped", False))}
    raise ValueError(f"Unknown lambda sweep worker_kind: {worker_kind!r}")


def aggregate_lambda_sweep_batch(
    prepared: dict[str, Any],
    out_rows: Sequence[dict[str, Any]],
    *,
    workers_used: int,
) -> dict[str, Any]:
    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    project_id = str(prepared["project_id"])
    system_id = str(prepared["system_id"])
    cluster_id = str(prepared["cluster_id"])
    system_dir = Path(str(prepared["system_dir"]))
    analysis_dir = Path(str(prepared["analysis_dir"]))
    npz_path = Path(str(prepared["analysis_npz"]))
    meta_path = Path(str(prepared["analysis_metadata"]))
    lambda_values = np.asarray(prepared["lambda_values"], dtype=float)
    payloads = prepared.get("payloads") or []
    sample_records = prepared.get("sample_records") or []

    rows_by_lambda: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    burnin_clipped = False
    for payload, out in zip(payloads, out_rows):
        lambda_index = int(payload["lambda_index"])
        chain_index = int(payload["chain_index"])
        rows_by_lambda.setdefault(lambda_index, []).append((chain_index, out))
        burnin_clipped = burnin_clipped or bool(out.get("burnin_clipped", False))

    new_entries: list[dict[str, Any]] = []
    sample_ids: list[str] = []
    sample_names: list[str] = []
    for record in sample_records:
        lambda_index = int(record["lambda_index"])
        sample_dir = Path(str(record["sample_dir"]))
        sample_path = sample_dir / SAMPLE_NPZ_FILENAME
        chain_rows = sorted(rows_by_lambda.get(lambda_index, []), key=lambda item: item[0])
        parts = [np.asarray(row.get("labels"), dtype=np.int32) for _, row in chain_rows if row.get("labels") is not None]
        labels = np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.int32)
        save_sample_npz(sample_path, labels=labels)
        rel_summary = str(sample_path.relative_to(system_dir))
        entry = {
            "sample_id": str(record["sample_id"]),
            "name": str(record["sample_name"]),
            "type": "potts_lambda_sweep",
            "method": "gibbs",
            "source": "lambda_sweep",
            "model_id": None,
            "model_ids": [str(prepared["model_b_id"]), str(prepared["model_a_id"])],
            "model_names": [str(prepared["model_b_name"]), str(prepared["model_a_name"])],
            "created_at": _utc_now(),
            "path": rel_summary,
            "paths": {"summary_npz": rel_summary},
            "params": _filter_lambda_sampling_params(dict(prepared["sampling_params"])),
            "series_kind": "lambda_sweep",
            "series_id": str(prepared["series_id"]),
            "series_label": str(prepared["series_label"]),
            "lambda": float(record["lambda"]),
            "lambda_index": int(record["lambda_index"]),
            "lambda_count": int(len(sample_records)),
            "endpoint_model_a_id": str(prepared["model_a_id"]),
            "endpoint_model_b_id": str(prepared["model_b_id"]),
        }
        new_entries.append(entry)
        sample_ids.append(str(record["sample_id"]))
        sample_names.append(str(record["sample_name"]))

    system_meta = store.get_system(project_id, system_id)
    clusters = system_meta.metastable_clusters or []
    cluster_entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not isinstance(cluster_entry, dict):
        raise FileNotFoundError(f"Cluster '{cluster_id}' not found while persisting lambda sweep samples.")
    samples_list = cluster_entry.get("samples")
    if not isinstance(samples_list, list):
        samples_list = []
    samples_list.extend(new_entries)
    cluster_entry["samples"] = samples_list
    system_meta.metastable_clusters = clusters
    store.save_system(system_meta)

    analysis_payload = compute_lambda_sweep_analysis(
        project_id=project_id,
        system_id=system_id,
        cluster_id=cluster_id,
        model_a_ref=str(prepared["model_a_id"]),
        model_b_ref=str(prepared["model_b_id"]),
        lambda_sample_ids=sample_ids,
        lambdas=lambda_values.tolist(),
        reference_sample_ids=[str(v) for v in prepared["reference_sample_ids"]],
        md_label_mode=str(prepared["md_label_mode"]),
        drop_invalid=bool(prepared["drop_invalid"]),
        alpha=float(prepared["alpha"]),
    )

    np.savez_compressed(
        npz_path,
        lambdas=np.asarray(analysis_payload["lambdas"], dtype=float),
        edges=np.asarray(analysis_payload["edges"], dtype=int),
        node_js_mean=np.asarray(analysis_payload["node_js_mean"], dtype=float),
        edge_js_mean=np.asarray(analysis_payload["edge_js_mean"], dtype=float),
        combined_distance=np.asarray(analysis_payload["combined_distance"], dtype=float),
        deltaE_mean=np.asarray(analysis_payload["deltaE_mean"], dtype=float),
        deltaE_q25=np.asarray(analysis_payload["deltaE_q25"], dtype=float),
        deltaE_q75=np.asarray(analysis_payload["deltaE_q75"], dtype=float),
        sample_ids=np.asarray(analysis_payload["sample_ids"], dtype=str),
        sample_names=np.asarray(analysis_payload["sample_names"], dtype=str),
        reference_sample_ids=np.asarray(analysis_payload["reference_sample_ids"], dtype=str),
        reference_sample_names=np.asarray(analysis_payload["reference_sample_names"], dtype=str),
        comparison_sample_ids=np.asarray(analysis_payload["comparison_sample_ids"], dtype=str),
        comparison_sample_names=np.asarray(analysis_payload["comparison_sample_names"], dtype=str),
        ref_md_sample_ids=np.asarray(analysis_payload["ref_md_sample_ids"], dtype=str),
        ref_md_sample_names=np.asarray(analysis_payload["ref_md_sample_names"], dtype=str),
        alpha=np.asarray([analysis_payload["alpha"]], dtype=float),
        match_ref_index=np.asarray([analysis_payload["match_ref_index"]], dtype=int),
        comparison_ref_indices=np.asarray(analysis_payload["comparison_ref_indices"], dtype=int),
        lambda_star_index=np.asarray([analysis_payload["lambda_star_index"]], dtype=int),
        lambda_star=np.asarray([analysis_payload["lambda_star"]], dtype=float),
        lambda_star_index_by_reference=np.asarray(analysis_payload["lambda_star_index_by_reference"], dtype=int),
        lambda_star_by_reference=np.asarray(analysis_payload["lambda_star_by_reference"], dtype=float),
        match_min=np.asarray([analysis_payload["match_min"]], dtype=float),
        match_min_by_reference=np.asarray(analysis_payload["match_min_by_reference"], dtype=float),
    )

    meta = {
        "analysis_id": str(prepared["analysis_id"]),
        "analysis_type": "lambda_sweep",
        "created_at": _utc_now(),
        "project_id": project_id,
        "system_id": system_id,
        "cluster_id": cluster_id,
        "series_kind": "lambda_sweep",
        "series_id": str(prepared["series_id"]),
        "series_label": str(prepared["series_label"]),
        "model_a_id": str(prepared["model_a_id"]),
        "model_a_name": str(prepared["model_a_name"]),
        "model_b_id": str(prepared["model_b_id"]),
        "model_b_name": str(prepared["model_b_name"]),
        "reference_sample_id_a": str(prepared["reference_sample_id_a"]),
        "reference_sample_id_b": str(prepared["reference_sample_id_b"]),
        "comparison_sample_ids": [str(v) for v in prepared["comparison_sample_ids"]],
        "reference_sample_ids": [str(v) for v in prepared["reference_sample_ids"]],
        "reference_sample_names": analysis_payload.get("reference_sample_names") or [],
        "comparison_sample_names": analysis_payload.get("comparison_sample_names") or [],
        "md_sample_ids": [str(v) for v in prepared["reference_sample_ids"]],
        "md_sample_names": analysis_payload.get("reference_sample_names") or [],
        "md_label_mode": str(prepared["md_label_mode"]),
        "drop_invalid": bool(prepared["drop_invalid"]),
        "alpha": float(prepared["alpha"]),
        "lambda_count": int(lambda_values.shape[0]),
        "workers": int(max(1, workers_used)),
        "paths": {"analysis_npz": _relativize(npz_path, system_dir)},
        "summary": {
            "lambda_star": analysis_payload.get("lambda_star"),
            "lambda_star_index": analysis_payload.get("lambda_star_index"),
            "match_min": analysis_payload.get("match_min"),
            "lambda_star_by_reference": [float(v) for v in np.asarray(analysis_payload["lambda_star_by_reference"], dtype=float).tolist()],
            "lambda_star_index_by_reference": [int(v) for v in np.asarray(analysis_payload["lambda_star_index_by_reference"], dtype=int).tolist()],
            "match_min_by_reference": [float(v) for v in np.asarray(analysis_payload["match_min_by_reference"], dtype=float).tolist()],
            "burnin_clipped": bool(burnin_clipped),
            "sample_ids": sample_ids,
            "sample_names": sample_names,
        },
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
    result = {
        "metadata": _convert_nan_to_none(meta),
        "analysis_id": str(prepared["analysis_id"]),
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(npz_path),
        "sample_ids": sample_ids,
        "sample_names": sample_names,
        "series_id": str(prepared["series_id"]),
    }
    atomic_pickle_dump(orchestration_paths(analysis_dir)["aggregate"], result)
    return result


def run_lambda_sweep_local(
    *,
    progress_callback: ProgressCallback = None,
    max_workers_override: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    prepared = prepare_lambda_sweep_batch(**kwargs)
    payloads = prepared["payloads"]
    max_workers = prepared["requested_workers"] if max_workers_override is None else max(1, int(max_workers_override))
    out_rows = run_local_payload_batch(
        payloads,
        worker_fn=run_lambda_sweep_payload,
        max_workers=max_workers,
        progress_callback=progress_callback,
        progress_label="Running lambda sweep",
    )
    workers_used = 1 if not payloads else max(1, min(max_workers, len(payloads)))
    return aggregate_lambda_sweep_batch(prepared, out_rows, workers_used=workers_used)


def _resolve_cluster_sample_entry(
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_id: str,
    sample_id: str,
) -> dict[str, Any]:
    sample_id = str(sample_id or "").strip()
    if not sample_id:
        raise ValueError("sample_id is required.")
    entries = store.list_samples(project_id, system_id, cluster_id)
    entry = next((s for s in entries if isinstance(s, dict) and str(s.get("sample_id") or "") == sample_id), None)
    if not isinstance(entry, dict):
        raise FileNotFoundError(f"Sample not found on cluster: {sample_id}")
    return entry


def _resolve_cluster_sample_path(
    store: ProjectStore,
    project_id: str,
    system_id: str,
    cluster_dir: Path,
    sample_entry: dict[str, Any],
) -> Path:
    paths = sample_entry.get("paths") or {}
    rel = None
    if isinstance(paths, dict):
        rel = paths.get("summary_npz") or paths.get("path")
    rel = rel or sample_entry.get("path")
    if not rel:
        raise FileNotFoundError(f"Sample entry '{sample_entry.get('sample_id')}' missing NPZ path.")
    p = Path(str(rel))
    if not p.is_absolute():
        resolved = store.resolve_path(project_id, system_id, str(rel))
        if resolved.exists():
            p = resolved
        else:
            alt = cluster_dir / str(rel)
            p = alt if alt.exists() else resolved
    if not p.exists():
        raise FileNotFoundError(f"Sample NPZ not found: {p}")
    return p


def _load_cluster_sample_labels(
    store: ProjectStore,
    *,
    project_id: str,
    system_id: str,
    cluster_dir: Path,
    sample_entry: dict[str, Any],
    md_label_mode: str,
    drop_invalid: bool,
    is_md: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = load_sample_npz(
        _resolve_cluster_sample_path(store, project_id, system_id, cluster_dir, sample_entry)
    )
    labels = sample.labels_halo if is_md and str(md_label_mode or "assigned").lower() == "halo" and sample.labels_halo is not None else sample.labels
    labels = np.asarray(labels, dtype=np.int32)
    if labels.ndim != 2:
        raise ValueError(f"Sample '{sample_entry.get('sample_id')}' labels must be 2D.")
    keep = np.ones(labels.shape[0], dtype=bool)
    if drop_invalid and sample.invalid_mask is not None:
        invalid_mask = np.asarray(sample.invalid_mask, dtype=bool).reshape(-1)
        if invalid_mask.shape[0] == labels.shape[0]:
            keep &= ~invalid_mask
    keep &= np.all(labels >= 0, axis=1)
    labels = labels[keep]
    original_indices = np.arange(int(keep.shape[0]), dtype=np.int64)[keep]
    if sample.frame_indices is not None:
        frame_indices_full = np.asarray(sample.frame_indices, dtype=np.int64).reshape(-1)
        if frame_indices_full.shape[0] == keep.shape[0]:
            frame_indices = frame_indices_full[keep]
        else:
            frame_indices = original_indices.copy()
    else:
        frame_indices = original_indices.copy()
    return labels, original_indices, frame_indices


def _unique_sequences_with_inverse(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    X = np.asarray(X, dtype=np.int32)
    if X.ndim != 2:
        raise ValueError("Expected a 2D label matrix.")
    if X.shape[0] == 0:
        n_res = int(X.shape[1]) if X.ndim == 2 else 0
        return (
            np.zeros((0, n_res), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    unique, inv, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    groups = [np.flatnonzero(inv == idx).astype(np.int64) for idx in range(int(unique.shape[0]))]
    return unique.astype(np.int32), inv.astype(np.int32), counts.astype(np.int64), groups


def _potts_nn_row_worker(payload: dict[str, Any]) -> dict[str, Any]:
    prepared = pickle_load(Path(str(payload["prepared_path"])))
    row = int(payload["row"])
    eps = 1e-12

    sample_unique = np.asarray(prepared["sample_unique_sequences"], dtype=np.int32)
    if row < 0 or row >= sample_unique.shape[0]:
        raise IndexError(f"Invalid sample unique row: {row}")
    s = np.asarray(sample_unique[row], dtype=np.int32)

    cache = _get_potts_nn_worker_cache(prepared)
    h = cache["h"]
    edges = cache["edges"]
    max_gap_h = cache["max_gap_h"]
    max_gap_J = cache["max_gap_J"]
    z_node = float(cache["z_node"])
    z_edge = float(cache["z_edge"])
    z_edge_per_residue = cache["z_edge_per_residue"]
    beta_node = float(prepared["beta_node"])
    beta_edge = float(prepared["beta_edge"])
    normalize = bool(prepared["normalize"])
    compute_per_residue = bool(prepared["compute_per_residue"])
    top_k_candidates = prepared.get("top_k_candidates")
    top_k_candidates = None if top_k_candidates in (None, "", 0) else int(top_k_candidates)
    chunk_size = max(1, int(prepared.get("chunk_size") or 256))
    md_unique = cache["md_unique"]
    md_unique_t = cache["md_unique_t"]
    edge_mats = cache["edge_mats"]
    edge_r_idx = cache["edge_r_idx"]
    edge_s_idx = cache["edge_s_idx"]
    md_edge_r = cache["md_edge_r"]
    md_edge_s = cache["md_edge_s"]

    md_count = int(md_unique.shape[0])
    node_raw_all = np.zeros(md_count, dtype=float)
    for i in range(md_unique_t.shape[0]):
        ref = int(s[i])
        tv = md_unique_t[i]
        vals = np.abs(h[i][tv] - h[i][ref])
        if np.any(tv == ref):
            vals = vals.copy()
            vals[tv == ref] = 0.0
        node_raw_all += vals

    candidate_indices = np.arange(md_count, dtype=np.int64)
    if top_k_candidates is not None and top_k_candidates > 0 and top_k_candidates < md_count:
        node_rank = node_raw_all / (z_node + eps) if normalize else node_raw_all
        candidate_indices = np.argsort(node_rank)[: int(top_k_candidates)].astype(np.int64)

    best_idx = -1
    best_global = float("inf")
    best_node = float("inf")
    best_edge = float("inf")
    denom = beta_node + beta_edge

    for start in range(0, int(candidate_indices.shape[0]), chunk_size):
        cand_idx = candidate_indices[start : start + chunk_size]
        node_raw = np.asarray(node_raw_all[cand_idx], dtype=float)
        edge_raw = np.zeros(cand_idx.shape[0], dtype=float)
        for edge_idx, (r, s_) in enumerate(edges):
            tvr = md_edge_r[edge_idx, cand_idx]
            tvs = md_edge_s[edge_idx, cand_idx]
            mismatch = (tvr != int(s[r])) | (tvs != int(s[s_]))
            if not np.any(mismatch):
                continue
            mat = edge_mats[edge_idx]
            pair_ref = float(mat[int(s[r]), int(s[s_])])
            edge_raw += np.abs(mat[tvr, tvs] - pair_ref) * mismatch
        node_val = node_raw / (z_node + eps) if normalize else node_raw
        edge_val = edge_raw / (z_edge + eps) if normalize else edge_raw
        global_val = (beta_node * node_val + beta_edge * edge_val) / (denom + eps)
        local_best = int(np.argmin(global_val))
        if float(global_val[local_best]) < best_global:
            best_idx = int(cand_idx[local_best])
            best_global = float(global_val[local_best])
            best_node = float(node_val[local_best])
            best_edge = float(edge_val[local_best])

    if best_idx < 0:
        raise RuntimeError(f"Failed to resolve nearest-neighbor row for sample unique index {row}.")

    residue_node = np.zeros_like(max_gap_h, dtype=float)
    residue_edge = np.zeros_like(max_gap_h, dtype=float)
    per_edge = np.zeros(len(edges), dtype=float)
    if compute_per_residue:
        nn = np.asarray(md_unique[best_idx], dtype=np.int32)
        for i in range(int(max_gap_h.shape[0])):
            if int(s[i]) != int(nn[i]):
                residue_node[i] = abs(float(h[i][int(nn[i])]) - float(h[i][int(s[i])]))
        residue_node = residue_node / (max_gap_h + eps) if normalize else residue_node
        for edge_idx, (r, s_) in enumerate(edges):
            if int(s[r]) == int(nn[r]) and int(s[s_]) == int(nn[s_]):
                continue
            mat = edge_mats[edge_idx]
            contrib = abs(float(mat[int(nn[r]), int(nn[s_])]) - float(mat[int(s[r]), int(s[s_])]))
            per_edge[edge_idx] = contrib / (float(max_gap_J[edge_idx]) + eps) if normalize else contrib
            residue_edge[r] += contrib
            residue_edge[s_] += contrib
        residue_edge = residue_edge / (z_edge_per_residue + eps) if normalize else residue_edge
    else:
        nn = np.asarray(md_unique[best_idx], dtype=np.int32)
        for edge_idx, (r, s_) in enumerate(edges):
            if int(s[r]) == int(nn[r]) and int(s[s_]) == int(nn[s_]):
                continue
            mat = edge_mats[edge_idx]
            contrib = abs(float(mat[int(nn[r]), int(nn[s_])]) - float(mat[int(s[r]), int(s[s_])]))
            per_edge[edge_idx] = contrib / (float(max_gap_J[edge_idx]) + eps) if normalize else contrib

    return {
        "row": row,
        "nn_md_unique_idx": int(best_idx),
        "nn_dist_global": float(best_global),
        "nn_dist_node": float(best_node),
        "nn_dist_edge": float(best_edge),
        "nn_dist_residue_node": np.asarray(residue_node, dtype=float),
        "nn_dist_residue_edge": np.asarray(residue_edge, dtype=float),
        "nn_dist_edge_per_edge": np.asarray(per_edge, dtype=float),
    }


def prepare_potts_nn_mapping_batch(
    *,
    project_id: str,
    system_id: str,
    cluster_id: str,
    model_id: str | None = None,
    model_path: str | None = None,
    sample_id: str,
    md_sample_id: str,
    md_label_mode: str = "assigned",
    keep_invalid: bool = False,
    use_unique: bool = True,
    normalize: bool = True,
    compute_per_residue: bool = True,
    alpha: float = 0.75,
    beta_node: float = 1.0,
    beta_edge: float = 1.0,
    top_k_candidates: int | None = None,
    chunk_size: int = 256,
    distance_thresholds: Sequence[float] | None = None,
    n_workers: int | None = None,
) -> dict[str, Any]:
    if not (model_id or model_path):
        raise ValueError("Provide model_id or model_path.")
    if not sample_id or not md_sample_id:
        raise ValueError("sample_id and md_sample_id are required.")
    if float(beta_node) < 0.0 or float(beta_edge) < 0.0:
        raise ValueError("beta_node and beta_edge must be non-negative.")
    if float(beta_node) == 0.0 and float(beta_edge) == 0.0:
        raise ValueError("At least one of beta_node or beta_edge must be > 0.")

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    cluster_dir = cluster_dirs["cluster_dir"]
    analyses_dir = cluster_dir / "analyses" / "potts_nn_mapping"
    analyses_dir.mkdir(parents=True, exist_ok=True)

    sample_entry = _resolve_cluster_sample_entry(store, project_id, system_id, cluster_id, sample_id)
    md_entry = _resolve_cluster_sample_entry(store, project_id, system_id, cluster_id, md_sample_id)
    if str(md_entry.get("type") or "") != "md_eval":
        raise ValueError("md_sample_id must reference an md_eval sample.")

    resolved_model_id = str(model_id or "").strip()
    resolved_model_path = None
    if str(model_path or "").strip():
        resolved_model_path = Path(str(model_path).strip())
    resolved_model_name = resolved_model_id
    if resolved_model_id and resolved_model_path is None:
        models = store.list_potts_models(project_id, system_id, cluster_id)
        entry = next((m for m in models if isinstance(m, dict) and str(m.get("model_id") or "") == resolved_model_id), None)
        if not entry or not entry.get("path"):
            raise FileNotFoundError(f"Potts model_id not found on this cluster: {resolved_model_id}")
        resolved_model_name = str(entry.get("name") or resolved_model_id)
        resolved_model_path = store.resolve_path(project_id, system_id, str(entry.get("path")))
    elif resolved_model_path is not None:
        if not resolved_model_path.is_absolute():
            resolved_model_path = store.resolve_path(project_id, system_id, str(resolved_model_path))
        resolved_model_name = resolved_model_name or resolved_model_path.stem
    if resolved_model_path is None:
        raise ValueError("Provide model_id or model_path.")
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Potts model NPZ not found: {resolved_model_path}")

    sample_labels, sample_original_indices, sample_frame_indices = _load_cluster_sample_labels(
        store,
        project_id=project_id,
        system_id=system_id,
        cluster_dir=cluster_dir,
        sample_entry=sample_entry,
        md_label_mode=str(md_label_mode),
        drop_invalid=bool(not keep_invalid),
        is_md=False,
    )
    md_labels, md_original_indices, md_frame_indices = _load_cluster_sample_labels(
        store,
        project_id=project_id,
        system_id=system_id,
        cluster_dir=cluster_dir,
        sample_entry=md_entry,
        md_label_mode=str(md_label_mode),
        drop_invalid=bool(not keep_invalid),
        is_md=True,
    )
    if sample_labels.shape[0] == 0:
        raise ValueError("Selected sample has no usable frames after filtering invalid/unassigned rows.")
    if md_labels.shape[0] == 0:
        raise ValueError("Selected MD sample has no usable frames after filtering invalid/unassigned rows.")
    if sample_labels.shape[1] != md_labels.shape[1]:
        raise ValueError("Sample and MD label matrices do not have the same number of residues.")

    residue_keys: list[str]
    cluster_npz_path = cluster_dir / "cluster.npz"
    if cluster_npz_path.exists():
        try:
            with np.load(cluster_npz_path, allow_pickle=True) as cluster_npz:
                residue_keys = [str(v) for v in np.asarray(cluster_npz["residue_keys"], dtype=str).tolist()] if "residue_keys" in cluster_npz else []
        except Exception:
            residue_keys = []
    else:
        residue_keys = []
    if len(residue_keys) != int(sample_labels.shape[1]):
        residue_keys = [f"res_{idx + 1}" for idx in range(int(sample_labels.shape[1]))]
    residue_display_labels = _derive_residue_display_labels(
        store=store,
        project_id=project_id,
        system_id=system_id,
        cluster_dir=cluster_dir,
        residue_keys=residue_keys,
    )
    if len(residue_display_labels) != len(residue_keys):
        residue_display_labels = [str(v) for v in residue_keys]

    if bool(use_unique):
        sample_unique, sample_inv, sample_counts, sample_groups = _unique_sequences_with_inverse(sample_labels)
        md_unique, md_inv, md_counts, md_groups = _unique_sequences_with_inverse(md_labels)
    else:
        sample_unique = np.asarray(sample_labels, dtype=np.int32)
        sample_inv = np.arange(sample_labels.shape[0], dtype=np.int32)
        sample_counts = np.ones(sample_labels.shape[0], dtype=np.int64)
        sample_groups = [np.asarray([i], dtype=np.int64) for i in range(sample_labels.shape[0])]
        md_unique = np.asarray(md_labels, dtype=np.int32)
        md_inv = np.arange(md_labels.shape[0], dtype=np.int32)
        md_counts = np.ones(md_labels.shape[0], dtype=np.int64)
        md_groups = [np.asarray([i], dtype=np.int64) for i in range(md_labels.shape[0])]

    model = zero_sum_gauge_model(load_potts_model(str(resolved_model_path)))
    max_gap_h = np.asarray([float(np.max(hr) - np.min(hr)) for hr in model.h], dtype=float)
    edges = [(int(r), int(s_)) for r, s_ in model.edges]
    max_gap_J = np.asarray([float(np.max(model.J[(r, s_)]) - np.min(model.J[(r, s_)])) for r, s_ in edges], dtype=float)
    z_node = float(np.sum(max_gap_h))
    z_edge = float(np.sum(max_gap_J))
    z_edge_per_residue = np.zeros(len(model.h), dtype=float)
    for edge_idx, (r, s_) in enumerate(edges):
        z_edge_per_residue[int(r)] += float(max_gap_J[edge_idx])
        z_edge_per_residue[int(s_)] += float(max_gap_J[edge_idx])

    threshold_values = [float(v) for v in (distance_thresholds or [0.05, 0.1, 0.2])]
    analysis_id = str(uuid.uuid4())
    analysis_dir = analyses_dir / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)

    sample_unique_rep_original_indices = np.asarray([int(sample_original_indices[group[0]]) for group in sample_groups], dtype=np.int64)
    md_unique_rep_original_indices = np.asarray([int(md_original_indices[group[0]]) for group in md_groups], dtype=np.int64)
    sample_unique_rep_frame_indices = np.asarray([int(sample_frame_indices[group[0]]) for group in sample_groups], dtype=np.int64)
    md_unique_rep_frame_indices = np.asarray([int(md_frame_indices[group[0]]) for group in md_groups], dtype=np.int64)

    prepared_path = orchestration_paths(analysis_dir)["prepared"]
    payloads = [{"row": int(row), "prepared_path": str(prepared_path)} for row in range(int(sample_unique.shape[0]))]
    requested_workers = int(n_workers) if n_workers is not None else max(1, len(payloads))
    if requested_workers <= 0:
        requested_workers = max(1, len(payloads))

    prepared = {
        "analysis_id": analysis_id,
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(analysis_dir / "analysis.npz"),
        "analysis_metadata": str(analysis_dir / ANALYSIS_METADATA_FILENAME),
        "project_id": str(project_id),
        "system_id": str(system_id),
        "cluster_id": str(cluster_id),
        "system_dir": str(system_dir),
        "model_id": resolved_model_id or None,
        "model_name": resolved_model_name or None,
        "model_path": str(resolved_model_path),
        "sample_id": str(sample_entry.get("sample_id") or sample_id),
        "sample_name": str(sample_entry.get("name") or sample_id),
        "sample_type": str(sample_entry.get("type") or "sample"),
        "md_sample_id": str(md_entry.get("sample_id") or md_sample_id),
        "md_sample_name": str(md_entry.get("name") or md_sample_id),
        "md_label_mode": str(md_label_mode or "assigned"),
        "drop_invalid": bool(not keep_invalid),
        "use_unique": bool(use_unique),
        "normalize": bool(normalize),
        "compute_per_residue": bool(compute_per_residue),
        "alpha": float(alpha),
        "beta_node": float(beta_node),
        "beta_edge": float(beta_edge),
        "top_k_candidates": None if top_k_candidates is None or int(top_k_candidates) <= 0 else int(top_k_candidates),
        "chunk_size": int(max(1, chunk_size)),
        "distance_thresholds": threshold_values,
        "max_gap_h": np.asarray(max_gap_h, dtype=float),
        "max_gap_J": np.asarray(max_gap_J, dtype=float),
        "residue_keys": np.asarray(residue_keys, dtype=str),
        "residue_display_labels": np.asarray(residue_display_labels, dtype=str),
        "z_node": float(z_node),
        "z_edge": float(z_edge),
        "z_edge_per_residue": np.asarray(z_edge_per_residue, dtype=float),
        "edges": np.asarray(edges, dtype=np.int32),
        "sample_unique_sequences": np.asarray(sample_unique, dtype=np.int32),
        "md_unique_sequences": np.asarray(md_unique, dtype=np.int32),
        "sample_unique_counts": np.asarray(sample_counts, dtype=np.int64),
        "md_unique_counts": np.asarray(md_counts, dtype=np.int64),
        "sample_inv": np.asarray(sample_inv, dtype=np.int32),
        "md_inv": np.asarray(md_inv, dtype=np.int32),
        "sample_unique_rep_original_indices": np.asarray(sample_unique_rep_original_indices, dtype=np.int64),
        "md_unique_rep_original_indices": np.asarray(md_unique_rep_original_indices, dtype=np.int64),
        "sample_unique_rep_frame_indices": np.asarray(sample_unique_rep_frame_indices, dtype=np.int64),
        "md_unique_rep_frame_indices": np.asarray(md_unique_rep_frame_indices, dtype=np.int64),
        "payloads": payloads,
        "requested_workers": int(requested_workers),
        "prepared_path": str(prepared_path),
    }
    atomic_pickle_dump(prepared_path, prepared)
    return prepared


def aggregate_potts_nn_mapping_batch(
    prepared: dict[str, Any],
    out_rows: Sequence[dict[str, Any]],
    *,
    workers_used: int,
) -> dict[str, Any]:
    def _weighted_stats_matrix(matrix: np.ndarray, weights_in: np.ndarray) -> dict[str, np.ndarray]:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            width = int(arr.shape[1]) if arr.ndim == 2 else 0
            zeros = np.zeros((width,), dtype=float)
            return {
                "mean": zeros.copy(),
                "std": zeros.copy(),
                "median": zeros.copy(),
                "q25": zeros.copy(),
                "q75": zeros.copy(),
            }
        weights_local = np.asarray(weights_in, dtype=float).reshape(-1)
        if weights_local.shape[0] != arr.shape[0]:
            raise ValueError("weights length mismatch while aggregating Potts NN mapping stats.")
        weights_local = np.where(np.isfinite(weights_local) & (weights_local > 0), weights_local, 0.0)
        total_weight = float(np.sum(weights_local))
        if total_weight <= 0.0:
            zeros = np.zeros((arr.shape[1],), dtype=float)
            return {
                "mean": zeros.copy(),
                "std": zeros.copy(),
                "median": zeros.copy(),
                "q25": zeros.copy(),
                "q75": zeros.copy(),
            }
        mean = np.average(arr, axis=0, weights=weights_local)
        var = np.average((arr - mean[None, :]) ** 2, axis=0, weights=weights_local)
        std = np.sqrt(np.maximum(var, 0.0))
        order = np.argsort(arr, axis=0)
        sorted_arr = np.take_along_axis(arr, order, axis=0)
        weight_matrix = np.broadcast_to(weights_local[:, None], arr.shape)
        sorted_weights = np.take_along_axis(weight_matrix, order, axis=0)
        cumsum = np.cumsum(sorted_weights, axis=0)
        totals = cumsum[-1]
        col_index = np.arange(arr.shape[1], dtype=int)

        def _pick_quantile(q: float) -> np.ndarray:
            target = np.asarray(totals * float(q), dtype=float)
            hits = cumsum >= target[None, :]
            idx = np.argmax(hits, axis=0)
            return np.asarray(sorted_arr[idx, col_index], dtype=float)

        return {
            "mean": np.asarray(mean, dtype=float),
            "std": np.asarray(std, dtype=float),
            "median": _pick_quantile(0.5),
            "q25": _pick_quantile(0.25),
            "q75": _pick_quantile(0.75),
        }

    analysis_dir = Path(str(prepared["analysis_dir"]))
    npz_path = Path(str(prepared["analysis_npz"]))
    meta_path = Path(str(prepared["analysis_metadata"]))
    system_dir = Path(str(prepared["system_dir"]))
    sample_unique = np.asarray(prepared["sample_unique_sequences"], dtype=np.int32)
    md_unique = np.asarray(prepared["md_unique_sequences"], dtype=np.int32)
    sample_counts = np.asarray(prepared["sample_unique_counts"], dtype=np.int64)
    sample_inv = np.asarray(prepared["sample_inv"], dtype=np.int32)
    md_unique_rep_original_indices = np.asarray(prepared["md_unique_rep_original_indices"], dtype=np.int64)
    md_unique_rep_frame_indices = np.asarray(prepared["md_unique_rep_frame_indices"], dtype=np.int64)
    thresholds = np.asarray(prepared["distance_thresholds"], dtype=float)
    alpha = float(prepared["alpha"])

    rows_sorted = sorted(out_rows, key=lambda row: int(row["row"]))
    nn_md_unique_idx = np.asarray([int(row["nn_md_unique_idx"]) for row in rows_sorted], dtype=np.int32)
    nn_dist_global = np.asarray([float(row["nn_dist_global"]) for row in rows_sorted], dtype=float)
    nn_dist_node = np.asarray([float(row["nn_dist_node"]) for row in rows_sorted], dtype=float)
    nn_dist_edge = np.asarray([float(row["nn_dist_edge"]) for row in rows_sorted], dtype=float)
    nn_dist_residue_node = np.asarray([np.asarray(row["nn_dist_residue_node"], dtype=float) for row in rows_sorted], dtype=float)
    nn_dist_residue_edge = np.asarray([np.asarray(row["nn_dist_residue_edge"], dtype=float) for row in rows_sorted], dtype=float)
    nn_dist_edge_per_edge = np.asarray([np.asarray(row["nn_dist_edge_per_edge"], dtype=float) for row in rows_sorted], dtype=float)
    nn_dist_residue = (1.0 - alpha) * nn_dist_residue_node + alpha * nn_dist_residue_edge

    weights = sample_counts.astype(float)
    total_weight = float(np.sum(weights)) if weights.size else 1.0
    threshold_coverage = np.asarray(
        [float(np.sum(weights[nn_dist_global <= thr]) / total_weight) if total_weight > 0 else 0.0 for thr in thresholds.tolist()],
        dtype=float,
    )
    residue_stats = _weighted_stats_matrix(nn_dist_residue, weights)
    residue_node_stats = _weighted_stats_matrix(nn_dist_residue_node, weights)
    residue_edge_stats = _weighted_stats_matrix(nn_dist_residue_edge, weights)
    edge_stats = _weighted_stats_matrix(nn_dist_edge_per_edge, weights)
    per_residue_mean = residue_stats["mean"]
    per_residue_mean_node = residue_node_stats["mean"]
    per_residue_mean_edge = residue_edge_stats["mean"]
    nn_md_rep_original_idx = md_unique_rep_original_indices[nn_md_unique_idx] if nn_md_unique_idx.size else np.zeros((0,), dtype=np.int64)
    nn_md_rep_frame_idx = md_unique_rep_frame_indices[nn_md_unique_idx] if nn_md_unique_idx.size else np.zeros((0,), dtype=np.int64)

    np.savez_compressed(
        npz_path,
        analysis_format_version=np.asarray([2], dtype=np.int32),
        residue_keys=np.asarray(prepared["residue_keys"], dtype=str),
        residue_display_labels=np.asarray(prepared.get("residue_display_labels", prepared["residue_keys"]), dtype=str),
        sample_unique_sequences=np.asarray(sample_unique, dtype=np.int32),
        md_unique_sequences=np.asarray(md_unique, dtype=np.int32),
        sample_unique_counts=np.asarray(sample_counts, dtype=np.int64),
        md_unique_counts=np.asarray(prepared["md_unique_counts"], dtype=np.int64),
        sample_inv=np.asarray(sample_inv, dtype=np.int32),
        md_inv=np.asarray(prepared["md_inv"], dtype=np.int32),
        sample_unique_rep_original_indices=np.asarray(prepared["sample_unique_rep_original_indices"], dtype=np.int64),
        md_unique_rep_original_indices=np.asarray(md_unique_rep_original_indices, dtype=np.int64),
        sample_unique_rep_frame_indices=np.asarray(prepared["sample_unique_rep_frame_indices"], dtype=np.int64),
        md_unique_rep_frame_indices=np.asarray(md_unique_rep_frame_indices, dtype=np.int64),
        nn_md_unique_idx=np.asarray(nn_md_unique_idx, dtype=np.int32),
        nn_md_rep_original_idx=np.asarray(nn_md_rep_original_idx, dtype=np.int64),
        nn_md_rep_frame_idx=np.asarray(nn_md_rep_frame_idx, dtype=np.int64),
        nn_dist_global=np.asarray(nn_dist_global, dtype=float),
        nn_dist_node=np.asarray(nn_dist_node, dtype=float),
        nn_dist_edge=np.asarray(nn_dist_edge, dtype=float),
        nn_dist_residue_node=np.asarray(nn_dist_residue_node, dtype=float),
        nn_dist_residue_edge=np.asarray(nn_dist_residue_edge, dtype=float),
        nn_dist_edge_per_edge=np.asarray(nn_dist_edge_per_edge, dtype=float),
        nn_dist_residue=np.asarray(nn_dist_residue, dtype=float),
        threshold_values=np.asarray(thresholds, dtype=float),
        threshold_coverage=np.asarray(threshold_coverage, dtype=float),
        per_residue_mean=np.asarray(per_residue_mean, dtype=float),
        per_residue_std=np.asarray(residue_stats["std"], dtype=float),
        per_residue_median=np.asarray(residue_stats["median"], dtype=float),
        per_residue_q25=np.asarray(residue_stats["q25"], dtype=float),
        per_residue_q75=np.asarray(residue_stats["q75"], dtype=float),
        per_residue_mean_node=np.asarray(per_residue_mean_node, dtype=float),
        per_residue_std_node=np.asarray(residue_node_stats["std"], dtype=float),
        per_residue_median_node=np.asarray(residue_node_stats["median"], dtype=float),
        per_residue_q25_node=np.asarray(residue_node_stats["q25"], dtype=float),
        per_residue_q75_node=np.asarray(residue_node_stats["q75"], dtype=float),
        per_residue_mean_edge=np.asarray(per_residue_mean_edge, dtype=float),
        per_residue_std_edge=np.asarray(residue_edge_stats["std"], dtype=float),
        per_residue_median_edge=np.asarray(residue_edge_stats["median"], dtype=float),
        per_residue_q25_edge=np.asarray(residue_edge_stats["q25"], dtype=float),
        per_residue_q75_edge=np.asarray(residue_edge_stats["q75"], dtype=float),
        per_edge_mean=np.asarray(edge_stats["mean"], dtype=float),
        per_edge_std=np.asarray(edge_stats["std"], dtype=float),
        per_edge_median=np.asarray(edge_stats["median"], dtype=float),
        per_edge_q25=np.asarray(edge_stats["q25"], dtype=float),
        per_edge_q75=np.asarray(edge_stats["q75"], dtype=float),
        max_gap_h=np.asarray(prepared["max_gap_h"], dtype=float),
        max_gap_J=np.asarray(prepared["max_gap_J"], dtype=float),
        edges=np.asarray(prepared["edges"], dtype=np.int32),
        alpha=np.asarray([alpha], dtype=float),
        beta_node=np.asarray([float(prepared["beta_node"])], dtype=float),
        beta_edge=np.asarray([float(prepared["beta_edge"])], dtype=float),
    )

    summary = {
        "n_sample_frames": int(np.asarray(prepared["sample_inv"]).shape[0]),
        "n_md_frames": int(np.asarray(prepared["md_inv"]).shape[0]),
        "n_sample_unique": int(sample_unique.shape[0]),
        "n_md_unique": int(md_unique.shape[0]),
        "distance_mean": float(np.average(nn_dist_global, weights=weights)) if nn_dist_global.size else None,
        "distance_median": float(np.median(np.repeat(nn_dist_global, sample_counts.astype(int)))) if nn_dist_global.size and np.all(sample_counts < 10000) else float(np.median(nn_dist_global)) if nn_dist_global.size else None,
        "distance_min": float(np.min(nn_dist_global)) if nn_dist_global.size else None,
        "distance_max": float(np.max(nn_dist_global)) if nn_dist_global.size else None,
        "threshold_coverage": {str(float(thr)): float(val) for thr, val in zip(thresholds.tolist(), threshold_coverage.tolist())},
        "analysis_format_version": 2,
    }
    meta = {
        "analysis_id": str(prepared["analysis_id"]),
        "analysis_type": "potts_nn_mapping",
        "created_at": _utc_now(),
        "project_id": str(prepared["project_id"]),
        "system_id": str(prepared["system_id"]),
        "cluster_id": str(prepared["cluster_id"]),
        "model_id": prepared.get("model_id"),
        "model_name": prepared.get("model_name"),
        "sample_id": str(prepared["sample_id"]),
        "sample_name": str(prepared["sample_name"]),
        "sample_type": str(prepared["sample_type"]),
        "md_sample_id": str(prepared["md_sample_id"]),
        "md_sample_name": str(prepared["md_sample_name"]),
        "md_label_mode": str(prepared["md_label_mode"]),
        "drop_invalid": bool(prepared["drop_invalid"]),
        "use_unique": bool(prepared["use_unique"]),
        "normalize": bool(prepared["normalize"]),
        "compute_per_residue": bool(prepared["compute_per_residue"]),
        "alpha": float(prepared["alpha"]),
        "beta_node": float(prepared["beta_node"]),
        "beta_edge": float(prepared["beta_edge"]),
        "top_k_candidates": prepared.get("top_k_candidates"),
        "chunk_size": int(prepared["chunk_size"]),
        "distance_thresholds": [float(v) for v in thresholds.tolist()],
        "analysis_format_version": 2,
        "workers": int(max(1, workers_used)),
        "paths": {"analysis_npz": _relativize(npz_path, system_dir)},
        "summary": summary,
    }
    meta_path.write_text(json.dumps(_convert_nan_to_none(meta), indent=2), encoding="utf-8")
    result = {
        "metadata": _convert_nan_to_none(meta),
        "analysis_id": str(prepared["analysis_id"]),
        "analysis_dir": str(analysis_dir),
        "analysis_npz": str(npz_path),
    }
    atomic_pickle_dump(orchestration_paths(analysis_dir)["aggregate"], result)
    return result


def run_potts_nn_mapping_local(
    *,
    progress_callback: ProgressCallback = None,
    **kwargs: Any,
) -> dict[str, Any]:
    prepared = prepare_potts_nn_mapping_batch(**kwargs)
    payloads = prepared.get("payloads") or []
    workers = max(1, int(kwargs.get("n_workers") or prepared.get("requested_workers") or 1))
    out_rows = run_local_payload_batch(
        payloads,
        worker_fn=_potts_nn_row_worker,
        max_workers=workers,
        progress_callback=progress_callback,
        progress_label="Running Potts nearest-neighbor mapping",
    )
    workers_used = 1 if not payloads else max(1, min(workers, len(payloads)))
    return aggregate_potts_nn_mapping_batch(prepared, out_rows, workers_used=workers_used)


def wait_for_batch_marker(
    check_fn: Callable[[], tuple[bool, str | None]],
    *,
    poll_seconds: float = 1.0,
    timeout_seconds: float | None = None,
) -> None:
    start = time.monotonic()
    while True:
        done, error = check_fn()
        if done:
            if error:
                raise RuntimeError(error)
            return
        if timeout_seconds is not None and (time.monotonic() - start) > float(timeout_seconds):
            raise TimeoutError("Timed out while waiting for distributed ligand completion batch.")
        time.sleep(max(0.05, float(poll_seconds)))
