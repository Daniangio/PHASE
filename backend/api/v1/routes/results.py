import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, File, Form
from fastapi.responses import FileResponse

from backend.api.v1.common import DATA_ROOT, get_cluster_entry, project_store, stream_upload
from backend.services.metastable_clusters import (
    assign_cluster_labels_to_states,
    update_cluster_metadata_with_assignments,
)


router = APIRouter()


def _resolve_result_artifact_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = DATA_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Artifact path escapes data root.") from exc
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found.")
    return candidate


def _artifact_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix == ".html":
        return "text/html"
    return "application/octet-stream"


def _relativize_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_ROOT))
    except Exception:
        return str(path)


def _safe_filename(name: str | None, fallback: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return fallback
    base = Path(name).name.strip()
    return base or fallback


def _resolve_cluster_path(project_id: str, system_id: str, entry: dict) -> Path:
    rel_path = entry.get("path")
    if not rel_path:
        raise HTTPException(status_code=404, detail="Cluster NPZ path missing in system metadata.")
    cluster_path = Path(rel_path)
    if not cluster_path.is_absolute():
        cluster_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not cluster_path.exists():
        raise HTTPException(status_code=404, detail="Cluster NPZ file is missing on disk.")
    return cluster_path


def _load_assigned_labels(path: Path) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "assigned__labels" in data:
                return np.asarray(data["assigned__labels"], dtype=int)
    except Exception:
        return None
    return None


def _labels_for_cluster_frames(
    cluster_path: Path,
    *,
    assigned_state_paths: dict,
    base_dir: Path,
    n_residues: int,
) -> np.ndarray | None:
    with np.load(cluster_path, allow_pickle=True) as data:
        if "merged__frame_state_ids" not in data or "merged__frame_indices" not in data:
            return None
        state_ids = np.asarray(data["merged__frame_state_ids"]).astype(str)
        frame_indices = np.asarray(data["merged__frame_indices"]).astype(int)

    if state_ids.size == 0 or frame_indices.size == 0:
        return None

    labels_chunks: list[np.ndarray] = []
    for state_id in np.unique(state_ids):
        rel_path = assigned_state_paths.get(str(state_id))
        if not rel_path:
            continue
        path = Path(rel_path)
        if not path.is_absolute():
            path = base_dir / rel_path
        assigned = _load_assigned_labels(path)
        if assigned is None or assigned.size == 0:
            continue
        if assigned.shape[1] != n_residues:
            continue
        mask = state_ids == state_id
        frames = frame_indices[mask]
        valid = (frames >= 0) & (frames < assigned.shape[0])
        if not np.any(valid):
            continue
        labels_chunks.append(assigned[frames[valid]])

    if not labels_chunks:
        return None
    labels = np.concatenate(labels_chunks, axis=0)
    if labels.size == 0:
        return None
    keep = np.all(labels >= 0, axis=1)
    labels = labels[keep]
    if labels.size == 0:
        return None
    return labels


def _augment_sampling_summary(
    summary_path: Path,
    *,
    base_cluster_path: Path,
    base_cluster_id: str,
    base_cluster_name: str | None,
    compare_clusters: list[dict],
    model_path: Path,
    project_id: str,
    system_id: str,
) -> None:
    from phase.io.data import load_npz
    from phase.simulation.metrics import per_edge_js_from_padded, per_residue_js_from_padded, marginals, pairwise_joints_padded
    from phase.simulation.main import (
        _build_md_sources,
        _build_sample_sources,
        _pad_marginals_for_save,
        _compute_cross_likelihood_classification,
        _compute_energy_histograms,
        _compute_nn_cdfs,
    )
    from phase.simulation.potts_model import load_potts_model

    assignments = assign_cluster_labels_to_states(base_cluster_path, project_id, system_id)
    update_cluster_metadata_with_assignments(base_cluster_path, assignments)

    ds = load_npz(str(base_cluster_path))
    labels = ds.labels
    K = ds.cluster_counts
    edges = ds.edges

    with np.load(summary_path, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}

    if "K" in payload and payload["K"].size:
        summary_k = np.asarray(payload["K"], dtype=int)
        if summary_k.shape != K.shape or np.any(summary_k != K):
            raise HTTPException(status_code=400, detail="Uploaded summary does not match selected cluster NPZ.")
    if "edges" in payload and payload["edges"].size:
        summary_edges = np.asarray(payload["edges"], dtype=int)
        if summary_edges.shape != np.asarray(edges, dtype=int).shape:
            raise HTTPException(status_code=400, detail="Uploaded summary edges do not match selected cluster NPZ.")

    X_gibbs = np.asarray(payload.get("X_gibbs", np.zeros((0, len(K)), dtype=int)), dtype=int)
    X_sa = np.asarray(payload.get("X_sa", np.zeros((0, len(K)), dtype=int)), dtype=int)
    sa_labels_raw = payload.get("sa_schedule_labels")
    sa_schedule_labels = None
    if isinstance(sa_labels_raw, np.ndarray) and sa_labels_raw.size:
        sa_schedule_labels = [str(v) for v in sa_labels_raw.tolist()]

    beta = 1.0
    if "target_beta" in payload and payload["target_beta"].size:
        beta = float(np.asarray(payload["target_beta"])[0])
    gibbs_label = f"Gibbs β={beta:g}"
    sa_samples = [X_sa] if X_sa.size else []
    sample_sources = _build_sample_sources(
        X_gibbs,
        sa_samples,
        sa_schedule_labels,
        K,
        edges,
        gibbs_label=gibbs_label,
    )

    md_sources = _build_md_sources(
        labels,
        K,
        edges,
        ds.frame_state_ids,
        ds.frame_metastable_ids,
        ds.metadata,
        base_cluster_path,
    )
    meta = ds.metadata or {}
    assigned_state_paths = meta.get("assigned_state_paths") or {}
    base_dir = base_cluster_path.resolve().parent

    def _normalize_label(raw: str) -> str:
        label = str(raw or "").lower().strip()
        for prefix in ("macro:", "metastable:", "md cluster:"):
            if label.startswith(prefix):
                label = label[len(prefix) :].strip()
        return label

    existing_labels = {_normalize_label(src.get("label", "")) for src in md_sources}

    for info in compare_clusters:
        cid = str(info.get("cluster_id"))
        if not cid or cid == base_cluster_id:
            continue
        labels_other = _labels_for_cluster_frames(
            info["path"],
            assigned_state_paths=assigned_state_paths,
            base_dir=base_dir,
            n_residues=labels.shape[1],
        )
        if labels_other is None or labels_other.size == 0:
            continue
        if any(src.get("id") == f"cluster:{cid}" for src in md_sources):
            continue
        name = info.get("name") or cid
        if _normalize_label(name) in existing_labels:
            continue
        md_sources.append(
            {
                "id": f"cluster:{cid}",
                "label": f"MD cluster: {name}",
                "type": "cluster",
                "count": int(labels_other.shape[0]),
                "labels": labels_other,
                "p": _pad_marginals_for_save(marginals(labels_other, K)),
                "p2": pairwise_joints_padded(labels_other, K, edges),
            }
        )
        existing_labels.add(_normalize_label(name))

    md_source_ids = [src["id"] for src in md_sources]
    md_source_labels = [src["label"] for src in md_sources]
    md_source_types = [src["type"] for src in md_sources]
    md_source_counts = np.asarray([src["count"] for src in md_sources], dtype=int)
    p_md_by_source = np.stack([src["p"] for src in md_sources], axis=0) if md_sources else np.zeros((0, 0, 0), dtype=float)
    p2_md_by_source = (
        np.stack([src["p2"] for src in md_sources], axis=0) if md_sources else np.zeros((0, 0, 0, 0), dtype=float)
    )

    sample_source_ids = [src["id"] for src in sample_sources]
    sample_source_labels = [src["label"] for src in sample_sources]
    sample_source_types = [src["type"] for src in sample_sources]
    sample_source_counts = np.asarray([src["count"] for src in sample_sources], dtype=int)
    p_sample_by_source = (
        np.stack([src["p"] for src in sample_sources], axis=0) if sample_sources else np.zeros((0, 0, 0), dtype=float)
    )
    p2_sample_by_source = (
        np.stack([src["p2"] for src in sample_sources], axis=0) if sample_sources else np.zeros((0, 0, 0, 0), dtype=float)
    )

    js_md_sample = np.zeros((len(md_sources), len(sample_sources), len(K)), dtype=float)
    js2_md_sample = np.zeros((len(md_sources), len(sample_sources), len(edges)), dtype=float)
    for i, md in enumerate(md_sources):
        for j, sample in enumerate(sample_sources):
            js_md_sample[i, j] = per_residue_js_from_padded(md["p"], sample["p"], K)
            if len(edges) > 0:
                js2_md_sample[i, j] = per_edge_js_from_padded(md["p2"], sample["p2"], edges, K)

    gibbs_idx = next((i for i, src in enumerate(sample_sources) if src["id"] == "gibbs"), None)
    js_gibbs_sample = np.zeros((len(sample_sources), len(K)), dtype=float)
    js2_gibbs_sample = np.zeros((len(sample_sources), len(edges)), dtype=float)
    if gibbs_idx is not None:
        gibbs_p = sample_sources[gibbs_idx]["p"]
        gibbs_p2 = sample_sources[gibbs_idx]["p2"]
        for j, sample in enumerate(sample_sources):
            js_gibbs_sample[j] = per_residue_js_from_padded(gibbs_p, sample["p"], K)
            if len(edges) > 0:
                js2_gibbs_sample[j] = per_edge_js_from_padded(gibbs_p2, sample["p2"], edges, K)

    p2_md = md_sources[0]["p2"] if md_sources else np.zeros((0, 0, 0), dtype=float)
    p2_gibbs = sample_sources[gibbs_idx]["p2"] if gibbs_idx is not None else np.zeros((0, 0, 0), dtype=float)
    sa_idx = next((i for i, src in enumerate(sample_sources) if src["type"] == "sa"), None)
    p2_sa = sample_sources[sa_idx]["p2"] if sa_idx is not None else np.zeros((0, 0, 0), dtype=float)
    js2_gibbs = js2_md_sample[0, gibbs_idx] if md_sources and gibbs_idx is not None else np.array([], dtype=float)
    js2_sa = js2_md_sample[0, sa_idx] if md_sources and sa_idx is not None else np.array([], dtype=float)
    js2_sa_vs_gibbs = js2_gibbs_sample[sa_idx] if sa_idx is not None else np.array([], dtype=float)

    model = load_potts_model(model_path) if model_path.exists() else None
    edge_strength = np.array([], dtype=float)
    if model is not None and len(edges) > 0:
        strengths = []
        for r, s in edges:
            strengths.append(float(np.linalg.norm(model.coupling(int(r), int(s)))))
        edge_strength = np.asarray(strengths, dtype=float)

    energy_payload = _compute_energy_histograms(
        model=model,
        md_sources=md_sources,
        sample_sources=sample_sources,
        n_bins=40,
    )
    nn_payload = _compute_nn_cdfs(
        md_sources=md_sources,
        sample_sources=sample_sources,
        max_md=2000,
        max_sample=1000,
        block_size=256,
    )

    cross_likelihood = _compute_cross_likelihood_classification(
        labels,
        md_sources,
        ds.frame_state_ids,
        ds.frame_metastable_ids,
        ds.metadata,
        K,
        edges,
        batch_size=512,
    )

    payload.update(
        {
            "md_source_ids": np.asarray(md_source_ids, dtype=str),
            "md_source_labels": np.asarray(md_source_labels, dtype=str),
            "md_source_types": np.asarray(md_source_types, dtype=str),
            "md_source_counts": md_source_counts,
            "p_md_by_source": p_md_by_source,
            "p2_md_by_source": p2_md_by_source,
            "sample_source_ids": np.asarray(sample_source_ids, dtype=str),
            "sample_source_labels": np.asarray(sample_source_labels, dtype=str),
            "sample_source_types": np.asarray(sample_source_types, dtype=str),
            "sample_source_counts": sample_source_counts,
            "p_sample_by_source": p_sample_by_source,
            "p2_sample_by_source": p2_sample_by_source,
            "js_md_sample": js_md_sample,
            "js2_md_sample": js2_md_sample,
            "js_gibbs_sample": js_gibbs_sample,
            "js2_gibbs_sample": js2_gibbs_sample,
            "p2_md": p2_md,
            "p2_gibbs": p2_gibbs,
            "p2_sa": p2_sa,
            "js2_gibbs": js2_gibbs,
            "js2_sa": js2_sa,
            "js2_sa_vs_gibbs": js2_sa_vs_gibbs,
            "energy_bins": energy_payload["bins"],
            "energy_hist_md": energy_payload["hist_md"],
            "energy_cdf_md": energy_payload["cdf_md"],
            "energy_hist_sample": energy_payload["hist_sample"],
            "energy_cdf_sample": energy_payload["cdf_sample"],
            "nn_bins": nn_payload["bins"],
            "nn_cdf_sample_to_md": nn_payload["cdf_sample_to_md"],
            "nn_cdf_md_to_sample": nn_payload["cdf_md_to_sample"],
            "edge_strength": edge_strength,
            "xlik_delta_active": cross_likelihood["delta_active"] if cross_likelihood else np.array([], dtype=float),
            "xlik_delta_inactive": cross_likelihood["delta_inactive"] if cross_likelihood else np.array([], dtype=float),
            "xlik_auc": np.array([cross_likelihood["auc"]], dtype=float) if cross_likelihood else np.array([], dtype=float),
            "xlik_active_state_ids": np.asarray(
                cross_likelihood["active_state_ids"], dtype=str
            )
            if cross_likelihood
            else np.array([], dtype=str),
            "xlik_inactive_state_ids": np.asarray(
                cross_likelihood["inactive_state_ids"], dtype=str
            )
            if cross_likelihood
            else np.array([], dtype=str),
            "xlik_active_state_labels": np.asarray(
                cross_likelihood["active_state_labels"], dtype=str
            )
            if cross_likelihood
            else np.array([], dtype=str),
            "xlik_inactive_state_labels": np.asarray(
                cross_likelihood["inactive_state_labels"], dtype=str
            )
            if cross_likelihood
            else np.array([], dtype=str),
            "xlik_delta_fit_by_other": cross_likelihood["delta_fit_by_other"]
            if cross_likelihood
            else np.zeros((0, 0), dtype=float),
            "xlik_delta_other_by_other": cross_likelihood["delta_other_by_other"]
            if cross_likelihood
            else np.zeros((0, 0), dtype=float),
            "xlik_delta_other_counts": cross_likelihood["delta_other_counts"]
            if cross_likelihood
            else np.array([], dtype=int),
            "xlik_other_state_ids": np.asarray(cross_likelihood["other_state_ids"], dtype=str)
            if cross_likelihood
            else np.array([], dtype=str),
            "xlik_other_state_labels": np.asarray(cross_likelihood["other_state_labels"], dtype=str)
            if cross_likelihood
            else np.array([], dtype=str),
            "xlik_auc_by_other": np.asarray(cross_likelihood["auc_by_other"], dtype=float)
            if cross_likelihood
            else np.array([], dtype=float),
            "xlik_roc_fpr_by_other": np.asarray(cross_likelihood["roc_fpr_by_other"], dtype=float)
            if cross_likelihood
            else np.zeros((0, 0), dtype=float),
            "xlik_roc_tpr_by_other": np.asarray(cross_likelihood["roc_tpr_by_other"], dtype=float)
            if cross_likelihood
            else np.zeros((0, 0), dtype=float),
            "xlik_roc_counts": np.asarray(cross_likelihood["roc_counts"], dtype=int)
            if cross_likelihood
            else np.array([], dtype=int),
            "xlik_score_fit_by_other": np.asarray(cross_likelihood["score_fit_by_other"], dtype=float)
            if cross_likelihood
            else np.zeros((0, 0), dtype=float),
            "xlik_score_other_by_other": np.asarray(cross_likelihood["score_other_by_other"], dtype=float)
            if cross_likelihood
            else np.zeros((0, 0), dtype=float),
            "xlik_score_other_counts": np.asarray(cross_likelihood["score_other_counts"], dtype=int)
            if cross_likelihood
            else np.array([], dtype=int),
            "xlik_auc_score_by_other": np.asarray(cross_likelihood["auc_score_by_other"], dtype=float)
            if cross_likelihood
            else np.array([], dtype=float),
            "xlik_score_roc_fpr_by_other": np.asarray(cross_likelihood["score_roc_fpr_by_other"], dtype=float)
            if cross_likelihood
            else np.zeros((0, 0), dtype=float),
            "xlik_score_roc_tpr_by_other": np.asarray(cross_likelihood["score_roc_tpr_by_other"], dtype=float)
            if cross_likelihood
            else np.zeros((0, 0), dtype=float),
            "xlik_score_roc_counts": np.asarray(cross_likelihood["score_roc_counts"], dtype=int)
            if cross_likelihood
            else np.array([], dtype=int),
        }
    )

    np.savez_compressed(summary_path, **payload)


def _iter_result_files() -> list[Path]:
    results: list[Path] = []
    base_dir = project_store.base_dir
    if not base_dir.exists():
        return results
    for project_dir in sorted(base_dir.glob("*")):
        systems_dir = project_dir / "systems"
        if not systems_dir.exists():
            continue
        for system_dir in sorted(systems_dir.glob("*")):
            jobs_dir = system_dir / "results" / "jobs"
            if not jobs_dir.exists():
                continue
            results.extend(jobs_dir.glob("*.json"))
    return results


def _find_result_file(job_uuid: str) -> Path | None:
    base_dir = project_store.base_dir
    if not base_dir.exists():
        return None
    for project_dir in base_dir.glob("*"):
        systems_dir = project_dir / "systems"
        if not systems_dir.exists():
            continue
        for system_dir in systems_dir.glob("*"):
            candidate = system_dir / "results" / "jobs" / f"{job_uuid}.json"
            if candidate.exists():
                return candidate
    return None


def _remove_results_dir(path_value: str | None, *, system_dir: Path | None = None) -> None:
    if not isinstance(path_value, str) or not path_value:
        return
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = DATA_ROOT / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(DATA_ROOT)
    except ValueError:
        return
    if system_dir is not None:
        results_root = (system_dir / "results").resolve()
        try:
            candidate.relative_to(results_root)
        except ValueError:
            return
    if candidate.exists() and candidate.is_dir():
        shutil.rmtree(candidate, ignore_errors=True)


def _cleanup_empty_result_dirs() -> int:
    removed = 0
    base_dir = project_store.base_dir
    if not base_dir.exists():
        return removed
    for project_dir in base_dir.glob("*"):
        systems_dir = project_dir / "systems"
        if not systems_dir.exists():
            continue
        for system_dir in systems_dir.glob("*"):
            results_dir = system_dir / "results"
            jobs_dir = results_dir / "jobs"
            if jobs_dir.exists():
                try:
                    if not any(jobs_dir.iterdir()):
                        jobs_dir.rmdir()
                        removed += 1
                except Exception:
                    pass
            if results_dir.exists():
                try:
                    if not any(results_dir.iterdir()):
                        results_dir.rmdir()
                        removed += 1
                except Exception:
                    pass
    return removed


def _cleanup_tmp_artifacts(tmp_root: Path) -> int:
    removed = 0
    if not tmp_root.exists():
        return removed
    for path in tmp_root.rglob("__pycache__"):
        if not path.is_dir():
            continue
        try:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        except Exception:
            continue
    for path in tmp_root.rglob("_remote_module_non_scriptable.py"):
        if not path.is_file():
            continue
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue
    for path in sorted(tmp_root.rglob("*"), reverse=True):
        if not path.is_dir():
            continue
        try:
            if any(path.iterdir()):
                continue
            path.rmdir()
        except Exception:
            continue
    return removed


@router.get("/results", summary="List all available analysis results")
async def get_results_list():
    """
    Fetches the metadata for all jobs (finished, running, or failed)
    by reading the JSON files from the persistent results directory.
    """
    results_list = []
    try:
        sorted_files = sorted(_iter_result_files(), key=lambda f: f.stat().st_mtime, reverse=True)

        for result_file in sorted_files:
            try:
                with open(result_file, "r") as handle:
                    data = json.load(handle)
                system_ref = data.get("system_reference") or {}
                state_ref = system_ref.get("states") or {}
                results_payload = data.get("results") or {}
                results_list.append(
                    {
                        "job_id": data.get("job_id"),
                        "rq_job_id": data.get("rq_job_id"),
                        "analysis_type": data.get("analysis_type"),
                        "status": data.get("status"),
                        "created_at": data.get("created_at"),
                        "completed_at": data.get("completed_at"),
                        "error": data.get("error"),
                        "project_id": system_ref.get("project_id"),
                        "project_name": system_ref.get("project_name"),
                        "system_id": system_ref.get("system_id"),
                        "system_name": system_ref.get("system_name"),
                        "cluster_id": system_ref.get("cluster_id"),
                        "cluster_name": system_ref.get("cluster_name"),
                        "cluster_npz": results_payload.get("cluster_npz"),
                        "potts_model": results_payload.get("potts_model"),
                        "state_a_id": state_ref.get("state_a", {}).get("id"),
                        "state_a_name": state_ref.get("state_a", {}).get("name"),
                        "state_b_id": state_ref.get("state_b", {}).get("id"),
                        "state_b_name": state_ref.get("state_b", {}).get("name"),
                        "structures": system_ref.get("structures"),
                    }
                )
            except Exception as exc:
                print(f"Failed to read result file: {result_file}. Error: {exc}")

        return results_list
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list results: {exc}")


@router.get("/results/{job_uuid}", summary="Get the full JSON data for a specific result")
async def get_result_detail(job_uuid: str):
    """
    Fetches the complete, persisted JSON data for a single analysis job
    using its unique job_uuid.
    """
    try:
        result_file = _find_result_file(job_uuid)
        if not result_file or not result_file.exists() or not result_file.is_file():
            raise HTTPException(status_code=404, detail=f"Result file for job '{job_uuid}' not found.")

        return Response(
            content=result_file.read_text(),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={result_file.name}"},
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=f"Failed to read result: {exc}")


@router.get("/results/{job_uuid}/artifacts/{artifact}", summary="Download a result artifact")
async def download_result_artifact(job_uuid: str, artifact: str, download: bool = Query(False)):
    """
    Download stored analysis artifacts by name (summary_npz, metadata_json, marginals_plot, beta_scan_plot, potts_model).
    """
    result_file = _find_result_file(job_uuid)
    if not result_file or not result_file.exists() or not result_file.is_file():
        raise HTTPException(status_code=404, detail=f"Result file for job '{job_uuid}' not found.")

    try:
        payload = json.loads(result_file.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to read result payload.") from exc

    results = payload.get("results") or {}
    allowed = {
        "summary_npz": "summary_npz",
        "metadata_json": "metadata_json",
        "marginals_plot": "marginals_plot",
        "sampling_report": "sampling_report",
        "cross_likelihood_report": "cross_likelihood_report",
        "beta_scan_plot": "beta_scan_plot",
        "cluster_npz": "cluster_npz",
        "potts_model": "potts_model",
    }
    key = allowed.get(artifact)
    if not key:
        raise HTTPException(status_code=404, detail="Unknown artifact.")

    path_value = results.get(key)
    if (not isinstance(path_value, str) or not path_value) and artifact in ("sampling_report", "cross_likelihood_report"):
        summary_value = results.get("summary_npz")
        if not isinstance(summary_value, str) or not summary_value:
            report_name = "sampling report" if artifact == "sampling_report" else "cross-likelihood report"
            raise HTTPException(status_code=404, detail=f"{report_name} unavailable (missing summary NPZ).")
        summary_path = _resolve_result_artifact_path(summary_value)
        if artifact == "sampling_report":
            report_path = summary_path.parent / "sampling_report.html"
        else:
            report_path = summary_path.parent / "cross_likelihood_report.html"
        try:
            if artifact == "sampling_report":
                from phase.simulation.plotting import plot_sampling_report_from_npz
                plot_sampling_report_from_npz(summary_path=summary_path, out_path=report_path)
            else:
                from phase.simulation.plotting import plot_cross_likelihood_report_from_npz
                plot_cross_likelihood_report_from_npz(summary_path=summary_path, out_path=report_path)
        except Exception as exc:
            report_name = "sampling report" if artifact == "sampling_report" else "cross-likelihood report"
            raise HTTPException(status_code=500, detail=f"Failed to generate {report_name}: {exc}") from exc
        try:
            rel_path = str(report_path.relative_to(DATA_ROOT))
        except ValueError:
            rel_path = str(report_path)
        results[key] = rel_path
        payload["results"] = results
        try:
            result_file.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass
        path_value = rel_path

    if not isinstance(path_value, str) or not path_value:
        raise HTTPException(status_code=404, detail="Artifact not available for this job.")

    artifact_path = _resolve_result_artifact_path(path_value)
    media_type = _artifact_media_type(artifact_path)
    headers = {}
    if media_type == "text/html" and not download:
        headers["Content-Disposition"] = f"inline; filename={artifact_path.name}"
        filename = None
    else:
        filename = artifact_path.name
    return FileResponse(artifact_path, filename=filename, media_type=media_type, headers=headers)


@router.post("/results/simulation/upload", summary="Upload a local Potts sampling result")
async def upload_simulation_result(
    project_id: str = Form(...),
    system_id: str = Form(...),
    cluster_id: str = Form(...),
    compare_cluster_ids: list[str] = Form(default=[]),
    summary_npz: UploadFile = File(...),
    potts_model: UploadFile = File(...),
):
    try:
        system_meta = project_store.get_system(project_id, system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{system_id}' not found in project '{project_id}'.",
        )

    entry = get_cluster_entry(system_meta, cluster_id)
    cluster_path = _resolve_cluster_path(project_id, system_id, entry)

    if summary_npz.filename and not summary_npz.filename.lower().endswith(".npz"):
        raise HTTPException(status_code=400, detail="Summary upload must be an .npz file.")
    if potts_model.filename and not potts_model.filename.lower().endswith(".npz"):
        raise HTTPException(status_code=400, detail="Potts model upload must be an .npz file.")

    job_uuid = str(uuid.uuid4())
    sample_id = str(uuid.uuid4())
    dirs = project_store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = dirs["system_dir"]
    sample_dir = dirs["samples_dir"] / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    results_dir = sample_dir

    summary_path = results_dir / "run_summary.npz"
    await stream_upload(summary_npz, summary_path)

    model_filename = _safe_filename(potts_model.filename, f"{cluster_id}_potts_model.npz")
    if not model_filename.lower().endswith(".npz"):
        model_filename = f"{model_filename}.npz"
    model_id = str(uuid.uuid4())
    model_dir = dirs["potts_models_dir"] / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_filename
    await stream_upload(potts_model, model_path)
    model_rel = str(model_path.relative_to(system_dir))
    models = entry.get("potts_models")
    if not isinstance(models, list):
        models = []
    models.append(
        {
            "model_id": model_id,
            "name": Path(model_filename).stem,
            "path": model_rel,
            "created_at": datetime.utcnow().isoformat(),
            "source": "upload",
            "params": {},
        }
    )
    entry["potts_models"] = models
    sample_paths = {"summary_npz": str(summary_path.relative_to(system_dir))}

    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []
    samples.append(
        {
            "sample_id": sample_id,
            "name": f"Sampling {datetime.utcnow().strftime('%Y%m%d %H:%M')}",
            "type": "potts_sampling",
            "method": "upload",
            "model_id": model_id,
            "created_at": datetime.utcnow().isoformat(),
            "paths": sample_paths,
        }
    )
    entry["samples"] = samples
    project_store.save_system(system_meta)

    compare_ids = []
    for cid in compare_cluster_ids or []:
        cid = str(cid).strip()
        if not cid or cid == cluster_id or cid in compare_ids:
            continue
        compare_ids.append(cid)
    compare_clusters = []
    for cid in compare_ids:
        other_entry = get_cluster_entry(system_meta, cid)
        compare_clusters.append(
            {
                "cluster_id": cid,
                "name": other_entry.get("name"),
                "path": _resolve_cluster_path(project_id, system_id, other_entry),
            }
        )

    try:
        _augment_sampling_summary(
            summary_path,
            base_cluster_path=cluster_path,
            base_cluster_id=cluster_id,
            base_cluster_name=entry.get("name"),
            compare_clusters=compare_clusters,
            model_path=model_path,
            project_id=project_id,
            system_id=system_id,
        )

        from phase.simulation.plotting import (
            plot_beta_scan_curve,
            plot_cross_likelihood_report_from_npz,
            plot_marginal_summary_from_npz,
            plot_sampling_report_from_npz,
        )

        plot_path = plot_marginal_summary_from_npz(
            summary_path=summary_path,
            out_path=results_dir / "marginals.html",
            annotate=False,
        )
        report_path = plot_sampling_report_from_npz(
            summary_path=summary_path,
            out_path=results_dir / "sampling_report.html",
        )
        cross_likelihood_report_path = plot_cross_likelihood_report_from_npz(
            summary_path=summary_path,
            out_path=results_dir / "cross_likelihood_report.html",
        )
        beta_scan_path = None
        beta_eff_value = None
        with np.load(summary_path, allow_pickle=False) as data:
            beta_eff = data["beta_eff"] if "beta_eff" in data else np.array([])
            if beta_eff.size:
                beta_eff_value = float(beta_eff[0])
            grid = data["beta_eff_grid"] if "beta_eff_grid" in data else np.array([])
            distances = data["beta_eff_distances_by_schedule"] if "beta_eff_distances_by_schedule" in data else np.array([])
            if grid.size and distances.size:
                labels = data["sa_schedule_labels"] if "sa_schedule_labels" in data else None
                if labels is not None:
                    labels = [str(v) for v in labels.tolist()]
                beta_scan_path = plot_beta_scan_curve(
                    betas=grid,
                    distances=distances,
                    labels=labels,
                    out_path=results_dir / "beta_scan.html",
                )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate sampling plots: {exc}") from exc

    try:
        project_meta = project_store.get_project(project_id)
        project_name = project_meta.name
    except Exception:
        project_name = None

    cluster_name = entry.get("name") if isinstance(entry, dict) else None
    result_payload = {
        "job_id": job_uuid,
        "rq_job_id": f"upload-{job_uuid}",
        "analysis_type": "simulation",
        "status": "finished",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": datetime.utcnow().isoformat(),
        "params": {
            "source": "upload",
            "compare_cluster_ids": compare_ids,
        },
        "results": {
            "results_dir": _relativize_path(results_dir),
            "summary_npz": _relativize_path(summary_path),
            "metadata_json": None,
            "marginals_plot": _relativize_path(plot_path) if plot_path else None,
            "sampling_report": _relativize_path(report_path) if report_path else None,
            "cross_likelihood_report": _relativize_path(cross_likelihood_report_path) if cross_likelihood_report_path else None,
            "beta_scan_plot": _relativize_path(beta_scan_path) if beta_scan_path else None,
            "cluster_npz": _relativize_path(cluster_path),
            "potts_model": _relativize_path(model_path),
            "beta_eff": beta_eff_value,
        },
        "system_reference": {
            "project_id": project_id,
            "system_id": system_id,
            "project_name": project_name,
            "system_name": system_meta.name,
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "structures": {},
            "states": {},
        },
        "error": None,
    }

    result_file = _find_result_file(job_uuid)
    if not result_file or not result_file.exists():
        raise HTTPException(status_code=404, detail=f"No data found for job UUID '{job_uuid}'.")
    try:
        result_file.write_text(json.dumps(result_payload, indent=2))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write result metadata: {exc}") from exc

    return {"status": "uploaded", "job_id": job_uuid}


@router.delete("/results/{job_uuid}", summary="Delete a job and its associated data")
async def delete_result(job_uuid: str):
    """
    Deletes a job's persisted JSON file.
    """
    results_dirs = project_store.ensure_results_directories(project_id, system_id)
    result_file = results_dirs["jobs_dir"] / f"{job_uuid}.json"

    try:
        results_dir_value = None
        system_dir = None
        try:
            payload = json.loads(result_file.read_text())
            results_payload = payload.get("results") or {}
            results_dir_value = results_payload.get("results_dir")
            system_ref = payload.get("system_reference") or {}
            project_id = system_ref.get("project_id")
            system_id = system_ref.get("system_id")
            if project_id and system_id:
                system_dir = project_store.resolve_path(project_id, system_id, "")
        except Exception:
            results_dir_value = None
        _remove_results_dir(results_dir_value, system_dir=system_dir)
        result_file.unlink()
        _cleanup_empty_result_dirs()
        return {"status": "deleted", "job_id": job_uuid}
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=f"Failed to delete job data: {str(exc)}")


@router.post("/results/cleanup", summary="Cleanup empty result folders and tmp artifacts")
async def cleanup_results(include_tmp: bool = Query(True)):
    """
    Remove empty job result folders and stale tmp artifacts.
    """
    empty_removed = _cleanup_empty_result_dirs()
    tmp_removed = 0
    tmp_root_value = None
    if include_tmp:
        tmp_root = Path(os.getenv("TMPDIR") or (DATA_ROOT / "tmp")).resolve()
        try:
            tmp_root.relative_to(DATA_ROOT)
        except ValueError:
            tmp_root = None
        if tmp_root and tmp_root.exists():
            tmp_root_value = str(tmp_root)
            tmp_removed = _cleanup_tmp_artifacts(tmp_root)
    return {
        "empty_result_dirs_removed": empty_removed,
        "tmp_artifacts_removed": tmp_removed,
        "tmp_root": tmp_root_value,
    }
