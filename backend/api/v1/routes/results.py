import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, File, Form
from fastapi.responses import FileResponse

from backend.api.v1.common import DATA_ROOT, RESULTS_DIR, get_cluster_entry, project_store, stream_upload


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


def _remove_results_dir(path_value: str | None) -> None:
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
    if candidate.exists() and candidate.is_dir():
        shutil.rmtree(candidate, ignore_errors=True)


def _cleanup_empty_simulation_dirs() -> int:
    removed = 0
    sim_root = RESULTS_DIR / "simulation"
    if not sim_root.exists():
        return removed
    for entry in sim_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            if any(entry.iterdir()):
                continue
        except PermissionError:
            continue
        try:
            entry.rmdir()
            removed += 1
        except Exception:
            continue
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
        sorted_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

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
        result_file = RESULTS_DIR / f"{job_uuid}.json"
        if not result_file.exists() or not result_file.is_file():
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
    result_file = RESULTS_DIR / f"{job_uuid}.json"
    if not result_file.exists() or not result_file.is_file():
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
        "beta_scan_plot": "beta_scan_plot",
        "cluster_npz": "cluster_npz",
        "potts_model": "potts_model",
    }
    key = allowed.get(artifact)
    if not key:
        raise HTTPException(status_code=404, detail="Unknown artifact.")

    path_value = results.get(key)
    if (not isinstance(path_value, str) or not path_value) and artifact == "sampling_report":
        summary_value = results.get("summary_npz")
        if not isinstance(summary_value, str) or not summary_value:
            raise HTTPException(status_code=404, detail="Sampling report unavailable (missing summary NPZ).")
        summary_path = _resolve_result_artifact_path(summary_value)
        report_path = summary_path.parent / "sampling_report.html"
        try:
            from phase.simulation.plotting import plot_sampling_report_from_npz
            plot_sampling_report_from_npz(summary_path=summary_path, out_path=report_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to generate sampling report: {exc}") from exc
        try:
            rel_path = str(report_path.relative_to(DATA_ROOT))
        except ValueError:
            rel_path = str(report_path)
        results["sampling_report"] = rel_path
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
    rel_path = entry.get("path")
    if not rel_path:
        raise HTTPException(status_code=404, detail="Cluster NPZ path missing in system metadata.")
    cluster_path = Path(rel_path)
    if not cluster_path.is_absolute():
        cluster_path = project_store.resolve_path(project_id, system_id, rel_path)
    if not cluster_path.exists():
        raise HTTPException(status_code=404, detail="Cluster NPZ file is missing on disk.")

    if summary_npz.filename and not summary_npz.filename.lower().endswith(".npz"):
        raise HTTPException(status_code=400, detail="Summary upload must be an .npz file.")
    if potts_model.filename and not potts_model.filename.lower().endswith(".npz"):
        raise HTTPException(status_code=400, detail="Potts model upload must be an .npz file.")

    job_uuid = str(uuid.uuid4())
    results_dir = RESULTS_DIR / "simulation" / job_uuid
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "run_summary.npz"
    await stream_upload(summary_npz, summary_path)

    dirs = project_store.ensure_directories(project_id, system_id)
    system_dir = dirs["system_dir"]
    model_filename = _safe_filename(potts_model.filename, f"{cluster_id}_potts_model.npz")
    if not model_filename.lower().endswith(".npz"):
        model_filename = f"{model_filename}.npz"
    model_path = dirs["potts_models_dir"] / model_filename
    await stream_upload(potts_model, model_path)
    model_rel = str(model_path.relative_to(system_dir))
    entry["potts_model_path"] = model_rel
    entry["potts_model_updated_at"] = datetime.utcnow().isoformat()
    entry["potts_model_source"] = "upload"
    entry["potts_model_name"] = Path(model_filename).stem
    project_store.save_system(system_meta)

    try:
        from phase.simulation.plotting import plot_beta_scan_curve, plot_marginal_summary_from_npz, plot_sampling_report_from_npz

        plot_path = plot_marginal_summary_from_npz(
            summary_path=summary_path,
            out_path=results_dir / "marginals.html",
            annotate=False,
        )
        report_path = plot_sampling_report_from_npz(
            summary_path=summary_path,
            out_path=results_dir / "sampling_report.html",
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
        },
        "results": {
            "results_dir": _relativize_path(results_dir),
            "summary_npz": _relativize_path(summary_path),
            "metadata_json": None,
            "marginals_plot": _relativize_path(plot_path) if plot_path else None,
            "sampling_report": _relativize_path(report_path) if report_path else None,
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

    result_file = RESULTS_DIR / f"{job_uuid}.json"
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
    result_file = RESULTS_DIR / f"{job_uuid}.json"

    try:
        if not result_file.exists():
            raise HTTPException(status_code=404, detail=f"No data found for job UUID '{job_uuid}'.")
        results_dir_value = None
        try:
            payload = json.loads(result_file.read_text())
            results_payload = payload.get("results") or {}
            results_dir_value = results_payload.get("results_dir")
        except Exception:
            results_dir_value = None
        if not results_dir_value:
            fallback_dir = RESULTS_DIR / "simulation" / job_uuid
            if fallback_dir.exists():
                results_dir_value = str(fallback_dir)
        _remove_results_dir(results_dir_value)
        result_file.unlink()
        _cleanup_empty_simulation_dirs()
        return {"status": "deleted", "job_id": job_uuid}
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=f"Failed to delete job data: {str(exc)}")


@router.post("/results/cleanup", summary="Cleanup empty result folders and tmp artifacts")
async def cleanup_results(include_tmp: bool = Query(True)):
    """
    Remove empty simulation result folders and stale tmp artifacts.
    """
    empty_removed = _cleanup_empty_simulation_dirs()
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
        "empty_simulation_dirs_removed": empty_removed,
        "tmp_artifacts_removed": tmp_removed,
        "tmp_root": tmp_root_value,
    }
