import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

from backend.api.v1.common import DATA_ROOT, RESULTS_DIR


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
    Download stored analysis artifacts by name (summary_npz, metadata_json, marginals_plot, beta_scan_plot).
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
        "beta_scan_plot": "beta_scan_plot",
        "cluster_npz": "cluster_npz",
    }
    key = allowed.get(artifact)
    if not key:
        raise HTTPException(status_code=404, detail="Unknown artifact.")
    path_value = results.get(key)
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


@router.delete("/results/{job_uuid}", summary="Delete a job and its associated data")
async def delete_result(job_uuid: str):
    """
    Deletes a job's persisted JSON file.
    """
    result_file = RESULTS_DIR / f"{job_uuid}.json"

    try:
        if not result_file.exists():
            raise HTTPException(status_code=404, detail=f"No data found for job UUID '{job_uuid}'.")
        result_file.unlink()
        return {"status": "deleted", "job_id": job_uuid}
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=500, detail=f"Failed to delete job data: {str(exc)}")
