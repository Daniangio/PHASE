import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from backend.api.v1.common import ensure_system_ready, get_cluster_entry, get_queue, project_store
from backend.api.v1.schemas import PottsFitJobRequest, SimulationJobRequest, StaticJobRequest
from backend.tasks import run_analysis_job, run_potts_fit_job, run_simulation_job


router = APIRouter()


def _submit_job(
    analysis_type: str,
    project_id: str,
    system_id: str,
    state_a_id: str,
    state_b_id: str,
    params: Dict[str, Any],
    task_queue: Any,
):
    """Helper to enqueue a job backed by a preprocessed system."""
    system_meta, state_a, state_b = ensure_system_ready(project_id, system_id, state_a_id, state_b_id)
    job_uuid = str(uuid.uuid4())
    try:
        project_meta = project_store.get_project(project_id)
        project_name = project_meta.name
    except Exception:
        project_name = None

    dataset_ref = {
        "project_id": project_id,
        "project_name": project_name,
        "system_id": system_id,
        "system_name": system_meta.name,
        "state_a_id": state_a.state_id,
        "state_b_id": state_b.state_id,
        "state_a_name": state_a.name,
        "state_b_name": state_b.name,
    }

    try:
        job = task_queue.enqueue(
            run_analysis_job,
            args=(
                job_uuid,
                analysis_type,
                dataset_ref,
                params,
            ),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"analysis-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post("/submit/static", summary="Submit a Static Reporters analysis")
async def submit_static_job(
    payload: StaticJobRequest,
    task_queue: Any = Depends(get_queue),
):
    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "state_a_id", "state_b_id"})
    return _submit_job(
        "static",
        payload.project_id,
        payload.system_id,
        payload.state_a_id,
        payload.state_b_id,
        params,
        task_queue,
    )


@router.post("/submit/simulation", summary="Submit a Potts sampling simulation")
async def submit_simulation_job(
    payload: SimulationJobRequest,
    task_queue: Any = Depends(get_queue),
):
    try:
        system_meta = project_store.get_system(payload.project_id, payload.system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{payload.system_id}' not found in project '{payload.project_id}'.",
        )

    get_cluster_entry(system_meta, payload.cluster_id)

    sampling_method = (payload.sampling_method or "gibbs").lower()
    if sampling_method not in {"gibbs", "sa"}:
        raise HTTPException(status_code=400, detail="sampling_method must be 'gibbs' or 'sa'.")

    rex_betas = payload.rex_betas
    if isinstance(rex_betas, str) and not rex_betas.strip():
        rex_betas = None
    if isinstance(rex_betas, list) and len(rex_betas) == 0:
        rex_betas = None

    if sampling_method == "gibbs":
        if rex_betas is None:
            rex_params = [payload.rex_beta_min, payload.rex_beta_max, payload.rex_spacing]
            if any(val is not None for val in rex_params) and not all(val is not None for val in rex_params):
                raise HTTPException(
                    status_code=400,
                    detail="Provide rex_beta_min, rex_beta_max, rex_spacing together or rex_betas.",
                )

    if sampling_method == "gibbs" and payload.rex_spacing is not None and payload.rex_spacing not in {"geom", "lin"}:
        raise HTTPException(status_code=400, detail="rex_spacing must be 'geom' or 'lin'.")

    for name, value in {
        "rex_samples": payload.rex_samples,
        "rex_burnin": payload.rex_burnin,
        "rex_thin": payload.rex_thin,
        "sa_reads": payload.sa_reads,
        "sa_sweeps": payload.sa_sweeps,
        "plm_epochs": payload.plm_epochs,
        "plm_batch_size": payload.plm_batch_size,
        "plm_progress_every": payload.plm_progress_every,
    }.items():
        if value is not None and int(value) < 1:
            raise HTTPException(status_code=400, detail=f"{name} must be >= 1.")

    if payload.plm_lr is not None and float(payload.plm_lr) <= 0:
        raise HTTPException(status_code=400, detail="plm_lr must be > 0.")
    if payload.plm_lr_min is not None and float(payload.plm_lr_min) < 0:
        raise HTTPException(status_code=400, detail="plm_lr_min must be >= 0.")
    if payload.plm_l2 is not None and float(payload.plm_l2) < 0:
        raise HTTPException(status_code=400, detail="plm_l2 must be >= 0.")
    if payload.plm_lr_schedule is not None and payload.plm_lr_schedule not in {"cosine", "none"}:
        raise HTTPException(status_code=400, detail="plm_lr_schedule must be 'cosine' or 'none'.")

    if payload.contact_cutoff is not None and float(payload.contact_cutoff) <= 0:
        raise HTTPException(status_code=400, detail="contact_cutoff must be > 0.")
    if payload.contact_atom_mode is not None:
        mode = str(payload.contact_atom_mode).upper()
        if mode not in {"CA", "CM"}:
            raise HTTPException(status_code=400, detail="contact_atom_mode must be 'CA' or 'CM'.")
    if payload.sa_beta_hot is not None and float(payload.sa_beta_hot) <= 0:
        raise HTTPException(status_code=400, detail="sa_beta_hot must be > 0.")
    if payload.sa_beta_cold is not None and float(payload.sa_beta_cold) <= 0:
        raise HTTPException(status_code=400, detail="sa_beta_cold must be > 0.")
    if (payload.sa_beta_hot is None) != (payload.sa_beta_cold is None):
        raise HTTPException(status_code=400, detail="Provide both sa_beta_hot and sa_beta_cold, or neither.")
    if payload.sa_beta_hot is not None and payload.sa_beta_cold is not None:
        if float(payload.sa_beta_hot) > float(payload.sa_beta_cold):
            raise HTTPException(status_code=400, detail="sa_beta_hot must be <= sa_beta_cold.")
    if payload.sa_beta_schedules:
        for idx, schedule in enumerate(payload.sa_beta_schedules):
            if schedule is None or len(schedule) != 2:
                raise HTTPException(status_code=400, detail=f"sa_beta_schedules[{idx}] must be a (hot, cold) pair.")
            hot, cold = schedule
            if float(hot) <= 0 or float(cold) <= 0:
                raise HTTPException(status_code=400, detail=f"sa_beta_schedules[{idx}] values must be > 0.")
            if float(hot) > float(cold):
                raise HTTPException(status_code=400, detail=f"sa_beta_schedules[{idx}] must satisfy hot <= cold.")

    try:
        project_meta = project_store.get_project(payload.project_id)
        project_name = project_meta.name
    except Exception:
        project_name = None

    dataset_ref = {
        "project_id": payload.project_id,
        "project_name": project_name,
        "system_id": payload.system_id,
        "system_name": system_meta.name,
        "cluster_id": payload.cluster_id,
    }

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_simulation_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"simulation-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post("/submit/potts_fit", summary="Submit a Potts model fitting job")
async def submit_potts_fit_job(
    payload: PottsFitJobRequest,
    task_queue: Any = Depends(get_queue),
):
    try:
        system_meta = project_store.get_system(payload.project_id, payload.system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{payload.system_id}' not found in project '{payload.project_id}'.",
        )

    get_cluster_entry(system_meta, payload.cluster_id)

    if payload.fit_method is not None and payload.fit_method not in {"pmi", "plm", "pmi+plm"}:
        raise HTTPException(status_code=400, detail="fit_method must be 'pmi', 'plm', or 'pmi+plm'.")

    for name, value in {
        "plm_epochs": payload.plm_epochs,
        "plm_batch_size": payload.plm_batch_size,
        "plm_progress_every": payload.plm_progress_every,
    }.items():
        if value is not None and int(value) < 1:
            raise HTTPException(status_code=400, detail=f"{name} must be >= 1.")

    if payload.plm_lr is not None and float(payload.plm_lr) <= 0:
        raise HTTPException(status_code=400, detail="plm_lr must be > 0.")
    if payload.plm_lr_min is not None and float(payload.plm_lr_min) < 0:
        raise HTTPException(status_code=400, detail="plm_lr_min must be >= 0.")
    if payload.plm_l2 is not None and float(payload.plm_l2) < 0:
        raise HTTPException(status_code=400, detail="plm_l2 must be >= 0.")
    if payload.plm_lr_schedule is not None and payload.plm_lr_schedule not in {"cosine", "none"}:
        raise HTTPException(status_code=400, detail="plm_lr_schedule must be 'cosine' or 'none'.")

    try:
        project_meta = project_store.get_project(payload.project_id)
        project_name = project_meta.name
    except Exception:
        project_name = None

    dataset_ref = {
        "project_id": payload.project_id,
        "project_name": project_name,
        "system_id": payload.system_id,
        "system_name": system_meta.name,
        "cluster_id": payload.cluster_id,
    }

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_potts_fit_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"potts-fit-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc
