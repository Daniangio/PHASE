import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from backend.api.v1.common import ensure_system_ready, get_cluster_entry, get_queue, project_store
from backend.api.v1.schemas import (
    GibbsRelaxationJobRequest,
    DeltaEvalJobRequest,
    DeltaCommitmentJobRequest,
    DeltaTransitionJobRequest,
    LambdaSweepJobRequest,
    MdSamplesRefreshJobRequest,
    PottsAnalysisJobRequest,
    PottsFitJobRequest,
    SimulationJobRequest,
    StaticJobRequest,
)
from backend.tasks import (
    run_gibbs_relaxation_job,
    run_analysis_job,
    run_delta_eval_job,
    run_delta_commitment_job,
    run_delta_transition_job,
    run_lambda_sweep_job,
    run_md_samples_refresh_job,
    run_potts_analysis_job,
    run_potts_fit_job,
    run_simulation_job,
)


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

    if payload.sa_init is not None:
        sa_init = str(payload.sa_init)
        if sa_init not in {"md", "md-frame", "random-h", "random-uniform"}:
            raise HTTPException(status_code=400, detail="sa_init must be one of: md, md-frame, random-h, random-uniform.")
        if sa_init == "md-frame":
            if payload.sa_init_md_frame is None:
                raise HTTPException(status_code=400, detail="sa_init_md_frame is required when sa_init is md-frame.")
            if int(payload.sa_init_md_frame) < 0:
                raise HTTPException(status_code=400, detail="sa_init_md_frame must be >= 0.")
    if payload.sa_init_md_frame is not None and int(payload.sa_init_md_frame) < 0:
        raise HTTPException(status_code=400, detail="sa_init_md_frame must be >= 0.")

    if payload.sa_restart is not None:
        sa_restart = str(payload.sa_restart).strip().lower()
        # Accept current sampling modes + legacy UI values.
        if sa_restart in {"prev-topk", "prev-uniform", "prev", "chain"}:
            sa_restart = "previous"
        elif sa_restart in {"md-frame", "md_random", "md-random"}:
            sa_restart = "md"
        elif sa_restart in {"indep", "iid", "rand", "random"}:
            sa_restart = "independent"
        if sa_restart not in {"independent", "previous", "md"}:
            raise HTTPException(
                status_code=400,
                detail="sa_restart must be one of: independent, previous, md.",
            )
    if payload.sa_restart_topk is not None and int(payload.sa_restart_topk) < 1:
        raise HTTPException(status_code=400, detail="sa_restart_topk must be >= 1.")

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


@router.post("/submit/lambda_sweep", summary="Submit a lambda-interpolation sweep (validation ladder 4)")
async def submit_lambda_sweep_job(
    payload: LambdaSweepJobRequest,
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

    if payload.model_a_id == payload.model_b_id:
        raise HTTPException(status_code=400, detail="model_a_id and model_b_id must be different.")

    if payload.lambda_count is not None and int(payload.lambda_count) < 2:
        raise HTTPException(status_code=400, detail="lambda_count must be >= 2.")
    if payload.alpha is not None:
        alpha = float(payload.alpha)
        if not (0.0 <= alpha <= 1.0):
            raise HTTPException(status_code=400, detail="alpha must be in [0,1].")

    md_label_mode = (payload.md_label_mode or "assigned").lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise HTTPException(status_code=400, detail="md_label_mode must be 'assigned' or 'halo'.")

    if len({payload.md_sample_id_1, payload.md_sample_id_2, payload.md_sample_id_3}) < 3:
        raise HTTPException(status_code=400, detail="md_sample_id_1/2/3 must be three distinct samples.")

    gibbs_method = (payload.gibbs_method or "rex").lower()
    if gibbs_method not in {"single", "rex"}:
        raise HTTPException(status_code=400, detail="gibbs_method must be 'single' or 'rex'.")

    if payload.beta is not None and float(payload.beta) <= 0:
        raise HTTPException(status_code=400, detail="beta must be > 0.")

    for name, value in {
        "gibbs_samples": payload.gibbs_samples,
        "gibbs_burnin": payload.gibbs_burnin,
        "gibbs_thin": payload.gibbs_thin,
        "rex_n_replicas": payload.rex_n_replicas,
        "rex_rounds": payload.rex_rounds,
        "rex_burnin_rounds": payload.rex_burnin_rounds,
        "rex_sweeps_per_round": payload.rex_sweeps_per_round,
        "rex_thin_rounds": payload.rex_thin_rounds,
    }.items():
        if value is not None and int(value) < 1:
            raise HTTPException(status_code=400, detail=f"{name} must be >= 1.")

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
            run_lambda_sweep_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="4h",
            result_ttl=86400,
            job_id=f"lambda-sweep-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post("/submit/potts_analysis", summary="Submit a Potts sample analysis job")
async def submit_potts_analysis_job(
    payload: PottsAnalysisJobRequest,
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

    md_label_mode = (payload.md_label_mode or "assigned").lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise HTTPException(status_code=400, detail="md_label_mode must be 'assigned' or 'halo'.")

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    dataset_ref = {
        "project_id": payload.project_id,
        "system_id": payload.system_id,
        "cluster_id": payload.cluster_id,
    }

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_potts_analysis_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"potts-analysis-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post(
    "/submit/gibbs_relaxation",
    summary="Submit Gibbs relaxation analysis from random MD starts under a selected Potts model",
)
async def submit_gibbs_relaxation_job(
    payload: GibbsRelaxationJobRequest,
    task_queue: Any = Depends(get_queue),
):
    try:
        system_meta = project_store.get_system(payload.project_id, payload.system_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"System '{payload.system_id}' not found in project '{payload.project_id}'.",
        )

    cluster_entry = get_cluster_entry(system_meta, payload.cluster_id)

    model_id = str(payload.model_id or "").strip()
    model_path = str(payload.model_path or "").strip()
    if not model_id and not model_path:
        raise HTTPException(status_code=400, detail="Provide model_id or model_path.")

    sample_id = str(payload.start_sample_id or "").strip()
    if not sample_id:
        raise HTTPException(status_code=400, detail="start_sample_id is required.")
    sample_list = cluster_entry.get("samples") if isinstance(cluster_entry, dict) else []
    if not isinstance(sample_list, list) or not any(
        isinstance(s, dict) and str(s.get("sample_id")) == sample_id for s in sample_list
    ):
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found in cluster metadata.")

    label_mode = (payload.start_label_mode or "assigned").lower()
    if label_mode not in {"assigned", "halo"}:
        raise HTTPException(status_code=400, detail="start_label_mode must be 'assigned' or 'halo'.")

    if payload.beta is not None and float(payload.beta) <= 0:
        raise HTTPException(status_code=400, detail="beta must be > 0.")
    if payload.n_start_frames is not None and int(payload.n_start_frames) < 1:
        raise HTTPException(status_code=400, detail="n_start_frames must be >= 1.")
    if payload.gibbs_sweeps is not None and int(payload.gibbs_sweeps) < 1:
        raise HTTPException(status_code=400, detail="gibbs_sweeps must be >= 1.")
    if payload.workers is not None and int(payload.workers) < 0:
        raise HTTPException(status_code=400, detail="workers must be >= 0.")

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    dataset_ref = {
        "project_id": payload.project_id,
        "system_id": payload.system_id,
        "cluster_id": payload.cluster_id,
    }

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_gibbs_relaxation_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="6h",
            result_ttl=86400,
            job_id=f"gibbs-relaxation-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post("/submit/md_samples_refresh", summary="Recompute MD evaluation samples (md_eval) for all states in a cluster")
async def submit_md_samples_refresh_job(
    payload: MdSamplesRefreshJobRequest,
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

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    dataset_ref = {
        "project_id": payload.project_id,
        "system_id": payload.system_id,
        "cluster_id": payload.cluster_id,
    }

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_md_samples_refresh_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"md-samples-refresh-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post("/submit/delta_eval", summary="Submit a delta-Potts evaluation job on an MD sample (per-residue/edge preferences)")
async def submit_delta_eval_job(
    payload: DeltaEvalJobRequest,
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

    md_label_mode = (payload.md_label_mode or "assigned").lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise HTTPException(status_code=400, detail="md_label_mode must be 'assigned' or 'halo'.")

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    dataset_ref = {
        "project_id": payload.project_id,
        "system_id": payload.system_id,
        "cluster_id": payload.cluster_id,
    }

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_delta_eval_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"delta-eval-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post(
    "/submit/delta_transition",
    summary="Submit a transition-like (TS-band) delta-Potts analysis across Active/Inactive/pAS MD samples",
)
async def submit_delta_transition_job(
    payload: DeltaTransitionJobRequest,
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

    md_label_mode = (payload.md_label_mode or "assigned").lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise HTTPException(status_code=400, detail="md_label_mode must be 'assigned' or 'halo'.")

    if payload.band_fraction is not None:
        band = float(payload.band_fraction)
        if not (0 < band < 1):
            raise HTTPException(status_code=400, detail="band_fraction must be in (0,1).")

    if payload.top_k_residues is not None and int(payload.top_k_residues) < 1:
        raise HTTPException(status_code=400, detail="top_k_residues must be >= 1.")
    if payload.top_k_edges is not None and int(payload.top_k_edges) < 1:
        raise HTTPException(status_code=400, detail="top_k_edges must be >= 1.")

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    dataset_ref = {
        "project_id": payload.project_id,
        "system_id": payload.system_id,
        "cluster_id": payload.cluster_id,
    }

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_delta_transition_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"delta-transition-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post(
    "/submit/delta_commitment",
    summary="Submit an incremental delta-commitment analysis for a fixed (model A, model B) pair.",
)
async def submit_delta_commitment_job(
    payload: DeltaCommitmentJobRequest,
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

    md_label_mode = (payload.md_label_mode or "assigned").lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise HTTPException(status_code=400, detail="md_label_mode must be 'assigned' or 'halo'.")

    if not payload.sample_ids or not isinstance(payload.sample_ids, list):
        raise HTTPException(status_code=400, detail="sample_ids must be a non-empty list.")

    if payload.top_k_residues is not None and int(payload.top_k_residues) < 1:
        raise HTTPException(status_code=400, detail="top_k_residues must be >= 1.")
    if payload.top_k_edges is not None and int(payload.top_k_edges) < 1:
        raise HTTPException(status_code=400, detail="top_k_edges must be >= 1.")
    if payload.energy_bins is not None and int(payload.energy_bins) < 5:
        raise HTTPException(status_code=400, detail="energy_bins must be >= 5.")

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    dataset_ref = {
        "project_id": payload.project_id,
        "system_id": payload.system_id,
        "cluster_id": payload.cluster_id,
    }

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_delta_commitment_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"delta-commitment-{job_uuid}",
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

    fit_mode = payload.fit_mode
    if not fit_mode:
        if (
            payload.base_model_id
            or payload.base_model_path
            or payload.state_ids
            or payload.active_state_id
            or payload.inactive_state_id
        ):
            fit_mode = "delta"
        elif payload.active_npz or payload.inactive_npz:
            fit_mode = "delta"
        else:
            fit_mode = "standard"
    if fit_mode not in {"standard", "delta"}:
        raise HTTPException(status_code=400, detail="fit_mode must be 'standard' or 'delta'.")

    if fit_mode != "delta":
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
    else:
        if not payload.base_model_id and not payload.base_model_path:
            raise HTTPException(status_code=400, detail="Delta fit requires base_model_id or base_model_path.")
        if payload.active_npz or payload.inactive_npz:
            if not (payload.active_npz and payload.inactive_npz):
                raise HTTPException(
                    status_code=400,
                    detail="Provide both active_npz and inactive_npz for delta fit.",
                )
        elif payload.state_ids:
            if not isinstance(payload.state_ids, list) or len(payload.state_ids) < 1:
                raise HTTPException(status_code=400, detail="state_ids must contain at least one entry.")
        else:
            if not (payload.active_state_id and payload.inactive_state_id):
                raise HTTPException(
                    status_code=400,
                    detail="Provide state_ids (preferred) or active_state_id and inactive_state_id for delta fit.",
                )
        if payload.unassigned_policy is not None and payload.unassigned_policy not in {"drop_frames", "treat_as_state", "error"}:
            raise HTTPException(
                status_code=400,
                detail="unassigned_policy must be 'drop_frames', 'treat_as_state', or 'error'.",
            )
        for name, value in {
            "delta_epochs": payload.delta_epochs,
            "delta_batch_size": payload.delta_batch_size,
        }.items():
            if value is not None and int(value) < 1:
                raise HTTPException(status_code=400, detail=f"{name} must be >= 1.")
        if payload.delta_lr is not None and float(payload.delta_lr) <= 0:
            raise HTTPException(status_code=400, detail="delta_lr must be > 0.")
        if payload.delta_lr_min is not None and float(payload.delta_lr_min) < 0:
            raise HTTPException(status_code=400, detail="delta_lr_min must be >= 0.")
        if payload.delta_l2 is not None and float(payload.delta_l2) < 0:
            raise HTTPException(status_code=400, detail="delta_l2 must be >= 0.")
        if payload.delta_group_h is not None and float(payload.delta_group_h) < 0:
            raise HTTPException(status_code=400, detail="delta_group_h must be >= 0.")
        if payload.delta_group_j is not None and float(payload.delta_group_j) < 0:
            raise HTTPException(status_code=400, detail="delta_group_j must be >= 0.")
        if payload.delta_lr_schedule is not None and payload.delta_lr_schedule not in {"cosine", "none"}:
            raise HTTPException(status_code=400, detail="delta_lr_schedule must be 'cosine' or 'none'.")

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
