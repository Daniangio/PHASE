import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from backend.api.v1.common import ensure_system_ready, get_cluster_entry, get_queue, project_store
from backend.api.v1.schemas import (
    GibbsRelaxationJobRequest,
    LigandCompletionJobRequest,
    DeltaEvalJobRequest,
    DeltaCommitmentJobRequest,
    EndpointFrustrationJobRequest,
    DeltaJsJobRequest,
    TransientStatesJobRequest,
    DeltaTransitionJobRequest,
    LambdaSweepJobRequest,
    MdSamplesRefreshJobRequest,
    PottsNearestNeighborJobRequest,
    PottsAnalysisJobRequest,
    PottsFitJobRequest,
    SimulationJobRequest,
    StaticJobRequest,
)
from backend.tasks import (
    run_ligand_completion_job,
    run_gibbs_relaxation_job,
    run_analysis_job,
    run_delta_eval_job,
    run_delta_commitment_job,
    run_endpoint_frustration_job,
    run_delta_js_job,
    run_transient_states_job,
    run_delta_transition_job,
    run_lambda_sweep_job,
    run_md_samples_refresh_job,
    run_potts_analysis_job,
    run_potts_nearest_neighbor_job,
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

    if sampling_method == "sa":
        md_sample_id = str(payload.md_sample_id or "").strip()
        if not md_sample_id:
            raise HTTPException(status_code=400, detail="SA sampling requires md_sample_id.")
        cluster_entry = get_cluster_entry(system_meta, payload.cluster_id)
        sample_by_id = {
            str(sample.get("sample_id") or "").strip(): sample
            for sample in (cluster_entry.get("samples") or [])
            if isinstance(sample, dict) and str(sample.get("sample_id") or "").strip()
        }
        if md_sample_id not in sample_by_id:
            raise HTTPException(status_code=404, detail=f"MD sample '{md_sample_id}' not found on cluster.")
        if str(sample_by_id[md_sample_id].get("type") or "").strip() != "md_eval":
            raise HTTPException(status_code=400, detail="md_sample_id must reference an md_eval sample.")

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
        "sa_chains": payload.sa_chains,
        "sa_sweeps": payload.sa_sweeps,
        "sa_num_sweeps_per_beta": payload.sa_num_sweeps_per_beta,
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
    if payload.penalty_safety is not None and float(payload.penalty_safety) <= 0:
        raise HTTPException(status_code=400, detail="penalty_safety must be > 0.")
    if payload.repair is not None and str(payload.repair) not in {"none", "argmax"}:
        raise HTTPException(status_code=400, detail="repair must be 'none' or 'argmax'.")
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
    if payload.sa_schedule_type is not None:
        schedule_type = str(payload.sa_schedule_type).strip().lower()
        if schedule_type not in {"geometric", "linear", "custom", "geom", "lin"}:
            raise HTTPException(status_code=400, detail="sa_schedule_type must be geometric, linear, or custom.")
        if schedule_type == "custom" and not payload.sa_custom_beta_schedule:
            raise HTTPException(status_code=400, detail="sa_custom_beta_schedule is required when sa_schedule_type is custom.")
    if payload.sa_custom_beta_schedule is not None:
        if len(payload.sa_custom_beta_schedule) < 1:
            raise HTTPException(status_code=400, detail="sa_custom_beta_schedule must be non-empty.")
        for idx, beta in enumerate(payload.sa_custom_beta_schedule):
            if float(beta) < 0:
                raise HTTPException(status_code=400, detail=f"sa_custom_beta_schedule[{idx}] must be >= 0.")
        if payload.sa_beta_hot is not None or payload.sa_beta_cold is not None:
            raise HTTPException(status_code=400, detail="Use either sa_custom_beta_schedule or sa_beta_hot/sa_beta_cold, not both.")
    if payload.sa_acceptance_criteria is not None:
        criteria = str(payload.sa_acceptance_criteria).strip().lower()
        if criteria not in {"metropolis", "gibbs"}:
            raise HTTPException(status_code=400, detail="sa_acceptance_criteria must be 'Metropolis' or 'Gibbs'.")

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

    reference_a = str(payload.reference_sample_id_a or payload.md_sample_id_1 or "").strip()
    reference_b = str(payload.reference_sample_id_b or payload.md_sample_id_2 or "").strip()
    comparison_ids = [str(v).strip() for v in (payload.comparison_sample_ids or []) if str(v).strip()]
    if not comparison_ids:
        legacy_c = str(payload.md_sample_id_3 or "").strip()
        if legacy_c:
            comparison_ids = [legacy_c]

    if not reference_a or not reference_b:
        raise HTTPException(
            status_code=400,
            detail="reference_sample_id_a and reference_sample_id_b are required.",
        )
    if not comparison_ids:
        raise HTTPException(
            status_code=400,
            detail="comparison_sample_ids must contain at least one sample id.",
        )
    all_reference_ids = [reference_a, reference_b, *comparison_ids]
    if len(set(all_reference_ids)) != len(all_reference_ids):
        raise HTTPException(
            status_code=400,
            detail="Lambda sweep reference and comparison samples must be distinct.",
        )

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
    params.pop("md_sample_id_1", None)
    params.pop("md_sample_id_2", None)
    params.pop("md_sample_id_3", None)
    params["reference_sample_id_a"] = reference_a
    params["reference_sample_id_b"] = reference_b
    params["comparison_sample_ids"] = comparison_ids

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
    if payload.workers is not None and int(payload.workers) < 0:
        raise HTTPException(status_code=400, detail="workers must be >= 0.")
    analysis_edge_mode = (payload.analysis_edge_mode or "").strip().lower()
    if analysis_edge_mode and analysis_edge_mode not in {"model", "cluster", "contact", "all_vs_all"}:
        raise HTTPException(status_code=400, detail="analysis_edge_mode must be one of: model, cluster, contact, all_vs_all.")
    if payload.analysis_contact_cutoff is not None:
        cutoff = float(payload.analysis_contact_cutoff)
        if cutoff <= 0:
            raise HTTPException(status_code=400, detail="analysis_contact_cutoff must be > 0.")
    if payload.analysis_contact_atom_mode is not None:
        atom_mode = str(payload.analysis_contact_atom_mode).strip().upper()
        if atom_mode not in {"CA", "CM"}:
            raise HTTPException(status_code=400, detail="analysis_contact_atom_mode must be one of: CA, CM.")

    pose_only = bool(payload.pose_only)
    state_pose_ids = [str(v).strip() for v in (payload.state_pose_ids or []) if str(v).strip()]
    if pose_only:
        if not (payload.model_id or payload.model_path):
            raise HTTPException(status_code=400, detail="pose_only requires model_id or model_path.")
        if not state_pose_ids:
            raise HTTPException(status_code=400, detail="pose_only requires at least one state_pose_id.")
        missing_states = [
            state_id for state_id in state_pose_ids if state_id not in (system_meta.states or {})
        ]
        if missing_states:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown state ids for pose analysis: {', '.join(missing_states)}",
            )
        for state_id in state_pose_ids:
            state = (system_meta.states or {}).get(state_id)
            if not getattr(state, "pdb_file", None):
                raise HTTPException(
                    status_code=400,
                    detail=f"State '{state_id}' has no stored PDB.",
                )

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
    "/submit/potts_nn_mapping",
    summary="Submit a Potts-weighted nearest-neighbor mapping analysis (samples to MD in cluster space)",
)
async def submit_potts_nearest_neighbor_job(
    payload: PottsNearestNeighborJobRequest,
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
    sample_entries = cluster_entry.get("samples") if isinstance(cluster_entry, dict) else []
    sample_by_id = {
        str(sample.get("sample_id") or "").strip(): sample
        for sample in (sample_entries or [])
        if isinstance(sample, dict) and sample.get("sample_id")
    }
    sample_id = str(payload.sample_id or "").strip()
    md_sample_id = str(payload.md_sample_id or "").strip()
    if not sample_id or not md_sample_id:
        raise HTTPException(status_code=400, detail="sample_id and md_sample_id are required.")
    if sample_id not in sample_by_id:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found on cluster.")
    if md_sample_id not in sample_by_id:
        raise HTTPException(status_code=404, detail=f"MD sample '{md_sample_id}' not found on cluster.")
    if str(sample_by_id[md_sample_id].get("type") or "").strip() != "md_eval":
        raise HTTPException(status_code=400, detail="md_sample_id must reference an md_eval sample.")

    md_label_mode = (payload.md_label_mode or "assigned").lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise HTTPException(status_code=400, detail="md_label_mode must be 'assigned' or 'halo'.")

    if not (payload.model_id or payload.model_path):
        raise HTTPException(status_code=400, detail="Provide model_id or model_path.")

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    dataset_ref = {
        "project_id": payload.project_id,
        "system_id": payload.system_id,
        "cluster_id": payload.cluster_id,
    }

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_potts_nearest_neighbor_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="8h",
            result_ttl=86400,
            job_id=f"potts-nn-mapping-{job_uuid}",
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


@router.post(
    "/submit/ligand_completion",
    summary="Submit ligand-guided conditional completion analysis (A/B endpoints).",
)
async def submit_ligand_completion_job(
    payload: LigandCompletionJobRequest,
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

    constraint_mode = str(payload.constraint_source_mode or "manual").strip().lower()
    if constraint_mode not in {"manual", "delta_js_auto"}:
        raise HTTPException(status_code=400, detail="constraint_source_mode must be 'manual' or 'delta_js_auto'.")
    if constraint_mode == "manual":
        if not payload.constrained_residues or not isinstance(payload.constrained_residues, list):
            raise HTTPException(status_code=400, detail="constrained_residues must be a non-empty list.")
    else:
        if not (
            str(payload.constraint_delta_js_analysis_id or "").strip()
            or str(payload.delta_js_experiment_id or "").strip()
        ):
            raise HTTPException(
                status_code=400,
                detail="constraint_delta_js_analysis_id (or delta_js_experiment_id) is required when constraint_source_mode='delta_js_auto'.",
            )
        if payload.constraint_auto_top_k is not None and int(payload.constraint_auto_top_k) < 1:
            raise HTTPException(status_code=400, detail="constraint_auto_top_k must be >= 1.")
        if payload.constraint_auto_edge_alpha is not None:
            alpha = float(payload.constraint_auto_edge_alpha)
            if alpha < 0.0 or alpha > 1.0:
                raise HTTPException(status_code=400, detail="constraint_auto_edge_alpha must be in [0,1].")

    sampler = str(payload.sampler or "sa").strip().lower()
    if sampler not in {"sa", "gibbs"}:
        raise HTTPException(status_code=400, detail="sampler must be 'sa' or 'gibbs'.")

    md_label_mode = (payload.md_label_mode or "assigned").lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise HTTPException(status_code=400, detail="md_label_mode must be 'assigned' or 'halo'.")

    success_mode = str(payload.success_metric_mode or "deltae").strip().lower()
    if success_mode not in {"deltae", "delta_js_edge"}:
        raise HTTPException(status_code=400, detail="success_metric_mode must be 'deltae' or 'delta_js_edge'.")
    shared_delta_js_id = str(payload.delta_js_experiment_id or "").strip()
    if success_mode == "delta_js_edge" and not (
        str(payload.delta_js_analysis_id or "").strip() or shared_delta_js_id
    ):
        raise HTTPException(
            status_code=400,
            detail="delta_js_analysis_id (or delta_js_experiment_id) is required when success_metric_mode='delta_js_edge'.",
        )
    if payload.delta_js_node_edge_alpha is not None:
        alpha = float(payload.delta_js_node_edge_alpha)
        if alpha < 0.0 or alpha > 1.0:
            raise HTTPException(status_code=400, detail="delta_js_node_edge_alpha must be in [0,1].")
    if payload.delta_js_filter_edge_alpha is not None:
        alpha = float(payload.delta_js_filter_edge_alpha)
        if alpha < 0.0 or alpha > 1.0:
            raise HTTPException(status_code=400, detail="delta_js_filter_edge_alpha must be in [0,1].")
    if payload.js_success_threshold is not None and float(payload.js_success_threshold) < 0:
        raise HTTPException(status_code=400, detail="js_success_threshold must be >= 0.")

    if payload.lambda_values is not None:
        if len(payload.lambda_values) < 2:
            raise HTTPException(status_code=400, detail="lambda_values must contain at least 2 values.")
        if any(float(v) < 0 for v in payload.lambda_values):
            raise HTTPException(status_code=400, detail="lambda_values must be >= 0.")

    for name, value in {
        "n_start_frames": payload.n_start_frames,
        "n_samples_per_frame": payload.n_samples_per_frame,
        "n_steps": payload.n_steps,
        "tail_steps": payload.tail_steps,
        "target_window_size": payload.target_window_size,
    }.items():
        if value is not None and int(value) < 1:
            raise HTTPException(status_code=400, detail=f"{name} must be >= 1.")

    if payload.n_steps is not None and payload.tail_steps is not None:
        if int(payload.tail_steps) > int(payload.n_steps):
            raise HTTPException(status_code=400, detail="tail_steps cannot exceed n_steps.")

    if payload.sa_beta_hot is not None and float(payload.sa_beta_hot) <= 0:
        raise HTTPException(status_code=400, detail="sa_beta_hot must be > 0.")
    if payload.sa_beta_cold is not None and float(payload.sa_beta_cold) <= 0:
        raise HTTPException(status_code=400, detail="sa_beta_cold must be > 0.")
    if payload.sa_beta_hot is not None and payload.sa_beta_cold is not None:
        if float(payload.sa_beta_hot) > float(payload.sa_beta_cold):
            raise HTTPException(status_code=400, detail="sa_beta_hot must be <= sa_beta_cold.")

    if payload.gibbs_beta is not None and float(payload.gibbs_beta) <= 0:
        raise HTTPException(status_code=400, detail="gibbs_beta must be > 0.")

    if payload.target_pseudocount is not None and float(payload.target_pseudocount) < 0:
        raise HTTPException(status_code=400, detail="target_pseudocount must be >= 0.")
    if payload.epsilon_logpenalty is not None and float(payload.epsilon_logpenalty) <= 0:
        raise HTTPException(status_code=400, detail="epsilon_logpenalty must be > 0.")

    if payload.completion_target_success is not None:
        pstar = float(payload.completion_target_success)
        if not (0.0 < pstar <= 1.0):
            raise HTTPException(status_code=400, detail="completion_target_success must be in (0,1].")

    if payload.constraint_weights is not None and payload.constraint_weight_mode not in {None, "custom"}:
        raise HTTPException(
            status_code=400,
            detail="constraint_weights may be provided only with constraint_weight_mode='custom' (or omitted mode).",
        )
    if (
        constraint_mode == "manual"
        and payload.constraint_weights is not None
        and payload.constrained_residues is not None
        and len(payload.constraint_weights) != len(payload.constrained_residues)
    ):
        raise HTTPException(
            status_code=400,
            detail="constraint_weights must have the same length as constrained_residues.",
        )

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
            run_ligand_completion_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="8h",
            result_ttl=86400,
            job_id=f"ligand-completion-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post(
    "/submit/md_samples_refresh",
    summary="Recompute MD evaluation samples (md_eval) for all or selected descriptor-ready states in a cluster",
)
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

    raw_state_ids = payload.state_ids or []
    state_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw in raw_state_ids:
        sid = str(raw or "").strip()
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)
        state_ids.append(sid)

    descriptor_state_ids = {
        str(sid)
        for sid, state in (system_meta.states or {}).items()
        if getattr(state, "descriptor_file", None)
    }
    invalid = [sid for sid in state_ids if sid not in descriptor_state_ids]
    if invalid:
        raise HTTPException(status_code=400, detail=f"State(s) missing descriptors: {', '.join(invalid)}")

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    if state_ids:
        params["state_ids"] = state_ids
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

    model_a_id = str(payload.model_a_id or "").strip()
    model_b_id = str(payload.model_b_id or "").strip()
    using_models = bool(model_a_id or model_b_id)
    if using_models and (not model_a_id or not model_b_id):
        raise HTTPException(status_code=400, detail="Provide both model_a_id and model_b_id, or neither.")
    if using_models and model_a_id == model_b_id:
        raise HTTPException(status_code=400, detail="model_a_id and model_b_id must be different.")
    if not using_models:
        if not payload.reference_sample_ids_a or not payload.reference_sample_ids_b:
            raise HTTPException(
                status_code=400,
                detail="reference_sample_ids_a and reference_sample_ids_b are required when no model pair is provided.",
            )

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


@router.post(
    "/submit/endpoint_frustration",
    summary="Submit an endpoint-local frustration analysis for a fixed (model A, model B) pair.",
)
async def submit_endpoint_frustration_job(
    payload: EndpointFrustrationJobRequest,
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

    model_a_id = str(payload.model_a_id or "").strip()
    model_b_id = str(payload.model_b_id or "").strip()
    if not model_a_id or not model_b_id:
        raise HTTPException(status_code=400, detail="model_a_id and model_b_id are required.")
    if model_a_id == model_b_id:
        raise HTTPException(status_code=400, detail="model_a_id and model_b_id must be different.")
    if payload.top_k_edges is not None and int(payload.top_k_edges) < 1:
        raise HTTPException(status_code=400, detail="top_k_edges must be >= 1.")
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
            run_endpoint_frustration_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"endpoint-frustration-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post(
    "/submit/delta_js",
    summary="Submit an incremental delta-JS A/B/Other analysis (model-pair optional).",
)
async def submit_delta_js_job(
    payload: DeltaJsJobRequest,
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

    model_a_id = str(payload.model_a_id or "").strip()
    model_b_id = str(payload.model_b_id or "").strip()
    using_models = bool(model_a_id or model_b_id)
    if using_models and (not model_a_id or not model_b_id):
        raise HTTPException(status_code=400, detail="Provide both model_a_id and model_b_id, or neither.")
    if using_models and model_a_id == model_b_id:
        raise HTTPException(status_code=400, detail="model_a_id and model_b_id must be different.")

    if payload.top_k_residues is not None and int(payload.top_k_residues) < 1:
        raise HTTPException(status_code=400, detail="top_k_residues must be >= 1.")
    if payload.top_k_edges is not None and int(payload.top_k_edges) < 1:
        raise HTTPException(status_code=400, detail="top_k_edges must be >= 1.")
    edge_mode = str(payload.edge_mode or "").strip().lower()
    if edge_mode and edge_mode not in {"cluster", "all_vs_all", "contact"}:
        raise HTTPException(status_code=400, detail="edge_mode must be one of: cluster, all_vs_all, contact.")
    if not using_models:
        if not edge_mode:
            raise HTTPException(status_code=400, detail="edge_mode is required when no model pair is provided.")
        if not payload.reference_sample_ids_a or not payload.reference_sample_ids_b:
            raise HTTPException(
                status_code=400,
                detail="reference_sample_ids_a and reference_sample_ids_b are required when no model pair is provided.",
            )
        if edge_mode == "contact":
            has_states = bool(payload.contact_state_ids)
            has_pdbs = bool(payload.contact_pdbs)
            if not (has_states or has_pdbs):
                raise HTTPException(
                    status_code=400,
                    detail="edge_mode=contact requires contact_state_ids and/or contact_pdbs.",
                )
            if payload.contact_cutoff is not None and float(payload.contact_cutoff) <= 0:
                raise HTTPException(status_code=400, detail="contact_cutoff must be > 0.")
            if payload.contact_atom_mode is not None:
                mode = str(payload.contact_atom_mode).upper()
                if mode not in {"CA", "CM"}:
                    raise HTTPException(status_code=400, detail="contact_atom_mode must be 'CA' or 'CM'.")

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    dataset_ref = {
        "project_id": payload.project_id,
        "system_id": payload.system_id,
        "cluster_id": payload.cluster_id,
    }

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_delta_js_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="2h",
            result_ttl=86400,
            job_id=f"delta-js-{job_uuid}",
        )
        return {"status": "queued", "job_id": job.id, "analysis_uuid": job_uuid}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Job submission failed: {exc}") from exc


@router.post(
    "/submit/transient_states",
    summary="Submit a transient low-occupancy cluster-state enrichment analysis.",
)
async def submit_transient_states_job(
    payload: TransientStatesJobRequest,
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

    if not payload.sample_ids or len(payload.sample_ids) < 2:
        raise HTTPException(status_code=400, detail="sample_ids must include at least two samples.")

    md_label_mode = (payload.md_label_mode or "assigned").lower()
    if md_label_mode not in {"assigned", "halo"}:
        raise HTTPException(status_code=400, detail="md_label_mode must be 'assigned' or 'halo'.")

    p_min = float(payload.p_min if payload.p_min is not None else 0.005)
    p_max = float(payload.p_max if payload.p_max is not None else 0.05)
    if not (0.0 <= p_min <= p_max <= 1.0):
        raise HTTPException(status_code=400, detail="Require 0 <= p_min <= p_max <= 1.")

    edge_mode = str(payload.edge_mode or "cluster").strip().lower()
    if edge_mode not in {"cluster", "all_vs_all"}:
        raise HTTPException(status_code=400, detail="edge_mode must be 'cluster' or 'all_vs_all'.")

    params = payload.dict(exclude_none=True, exclude={"project_id", "system_id", "cluster_id"})
    dataset_ref = {
        "project_id": payload.project_id,
        "system_id": payload.system_id,
        "cluster_id": payload.cluster_id,
    }

    try:
        job_uuid = str(uuid.uuid4())
        job = task_queue.enqueue(
            run_transient_states_job,
            args=(job_uuid, dataset_ref, params),
            job_timeout="4h",
            result_ttl=86400,
            job_id=f"transient-states-{job_uuid}",
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
        if payload.sample_ids is not None:
            sample_ids = [str(v).strip() for v in payload.sample_ids if str(v).strip()]
            if not sample_ids:
                raise HTTPException(status_code=400, detail="sample_ids must contain at least one MD sample when provided.")
            payload.sample_ids = sample_ids
        if payload.fit_method is not None and payload.fit_method not in {"pmi", "plm", "pmi+plm"}:
            raise HTTPException(status_code=400, detail="fit_method must be 'pmi', 'plm', or 'pmi+plm'.")

        for name, value in {
            "plm_epochs": payload.plm_epochs,
            "plm_batch_size": payload.plm_batch_size,
            "plm_grad_accum_steps": payload.plm_grad_accum_steps,
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
            "delta_grad_accum_steps": payload.delta_grad_accum_steps,
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
