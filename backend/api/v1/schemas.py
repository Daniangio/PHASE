"""
Pydantic Schemas for API request/response models.
"""

from pydantic import BaseModel
from typing import Any, Dict, Optional, List, Union, Tuple

class AnalysisPaths(BaseModel):
    """Input model for file paths."""
    active_traj: str
    active_topo: str
    inactive_traj: str
    inactive_topo: str
    config_file: str # Path to the config file (on the server)

class ErrorResponse(BaseModel):
    """Error response model."""
    status: str
    error: str


class ProjectCreateRequest(BaseModel):
    """Request payload for creating a new project."""
    name: str
    description: Optional[str] = None
    use_slug_ids: Optional[bool] = False


class SystemCloneRequest(BaseModel):
    """Request payload for selectively cloning an existing system."""
    name: str
    description: Optional[str] = None
    use_slug_ids: Optional[bool] = False


class AnalysisJobBase(BaseModel):
    """Shared fields for analysis job submissions."""
    project_id: str
    system_id: str
    state_a_id: str
    state_b_id: str


class StaticJobRequest(AnalysisJobBase):
    state_metric: str = "auc"
    maxk: Optional[int] = None


class SimulationJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    md_sample_id: Optional[str] = None
    sampling_method: Optional[str] = None
    sample_name: Optional[str] = None
    use_potts_model: Optional[bool] = True
    potts_model_path: Optional[str] = None
    potts_model_paths: Optional[List[str]] = None
    potts_model_id: Optional[str] = None
    potts_model_ids: Optional[List[str]] = None
    contact_cutoff: Optional[float] = None
    contact_atom_mode: Optional[str] = None
    rex_betas: Optional[Union[str, List[float]]] = None
    rex_beta_min: Optional[float] = None
    rex_beta_max: Optional[float] = None
    rex_spacing: Optional[str] = None
    rex_samples: Optional[int] = None
    rex_burnin: Optional[int] = None
    rex_thin: Optional[int] = None
    sa_reads: Optional[int] = None
    sa_chains: Optional[int] = None
    sa_sweeps: Optional[int] = None
    sa_beta_hot: Optional[float] = None
    sa_beta_cold: Optional[float] = None
    sa_beta_schedules: Optional[List[Tuple[float, float]]] = None
    sa_schedule_type: Optional[str] = None
    sa_custom_beta_schedule: Optional[List[float]] = None
    sa_num_sweeps_per_beta: Optional[int] = None
    sa_randomize_order: Optional[bool] = None
    sa_acceptance_criteria: Optional[str] = None
    sa_init: Optional[str] = None
    sa_init_md_frame: Optional[int] = None
    sa_restart: Optional[str] = None
    sa_restart_topk: Optional[int] = None
    sa_md_state_ids: Optional[str] = None
    penalty_safety: Optional[float] = None
    repair: Optional[str] = None
    plm_epochs: Optional[int] = None
    plm_lr: Optional[float] = None
    plm_lr_min: Optional[float] = None
    plm_lr_schedule: Optional[str] = None
    plm_l2: Optional[float] = None
    plm_batch_size: Optional[int] = None
    plm_grad_accum_steps: Optional[int] = None
    plm_progress_every: Optional[int] = None
    plm_device: Optional[str] = None
    plm_init: Optional[str] = None
    plm_init_model: Optional[str] = None
    plm_resume_model: Optional[str] = None
    plm_val_frac: Optional[float] = None


class PottsFitJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    model_name: Optional[str] = None
    fit_method: Optional[str] = None
    fit_mode: Optional[str] = None
    sample_ids: Optional[List[str]] = None
    base_model_id: Optional[str] = None
    base_model_path: Optional[str] = None
    state_ids: Optional[List[str]] = None
    active_state_id: Optional[str] = None
    inactive_state_id: Optional[str] = None
    active_npz: Optional[str] = None
    inactive_npz: Optional[str] = None
    unassigned_policy: Optional[str] = None
    delta_epochs: Optional[int] = None
    delta_lr: Optional[float] = None
    delta_lr_min: Optional[float] = None
    delta_lr_schedule: Optional[str] = None
    delta_batch_size: Optional[int] = None
    delta_grad_accum_steps: Optional[int] = None
    delta_seed: Optional[int] = None
    delta_device: Optional[str] = None
    delta_l2: Optional[float] = None
    delta_group_h: Optional[float] = None
    delta_group_j: Optional[float] = None
    delta_no_combined: Optional[bool] = None
    contact_cutoff: Optional[float] = None
    contact_atom_mode: Optional[str] = None
    plm_epochs: Optional[int] = None
    plm_lr: Optional[float] = None
    plm_lr_min: Optional[float] = None
    plm_lr_schedule: Optional[str] = None
    plm_l2: Optional[float] = None
    plm_batch_size: Optional[int] = None
    plm_grad_accum_steps: Optional[int] = None
    plm_progress_every: Optional[int] = None
    plm_device: Optional[str] = None


class PottsAnalysisJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    workers: Optional[int] = None
    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None
    sample_ids: Optional[List[str]] = None
    plm_init: Optional[str] = None
    plm_init_model: Optional[str] = None
    plm_resume_model: Optional[str] = None
    plm_val_frac: Optional[float] = None
    analysis_edge_mode: Optional[str] = None  # model|cluster|contact|all_vs_all
    analysis_contact_cutoff: Optional[float] = None
    analysis_contact_atom_mode: Optional[str] = None  # CA|CM
    analysis_contact_state_ids: Optional[List[str]] = None
    analysis_contact_pdbs: Optional[List[str]] = None


class MdSamplesRefreshJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    state_ids: Optional[List[str]] = None
    overwrite: Optional[bool] = True
    cleanup: Optional[bool] = True


class DeltaEvalJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    md_sample_id: str
    model_a_id: str
    model_b_id: str
    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None


class DeltaTransitionJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    active_md_sample_id: str
    inactive_md_sample_id: str
    pas_md_sample_id: str
    model_a_id: str
    model_b_id: str
    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None
    band_fraction: Optional[float] = None
    top_k_residues: Optional[int] = None
    top_k_edges: Optional[int] = None
    seed: Optional[int] = None


class DeltaCommitmentJobRequest(BaseModel):
    """
    Incremental delta-commitment analysis for a fixed (model A, model B) pair.

    The analysis stores discriminative power once for the pair and appends/overwrites
    per-sample commitment for the selected samples.
    """

    project_id: str
    system_id: str
    cluster_id: str
    model_a_id: str
    model_b_id: str
    sample_ids: List[str]
    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None
    top_k_residues: Optional[int] = None
    top_k_edges: Optional[int] = None
    ranking_method: Optional[str] = None
    energy_bins: Optional[int] = None


class EndpointFrustrationJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    model_a_id: str
    model_b_id: str
    sample_ids: List[str]
    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None
    top_k_edges: Optional[int] = None
    workers: Optional[int] = None


class DeltaEnergyJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    model_a_id: str
    model_b_id: str
    sample_ids: List[str]
    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None
    frame_limits: Optional[Dict[str, int]] = None
    seed: Optional[int] = None
    energy_bins: Optional[int] = None
    workers: Optional[int] = None


class HamiltonianSpectralJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    state_ids: List[str]
    top_k: Optional[int] = None
    overwrite: Optional[bool] = None


class SpectralIntersectionJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    single_analysis_id: str
    pair_analysis_id: str
    min_group_size: Optional[int] = None
    overwrite: Optional[bool] = None


class PistonLigandProjectionJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    intersection_analysis_id: str
    sample_ids: List[str]
    label_mode: Optional[str] = None
    drop_invalid: Optional[bool] = None
    overwrite: Optional[bool] = None


class DeltaJsJobRequest(BaseModel):
    """
    Incremental delta-JS A/B/Other analysis for a fixed (model A, model B) pair.
    """

    project_id: str
    system_id: str
    cluster_id: str
    model_a_id: Optional[str] = None
    model_b_id: Optional[str] = None
    sample_ids: List[str]
    reference_sample_ids_a: Optional[List[str]] = None
    reference_sample_ids_b: Optional[List[str]] = None
    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None
    top_k_residues: Optional[int] = None
    top_k_edges: Optional[int] = None
    ranking_method: Optional[str] = None
    edge_mode: Optional[str] = None  # cluster|all_vs_all|contact (required if no model pair)
    contact_state_ids: Optional[List[str]] = None
    contact_pdbs: Optional[List[str]] = None
    contact_cutoff: Optional[float] = None
    contact_atom_mode: Optional[str] = None  # CA|CM


class TransientStatesJobRequest(BaseModel):
    """
    Transient low-occupancy cluster-state enrichment analysis.
    """

    project_id: str
    system_id: str
    cluster_id: str
    sample_ids: List[str]
    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None
    p_min: Optional[float] = None
    p_max: Optional[float] = None
    enrichment_min: Optional[float] = None
    epsilon: Optional[float] = None
    top_k_nodes: Optional[int] = None
    include_edges: Optional[bool] = None
    edge_mode: Optional[str] = None  # cluster|all_vs_all
    edge_p_min: Optional[float] = None
    edge_p_max: Optional[float] = None
    edge_enrichment_min: Optional[float] = None
    delta_pmi_min: Optional[float] = None
    top_k_edges: Optional[int] = None


class LambdaSweepJobRequest(BaseModel):
    """
    Validation ladder 4: sample from an interpolated model E_λ between two endpoint Potts models.

    The job creates N correlated Gibbs samples (λ grid) and saves a dedicated analysis artifact
    with JS-distance curves vs three reference MD samples.
    """

    project_id: str
    system_id: str
    cluster_id: str

    # Endpoint models (λ=1 and λ=0)
    model_a_id: str
    model_b_id: str

    # Generic reference samples:
    #   - A/B anchor the endpoint interpretation
    #   - comparison samples define the match curves
    reference_sample_id_a: Optional[str] = None
    reference_sample_id_b: Optional[str] = None
    comparison_sample_ids: Optional[List[str]] = None

    # Backward compatibility for the old fixed 3-MD flow.
    md_sample_id_1: Optional[str] = None
    md_sample_id_2: Optional[str] = None
    md_sample_id_3: Optional[str] = None


class PottsNearestNeighborJobRequest(BaseModel):
    project_id: str
    system_id: str
    cluster_id: str
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    sample_id: str
    md_sample_id: str
    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None
    use_unique: Optional[bool] = None
    normalize: Optional[bool] = None
    compute_per_residue: Optional[bool] = None
    alpha: Optional[float] = None
    beta_node: Optional[float] = None
    beta_edge: Optional[float] = None
    top_k_candidates: Optional[int] = None
    chunk_size: Optional[int] = None
    distance_thresholds: Optional[List[float]] = None
    workers: Optional[int] = None

    series_id: Optional[str] = None
    series_label: Optional[str] = None

    lambda_count: Optional[int] = None
    alpha: Optional[float] = None

    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None

    # Gibbs params (single-site or replica exchange)
    gibbs_method: Optional[str] = None  # single|rex
    beta: Optional[float] = None
    seed: Optional[int] = None

    gibbs_samples: Optional[int] = None
    gibbs_burnin: Optional[int] = None
    gibbs_thin: Optional[int] = None

    rex_betas: Optional[Union[str, List[float]]] = None
    rex_beta_min: Optional[float] = None
    rex_beta_max: Optional[float] = None
    rex_spacing: Optional[str] = None
    rex_n_replicas: Optional[int] = None
    rex_rounds: Optional[int] = None
    rex_burnin_rounds: Optional[int] = None
    rex_sweeps_per_round: Optional[int] = None
    rex_thin_rounds: Optional[int] = None


class LigandCompletionJobRequest(BaseModel):
    """
    Ligand-guided conditional completion analysis (dev_docs.md).
    """

    project_id: str
    system_id: str
    cluster_id: str

    model_a_id: str
    model_b_id: str
    md_sample_id: str
    constrained_residues: Optional[List[Union[str, int]]] = None

    reference_sample_id_a: Optional[str] = None
    reference_sample_id_b: Optional[str] = None

    sampler: Optional[str] = None  # sa|gibbs
    lambda_values: Optional[List[float]] = None
    n_start_frames: Optional[int] = None
    n_samples_per_frame: Optional[int] = None
    n_steps: Optional[int] = None
    tail_steps: Optional[int] = None

    target_window_size: Optional[int] = None
    target_pseudocount: Optional[float] = None
    epsilon_logpenalty: Optional[float] = None

    constraint_weight_mode: Optional[str] = None  # uniform|js_abs|custom
    constraint_weights: Optional[List[float]] = None
    constraint_weight_min: Optional[float] = None
    constraint_weight_max: Optional[float] = None
    constraint_source_mode: Optional[str] = None  # manual|delta_js_auto
    constraint_delta_js_analysis_id: Optional[str] = None
    constraint_delta_js_sample_id: Optional[str] = None
    constraint_auto_top_k: Optional[int] = None
    constraint_auto_edge_alpha: Optional[float] = None
    constraint_auto_exclude_success: Optional[bool] = None

    gibbs_beta: Optional[float] = None
    sa_beta_hot: Optional[float] = None
    sa_beta_cold: Optional[float] = None
    sa_schedule: Optional[str] = None  # geom|lin

    md_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None
    success_metric_mode: Optional[str] = None  # deltae|delta_js_edge
    delta_js_experiment_id: Optional[str] = None
    delta_js_analysis_id: Optional[str] = None
    delta_js_filter_setup_id: Optional[str] = None
    delta_js_filter_edge_alpha: Optional[float] = None
    delta_js_d_residue_min: Optional[float] = None
    delta_js_d_residue_max: Optional[float] = None
    delta_js_d_edge_min: Optional[float] = None
    delta_js_d_edge_max: Optional[float] = None
    delta_js_node_edge_alpha: Optional[float] = None
    js_success_threshold: Optional[float] = None
    js_success_margin: Optional[float] = None
    deltae_margin: Optional[float] = None
    completion_target_success: Optional[float] = None
    completion_cost_if_unreached: Optional[float] = None
    workers: Optional[int] = None
    seed: Optional[int] = None


class UiSetupUpsertRequest(BaseModel):
    """Persist UI setup/preset payloads on a cluster."""

    name: str
    setup_type: str
    page: Optional[str] = None
    payload: Dict[str, Any]
    setup_id: Optional[str] = None


class GibbsRelaxationJobRequest(BaseModel):
    """
    Relaxation analysis from MD starts under a selected Potts Hamiltonian.

    Workflow:
      - randomly choose starting frames from one MD sample
      - run Gibbs trajectories under model H
      - aggregate first-flip/percentile statistics
    """

    project_id: str
    system_id: str
    cluster_id: str

    start_sample_id: str
    model_id: Optional[str] = None
    model_path: Optional[str] = None

    beta: Optional[float] = None
    n_start_frames: Optional[int] = None
    gibbs_sweeps: Optional[int] = None
    seed: Optional[int] = None
    workers: Optional[int] = None

    start_label_mode: Optional[str] = None  # assigned|halo
    keep_invalid: Optional[bool] = None


class LambdaPottsModelCreateRequest(BaseModel):
    """
    Create a derived Potts model by interpolating two existing endpoint models:
      E_λ = (1-λ) * E_B + λ * E_A
    where B corresponds to λ=0 and A corresponds to λ=1.
    """

    model_a_id: str
    model_b_id: str
    lam: float
    name: Optional[str] = None
    zero_sum_gauge: Optional[bool] = True
