"""
Pydantic Schemas for API request/response models.
"""

from pydantic import BaseModel
from typing import Dict, Optional, List, Union, Tuple

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
    sa_sweeps: Optional[int] = None
    sa_beta_hot: Optional[float] = None
    sa_beta_cold: Optional[float] = None
    sa_beta_schedules: Optional[List[Tuple[float, float]]] = None
    sa_init: Optional[str] = None
    sa_init_md_frame: Optional[int] = None
    sa_restart: Optional[str] = None
    sa_restart_topk: Optional[int] = None
    plm_epochs: Optional[int] = None
    plm_lr: Optional[float] = None
    plm_lr_min: Optional[float] = None
    plm_lr_schedule: Optional[str] = None
    plm_l2: Optional[float] = None
    plm_batch_size: Optional[int] = None
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
    plm_progress_every: Optional[int] = None
    plm_device: Optional[str] = None
    plm_init: Optional[str] = None
    plm_init_model: Optional[str] = None
    plm_resume_model: Optional[str] = None
    plm_val_frac: Optional[float] = None
