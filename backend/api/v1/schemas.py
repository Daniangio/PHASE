"""
Pydantic Schemas for API request/response models.
"""

from pydantic import BaseModel
from typing import Dict, Any, Optional

class AnalysisPaths(BaseModel):
    """Input model for file paths."""
    active_traj: str
    active_topo: str
    inactive_traj: str
    inactive_topo: str
    config_file: str # Path to the config file (on the server)

class StaticAnalysisParams(BaseModel):
    """Parameters for Goal 1."""
    paths: AnalysisPaths

class DynamicAnalysisParams(BaseModel):
    """Parameters for Goal 3."""
    paths: AnalysisPaths
    te_lag: Optional[int] = 10
    
class StaticAnalysisResponse(BaseModel):
    """Response model for Goal 1."""
    status: str
    goal: str
    results: Dict[str, float]

class DynamicAnalysisResponse(BaseModel):
    """Response model for Goal 3."""
    status: str
    goal: str
    results: Dict[str, Any] # Contains M_inactive, M_active
    
class ErrorResponse(BaseModel):
    """Error response model."""
    status: str
    error: str


class ProjectCreateRequest(BaseModel):
    """Request payload for creating a new project."""
    name: str
    description: Optional[str] = None


class AnalysisJobBase(BaseModel):
    """Shared fields for analysis job submissions."""
    project_id: str
    system_id: str
    state_a_id: str
    state_b_id: str


class StaticJobRequest(AnalysisJobBase):
    state_metric: str = "auc"
    maxk: Optional[int] = None


class DynamicJobRequest(AnalysisJobBase):
    te_lag: int = 10


class QUBOJobRequest(AnalysisJobBase):
    static_job_uuid: Optional[str] = None
    imbalance_matrix_paths_active: Optional[list[str]] = None
    imbalance_matrix_paths_inactive: Optional[list[str]] = None
    alpha_size: float = 1.0
    beta_hub: float = 1.0
    beta_switch: float = 5.0
    gamma_redundancy: float = 2.0
    ii_threshold: float = 0.9
    ii_scale: float = 0.6
    soft_threshold_power: float = 2.0
    num_reads: int = 2000
    num_solutions: int = 5
    seed: Optional[int] = None
    maxk: Optional[int] = None
    taxonomy_switch_high: float = 0.8
    taxonomy_switch_low: float = 0.3
    taxonomy_hub_high_percentile: float = 80.0
    taxonomy_hub_low_percentile: float = 50.0
    taxonomy_delta_hub_high_percentile: float = 80.0
    filter_top_total: int = 120
    filter_top_jsd: int = 20
    filter_min_id: float = 1.5
