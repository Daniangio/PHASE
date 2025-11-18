"""
QUBO Analysis using Random Forest Regression (Goal 2).
Extends BaseQUBO.
"""
import numpy as np
from typing import Tuple, Dict, Any, Callable
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from alloskin.common.types import FeatureDict

try:
    from .components import BaseQUBO
except ImportError:
    from alloskin.analysis.components import BaseQUBO


# --- Workers (Top-level) ---

def _rf_relevance_worker(item, y_target_3d, n_samples, cv_folds, n_estimators):
    """Compute R^2(Residue -> Target)."""
    key, X_3d = item
    try:
        X = X_3d.reshape(n_samples, -1)
        y = y_target_3d.reshape(n_samples, -1)
        reg = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=42)
        scores = cross_val_score(reg, X, y, cv=cv_folds, scoring='r2', n_jobs=1)
        return (key, np.mean(scores))
    except Exception:
        return (key, np.nan)

def _rf_redundancy_worker(item, all_features_static, n_samples, cv_folds, n_estimators):
    """Compute R^2 between two residues."""
    k_i, k_j = item
    try:
        X_i = all_features_static[k_i].reshape(n_samples, -1)
        X_j = all_features_static[k_j].reshape(n_samples, -1)
        
        reg = RandomForestRegressor(n_estimators=n_estimators, n_jobs=1, random_state=42)
        
        # i -> j
        s_ij = np.mean(cross_val_score(reg, X_i, X_j, cv=cv_folds, scoring='r2', n_jobs=1))
        # j -> i
        s_ji = np.mean(cross_val_score(reg, X_j, X_i, cv=cv_folds, scoring='r2', n_jobs=1))
        
        return (k_i, k_j, s_ij, s_ji)
    except Exception:
        return (k_i, k_j, np.nan, np.nan)


# --- Class ---

class QUBOSetRF(BaseQUBO):
    """
    QUBO implementation using Random Forest R^2.
    h_i = -R^2 (High R^2 is good, so energy term is negative)
    J_ij = lambda * max(R^2_ij, R^2_ji) (High redundancy is bad, energy term is positive)
    """
    
    def _get_relevance_worker(self) -> Callable:
        return _rf_relevance_worker

    def _get_redundancy_worker(self) -> Callable:
        return _rf_redundancy_worker

    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        return {
            'cv_folds': kwargs.get('qubo_cv_folds', 3),
            'n_estimators': kwargs.get('qubo_n_estimators', 50)
        }

    def _transform_relevance_to_h(self, r2_score: float) -> float:
        if np.isnan(r2_score): return 1e6
        # We want to MINIMIZE energy. High R^2 is good.
        # So h_i should be negative.
        return -max(r2_score, 0.0)

    def _transform_redundancy_to_J(self, r2_ij: float, r2_ji: float, lam: float) -> float:
        if np.isnan(r2_ij) or np.isnan(r2_ji): return np.nan
        # We want to penalize redundancy. High R^2 is bad.
        # So J_ij should be positive.
        max_r2 = np.nanmax([0.0, r2_ij, r2_ji])
        return lam * max_r2