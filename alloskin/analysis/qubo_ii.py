"""
QUBO Analysis using Information Imbalance (Goal 2).
Extends BaseQUBO.

Refactored to use 'dadapy.metric_comparisons'.
"""
import numpy as np
from typing import Tuple, Dict, Any, Callable
from alloskin.common.types import FeatureDict

try:
    from dadapy.metric_comparisons import MetricComparisons
    DADAPY_AVAILABLE = True
except ImportError:
    DADAPY_AVAILABLE = False

try:
    from .components import BaseQUBO
except ImportError:
    from alloskin.analysis.components import BaseQUBO


# --- Workers (Top-level) ---

def _compute_imbalance_raw(source_coords, target_coords=None, target_ranks=None, maxk=None):
    """
    Computes Delta(Source -> Target).
    Can accept either raw target_coords (computes ranks on fly)
    OR pre-computed target_ranks.
    """
    try:
        n_samples = source_coords.shape[0]
        if maxk is None: maxk = n_samples - 1

        # 1. Setup Source
        mc_source = MetricComparisons(coordinates=source_coords, maxk=maxk, n_jobs=1)
        mc_source.compute_distances()
        
        # 2. Setup Target Ranks
        if target_ranks is None:
            if target_coords is None: return np.nan
            mc_target = MetricComparisons(coordinates=target_coords, maxk=maxk, n_jobs=1)
            mc_target.compute_distances()
            t_ranks = mc_target.dist_indices
        else:
            t_ranks = target_ranks

        # 3. Compute Imbalance
        # Row 1 is Delta(Source -> Target)
        coord_list = [list(range(source_coords.shape[1]))]
        imbalances = mc_source.return_inf_imb_target_selected_coords(
            target_ranks=t_ranks,
            coord_list=coord_list
        )
        return imbalances[1, 0]
    except Exception as e:
        # print(f"DEBUG: II error: {e}") 
        return np.nan

def _ii_relevance_worker(item, y_target_3d, n_samples, target_ranks_S, maxk):
    """
    Computes Delta(Residue -> S).
    Uses pre-computed ranks for S.
    """
    key, X_3d = item
    if not DADAPY_AVAILABLE: return (key, np.nan)
    
    X_i = X_3d.reshape(n_samples, -1)
    
    # Use pre-computed ranks for target S
    val = _compute_imbalance_raw(source_coords=X_i, target_ranks=target_ranks_S, maxk=maxk)
    return (key, val)

def _ii_redundancy_worker(item, all_features_static, n_samples, maxk):
    """
    Computes Delta(i->j) and Delta(j->i).
    Must compute ranks for both on the fly.
    """
    k_i, k_j = item
    if not DADAPY_AVAILABLE: return (k_i, k_j, np.nan, np.nan)
    
    X_i = all_features_static[k_i].reshape(n_samples, -1)
    X_j = all_features_static[k_j].reshape(n_samples, -1)
    
    # We need both directions.
    # Optimization: Compute dist_indices for i and j once?
    # Inside _compute_imbalance_raw, we instantiate MetricComparisons.
    # To suffice, we can manually do it to avoid re-calc.
    
    try:
        if maxk is None: maxk = n_samples - 1
        
        # Source I
        mc_i = MetricComparisons(coordinates=X_i, maxk=maxk, n_jobs=1)
        mc_i.compute_distances()
        ranks_i = mc_i.dist_indices
        
        # Source J
        mc_j = MetricComparisons(coordinates=X_j, maxk=maxk, n_jobs=1)
        mc_j.compute_distances()
        ranks_j = mc_j.dist_indices
        
        # Delta(i -> j)
        # Source: i, Target Ranks: j
        imb_ij = mc_i.return_inf_imb_target_selected_coords(
            target_ranks=ranks_j, coord_list=[list(range(X_i.shape[1]))]
        )[1, 0]
        
        # Delta(j -> i)
        # Source: j, Target Ranks: i
        imb_ji = mc_j.return_inf_imb_target_selected_coords(
            target_ranks=ranks_i, coord_list=[list(range(X_j.shape[1]))]
        )[1, 0]
        
        return (k_i, k_j, imb_ij, imb_ji)
        
    except Exception:
        return (k_i, k_j, np.nan, np.nan)


# --- Class ---

class QUBOSetII(BaseQUBO):
    """
    QUBO implementation using Information Imbalance (dadapy).
    """
    
    def _get_relevance_worker(self) -> Callable:
        return _ii_relevance_worker
        
    def _get_redundancy_worker(self) -> Callable:
        return _ii_redundancy_worker
        
    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        if not DADAPY_AVAILABLE:
            print("  FATAL: 'dadapy' library not found.")
            return {}
            
        maxk = kwargs.get('maxk', n_samples - 1)
        if maxk >= n_samples: maxk = n_samples - 1
        
        # Default return (overridden by run)
        return {'target_ranks_S': kwargs.get('target_ranks_S', None), 'maxk': maxk}

    def run(self, data, num_workers=None, **kwargs):
        """Override to pre-calculate Target S ranks."""
        all_features_static, _, mapping = data
        
        # Extract target S logic (duplicated slightly from BaseQUBO for prep)
        target_key = 'qubo_target_selection'
        if target_key in all_features_static:
            S = all_features_static[target_key]
            n_samples = S.shape[0]
            maxk = kwargs.get('maxk', n_samples - 1)
            
            if DADAPY_AVAILABLE:
                print(f"  Pre-computing ranks for Target S (N={n_samples})...")
                try:
                    mc = MetricComparisons(coordinates=S.reshape(n_samples, -1), maxk=maxk, n_jobs=1)
                    mc.compute_distances()
                    kwargs['target_ranks_S'] = mc.dist_indices
                except Exception as e:
                    print(f"  Warning: Failed to pre-compute S ranks: {e}")
        
        return super().run(data, num_workers, **kwargs)

    def _transform_relevance_to_h(self, imbalance: float) -> float:
        if np.isnan(imbalance): return 1e6
        # Imbalance [0,1]. h = Delta - 1.
        return imbalance - 1.0

    def _transform_redundancy_to_J(self, imb_ij: float, imb_ji: float, lam: float) -> float:
        if np.isnan(imb_ij) or np.isnan(imb_ji): return np.nan
        # Avg Imbalance. J = lambda * (1 - avg).
        avg_imb = 0.5 * (imb_ij + imb_ji)
        return lam * (1.0 - avg_imb)