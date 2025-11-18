"""
Implements Static Reporters analysis using Information Imbalance.
(Default Method)

Refactored to use 'dadapy.metric_comparisons' for rank-based calculation.
"""
import os
# Set these BEFORE importing numpy/dadapy/sklearn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from typing import Tuple, Dict, Any, Callable

# dadapy import
try:
    from dadapy.metric_comparisons import MetricComparisons
    DADAPY_AVAILABLE = True
except ImportError:
    DADAPY_AVAILABLE = False

# Import the base class
try:
    from .components import BaseStaticReporter
except ImportError:
    from alloskin.analysis.components import BaseStaticReporter


# --- Worker Function ---

def _compute_ii_worker(
    item: Tuple[str, np.ndarray], 
    labels_Y: np.ndarray,
    n_samples: int,
    target_ranks: np.ndarray, 
    maxk: int
) -> Tuple[str, float]:
    """
    Worker function for Information Imbalance using MetricComparisons.
    Computes Delta(Residue -> TargetLabels).
    
    Args:
        item: (key, features)
        labels_Y: The raw labels (Ignored here, we use target_ranks)
        n_samples: Number of samples (Ignored, derived from target_ranks)
        target_ranks: Pre-computed ranks of the target space.
        maxk: maxk used for ID estimation.
    """
    res_key, features_3d = item
    
    if not DADAPY_AVAILABLE:
        return (res_key, np.nan)
    
    # Derived n_samples from ranks to be safe
    n_samples_ranks = target_ranks.shape[0]
    if features_3d.shape[0] != n_samples_ranks:
        return (res_key, np.nan)

    try:
        # Reshape features to (n_samples, n_features)
        features_2d = features_3d.reshape(n_samples_ranks, -1)
        
        # 1. Initialize MetricComparisons for the Source (Residue Features)
        mc_features = MetricComparisons(coordinates=features_2d, maxk=maxk, n_jobs=1)
        
        # 2. Compute distances/ranks for the source
        mc_features.compute_distances()
        
        # 3. Define coordinates to test (all of them)
        n_dims = features_2d.shape[1]
        coord_list = [list(range(n_dims))]
        
        # 4. Compute Imbalance: Source (Features) -> Target (Labels)
        imbalances = mc_features.return_inf_imb_target_selected_coords(
            target_ranks=target_ranks,
            coord_list=coord_list
        )
        
        # imbalances[0, 0] tells how protein state y (0 or 1) is predictive of input. This is ill posed as the ranks in y are random.
        # In fact, the neighbours of a set where all values are 0 (or 1) are random picked based on algorithm implementation order.
        # Still, we don't care as we are interested in imbalances[1, 0], which tell us how predictive are our features of the protein state
        delta_res_to_labels = imbalances[1, 0]
        return (res_key, delta_res_to_labels)

    except Exception as e:
        print(f"  Error in dadapy worker for {res_key}: {e}")
        return (res_key, np.nan)


# --- Specialized Class ---

class StaticReportersII(BaseStaticReporter):
    """
    II-based Static Reporters using dadapy.metric_comparisons.
    Score = Information Imbalance (Lower is better).
    """

    def _get_worker_function(self) -> Callable:
        return _compute_ii_worker

    @property
    def _sort_reverse(self) -> bool:
        return False # Lower imbalance is better

    @property
    def _metric_name(self) -> str:
        return "Imbalance"

    def run(self, data: Tuple[Dict[str, np.ndarray], np.ndarray], num_workers: int = None, **kwargs) -> Dict[str, float]:
        """
        Overridden run to handle label rank pre-calculation.
        """
        all_features_static, labels_Y = data
        
        if not DADAPY_AVAILABLE:
            print("  FATAL: 'dadapy' library not found.")
            return {}
            
        n_samples = labels_Y.shape[0]
        # Use a safer default for maxk (e.g., 100 or n_samples - 1)
        # Calculating ALL ranks (n-1) is expensive O(N^2). 
        # 100 is usually sufficient for local ID.
        maxk = kwargs.get('maxk', min(n_samples - 1, 100))
        
        print(f"  Pre-computing target ranks for labels (N={n_samples}, maxk={maxk})...")
        try:
            # Reshape labels to (N, 1)
            labels_2d = labels_Y.reshape(-1, 1)
            
            mc_labels = MetricComparisons(coordinates=labels_2d, maxk=maxk, n_jobs=1)
            mc_labels.compute_distances(n_jobs=1)
            self._cached_target_ranks = mc_labels.dist_indices
        except Exception as e:
            print(f"  Error computing label ranks: {e}")
            return {}

        # Pass pre-computed values via kwargs to _prepare_worker_params
        kwargs['target_ranks'] = self._cached_target_ranks
        kwargs['maxk'] = maxk
        
        return super().run(data, num_workers=num_workers, **kwargs)

    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        """Passes the pre-calculated ranks to the worker."""
        return {
            'target_ranks': kwargs['target_ranks'],
            'maxk': kwargs['maxk']
        }