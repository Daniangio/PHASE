"""
Implements Static Reporters analysis using Random Forest Classifiers.
(Backup Method)
"""

import numpy as np
from typing import Tuple, Dict, Any, Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Import the base class
try:
    from .components import BaseStaticReporter
except ImportError:
    # Fallback for direct execution/testing
    from alloskin.analysis.components import BaseStaticReporter


# --- Worker Function (Must remain top-level for pickling) ---

def _compute_rf_accuracy_worker(
    item: Tuple[str, np.ndarray], 
    labels_Y: np.ndarray, 
    n_samples: int, 
    cv_folds: int,
    n_estimators: int
) -> Tuple[str, float]:
    """Worker function for RF Classification."""
    res_key, features_3d = item
    
    if features_3d.shape[0] != n_samples:
        return (res_key, np.nan)

    try:
        features_2d = features_3d.reshape(n_samples, -1)
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=False,
            n_jobs=1, # Serial worker
            random_state=42
        )
        
        scores = cross_val_score(
            clf, 
            features_2d, 
            labels_Y, 
            cv=cv_folds, 
            scoring='accuracy', 
            n_jobs=1
        )
        
        return (res_key, np.mean(scores))

    except Exception as e:
        print(f"  Error in RF worker for {res_key}: {e}")
        return (res_key, np.nan)


# --- Specialized Class ---

class StaticReportersRF(BaseStaticReporter):
    """
    RF-based Static Reporters.
    Score = Mean Accuracy (Higher is better).
    """

    def _get_worker_function(self) -> Callable:
        return _compute_rf_accuracy_worker

    @property
    def _sort_reverse(self) -> bool:
        return True # Higher accuracy is better

    @property
    def _metric_name(self) -> str:
        return "Accuracy"

    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        """Configures CV folds and estimators."""
        
        if n_samples < 10:
             print(f"  FATAL: Too few samples ({n_samples}).")
             return {}

        cv_folds = kwargs.get('cv_folds', 5)
        n_estimators = kwargs.get('n_estimators', 100)
        
        if n_samples < cv_folds * 2:
            cv_folds = 3
            print(f"  Warning: Small sample size. Reducing CV folds to {cv_folds}.")
            
        return {
            'cv_folds': cv_folds,
            'n_estimators': n_estimators
        }