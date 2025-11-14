"""
Implements Static Reporters analysis.

Refactored to use Random Forest Classifiers instead of Information Imbalance.

This approach directly answers: "How well can this single residue's
features classify the protein state?"

- Uses concurrent.futures to parallelize training N classifiers,
  one for each residue.
- The "score" is the mean cross-validated accuracy.
- Higher score is better.
"""

import numpy as np
import os
import concurrent.futures
import functools
from typing import Tuple, Dict, Any, List

# --- New Imports ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler  # Optional: Good practice
# --- Removed dadapy imports ---

FeatureDict = Dict[str, np.ndarray]

# --- Worker Function (MUST be top-level) ---

def _compute_rf_accuracy_worker(
    item: Tuple[str, np.ndarray], 
    labels_Y: np.ndarray, 
    n_samples: int, 
    cv_folds: int,
    n_estimators: int
) -> Tuple[str, float]:
    """
    Worker function for parallel Random Forest classification.
    
    Trains a simple RF classifier on a *single* residue's
    features (e.g., 6 sin/cos values) to predict the state.
    
    Args:
        item: A (res_key, features_3d) tuple.
        labels_Y: The global (n_samples,) label array (0=inactive, 1=active).
        n_samples: Total number of samples (frames).
        cv_folds: Number of folds for cross-validation.
        n_estimators: Number of trees in the forest.
        
    Returns:
        A (res_key, mean_cv_accuracy) tuple.
    """
    res_key, features_3d = item
    
    if features_3d.shape[0] != n_samples:
        print(f"  Warning: Mismatch in frames for {res_key}. Skipping.")
        return (res_key, np.nan)

    try:
        # Reshape (n_frames, 1, 6) -> (n_frames, 6)
        features_2d = features_3d.reshape(n_samples, -1)
        
        # Optional: Scaling features. RFs are not sensitive to
        # feature magnitude, but it doesn't hurt.
        # scaler = StandardScaler()
        # features_2d = scaler.fit_transform(features_2d)
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=False,      # We are using cross-validation
            n_jobs=1,             # CRITICAL: The worker process must be serial
            random_state=42
        )
        
        # Use cross-validation for a robust estimate of classifier performance
        # n_jobs=1 is also critical here for the same reason.
        scores = cross_val_score(
            clf, 
            features_2d, 
            labels_Y, 
            cv=cv_folds, 
            scoring='accuracy', 
            n_jobs=1
        )
        
        # The score is the mean accuracy
        mean_accuracy = np.mean(scores)
        
        return (res_key, mean_accuracy)

    except Exception as e:
        print(f"  Error in RF worker for {res_key}: {e}")
        return (res_key, np.nan)

# --- End Worker Function ---


class AnalysisComponent:
    def run(self, data, **kwargs):
        raise NotImplementedError

class StaticReportersRF(AnalysisComponent):
    """
    Implements Static Reporters analysis using per-residue
    Random Forest classifiers.
    
    Higher accuracy score means the residue is a better
    classifier (and thus "reporter") of the state.
    """

    def run(self, 
            data: Tuple[FeatureDict, np.ndarray], 
            num_workers: int = None, 
            **kwargs
        ) -> Dict[str, float]:
        """
        Runs the per-residue RF classification in parallel.

        Args:
            data: A tuple from prepare_static_analysis_data
                  (all_features_static, labels_Y)
                  - all_features_static: Dict {res_key: (n_frames, 1, 6) array}
                  - labels_Y: (n_frames,) array
            num_workers (int, optional): Number of parallel processes. 
                                         Defaults to os.cpu_count().
            **kwargs: Can include `cv_folds` (default 5) or 
                      `n_estimators` (default 100).

        Returns:
            A dictionary of {res_key: accuracy_score}, sorted by score
            (highest is best).
        """
        print("\n--- Running Static Reporters (Random Forest) ---")
        
        max_workers = num_workers if num_workers is not None else None
        print(f"Using max {max_workers or 'all'} workers for analysis.")
            
        all_features_static, labels_Y = data
        scores: Dict[str, float] = {}
        
        if not all_features_static:
            print("  Warning: No features found to analyze.")
            return scores

        # --- Setup for Parallel RFs ---
        n_samples = labels_Y.shape[0]
        
        # Get analysis parameters, with defaults
        cv_folds = kwargs.get('cv_folds', 5)
        n_estimators = kwargs.get('n_estimators', 100)
        
        if n_samples < 10:
             print(f"  FATAL Error: Very few samples ({n_samples}). Cannot run analysis with less than 10 samples.")
             return scores
        if n_samples < cv_folds * 2:
            cv_folds = 3 # Adjust CV folds for small sample sizes
            print(f"  Warning: Small sample size. Reducing CV folds to {cv_folds}.")


        print(f"Calculating RF accuracy for {len(all_features_static)} residues in parallel...")
        print(f" (Params: {cv_folds}-fold CV, {n_estimators} trees)")

        # 2. Create a partial function with the shared arguments
        worker_func = functools.partial(
            _compute_rf_accuracy_worker,
            labels_Y=labels_Y,
            n_samples=n_samples,
            cv_folds=cv_folds,
            n_estimators=n_estimators
        )
        
        # 3. Run the parallel computation
        # We MUST use ProcessPoolExecutor for CPU-bound sklearn tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            
            results_iterator = executor.map(worker_func, all_features_static.items())
            
            # 4. Collect results from the iterator
            for res_key, acc_score in results_iterator:
                scores[res_key] = acc_score
                if not np.isnan(acc_score):
                    print(f"  Accuracy( {res_key} ) = {acc_score:.4f}")
                else:
                    print(f"  Accuracy( {res_key} ) = FAILED")

        # 5. Sort final results
        # We sort in REVERSE order (high accuracy is best)
        sorted_results = dict(sorted(
            scores.items(),
            key=lambda item: item[1] if not np.isnan(item[1]) else float('-inf'),
            reverse=True
        ))

        print("Static Reporters (RF) analysis complete.")
        return sorted_results