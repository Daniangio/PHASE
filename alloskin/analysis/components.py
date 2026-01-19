"""
Analysis Components and Base Classes.
Refactored to support the Hierarchical Information Atlas pipeline.
"""

import numpy as np
import os
import concurrent.futures
import functools
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Callable

FeatureDict = Dict[str, np.ndarray]

class AnalysisComponent(ABC):
    """Abstract base class for all analysis components."""
    @abstractmethod
    def run(self, data, **kwargs):
        pass


# --- GOAL 1: Static Atlas Filter (Entropy + State) ---

class BaseStaticReporter(AnalysisComponent):
    """
    Base class for Static Analysis (Goal 1).
    Now supports returning multiple metrics (e.g., ID and Imbalance).
    """

    @abstractmethod
    def _get_worker_function(self) -> Callable:
        pass

    @abstractmethod
    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        pass

    def run(self, 
            data: Tuple[FeatureDict, np.ndarray], 
            num_workers: int = None, 
            **kwargs
        ) -> Dict[str, Dict[str, float]]:
        
        method_name = self.__class__.__name__
        print(f"\n--- Running {method_name} ---")
        
        max_workers = num_workers if num_workers is not None else min(32, os.cpu_count() // 2)
        all_features_static, labels_Y = data
        results: Dict[str, Dict[str, float]] = {}
        
        if not all_features_static:
            print("  Warning: No features found to analyze.")
            return results

        n_samples = labels_Y.shape[0]
        worker_params = self._prepare_worker_params(n_samples, **kwargs)
        
        worker_func = functools.partial(
            self._get_worker_function(),
            labels_Y=labels_Y,
            n_samples=n_samples,
            **worker_params
        )
        
        print(f"  Analyzing {len(all_features_static)} residues with {max_workers} workers...")
        
        # Execution Strategy
        if max_workers is not None and max_workers <= 1:
            for item in list(all_features_static.items()):
                res_key, metrics = worker_func(item)
                results[res_key] = metrics
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {executor.submit(worker_func, item): item[0] for item in all_features_static.items()}
                for future in concurrent.futures.as_completed(future_to_key):
                    res_key = future_to_key[future]
                    try:
                        key_out, metrics = future.result()
                        results[res_key] = metrics
                    except Exception as exc:
                        print(f"  Error analyzing {res_key}: {exc}")
                        results[res_key] = {"error": np.nan}

        print(f"{method_name} complete.")
        return results
