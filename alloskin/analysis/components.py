"""
Analysis Components and Base Classes.
Refactored to support the Hierarchical Information Atlas pipeline.
"""

import numpy as np
import os
import concurrent.futures
import functools
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Callable, List, Union

# --- Optional Imports ---
try:
    import MDAnalysis as mda
    import pyqubo
    import neal
except ImportError:
    pass

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


# --- GOAL 2: Maximum Coverage QUBO ---

class BaseQUBO(AnalysisComponent):
    """
    Base class for QUBO Analysis (Goal 2).
    Refactored for the 'Maximum Coverage' All-vs-All formulation.
    """
    
    @abstractmethod
    def _compute_interaction_matrix(self, features: FeatureDict, **kwargs) -> Tuple[List[str], np.ndarray]:
        """
        Must return:
        1. List of keys (residue names) corresponding to matrix indices.
        2. The NxN matrix where M[i,j] = Delta(i -> j).
        """
        pass

    def run(self, 
            data: Tuple[FeatureDict, Any, Any], 
            num_workers: int = None,
            **kwargs
        ) -> Dict[str, Any]:
        
        print(f"\n--- Running {self.__class__.__name__} (Maximum Coverage) ---")
        
        # Unpack data (Goal 2 only needs the features dictionary)
        # We assume the runner has already filtered 'data' to the top candidates
        features_dict = data[0] 
        
        # 1. Compute All-vs-All Information Imbalance Matrix
        print("  Computing All-vs-All Information Matrix...")
        keys, interaction_matrix = self._compute_interaction_matrix(features_dict, num_workers=num_workers, **kwargs)
        
        N = len(keys)
        print(f"  Matrix shape: {interaction_matrix.shape} (for {N} candidates)")

        # 2. Build Hamiltonian
        # H(x) = alpha * sum(x) - beta * sum(coverage) + gamma * sum(redundancy)
        alpha = kwargs.get('alpha_size', 1.0)
        beta = kwargs.get('beta_coverage', 10.0)
        gamma = kwargs.get('gamma_redundancy', 5.0)
        num_solutions = kwargs.get('num_solutions', 5)

        print(f"  Building Hamiltonian: alpha={alpha}, beta={beta}, gamma={gamma}")
        
        h_linear = {} # Linear terms (h_i)
        J_quadratic = {} # Quadratic terms (J_ij)

        for i in range(N):
            key_i = keys[i]
            
            # --- Term 1: Penalty for Size (alpha) ---
            # Adds +alpha to energy for every selected residue
            h_val = alpha
            
            # --- Term 2: Reward for Coverage (beta) ---
            # We want to select 'i' if it predicts many 'j's.
            # Gain energy: -beta * sum_{j!=i} (1 - Delta(i->j))
            # (1 - Delta) is the "Explanation Power". 1 is perfect, 0 is none.
            explanation_power = 0.0
            for j in range(N):
                if i == j: continue
                delta_i_j = interaction_matrix[i, j]
                if not np.isnan(delta_i_j):
                    explanation_power += (1.0 - delta_i_j)
            
            h_val -= beta * explanation_power
            h_linear[key_i] = h_val

            # --- Term 3: Penalty for Redundancy (gamma) ---
            # Only iterate j > i for upper triangle
            for j in range(i + 1, N):
                key_j = keys[j]
                
                # Calculate symmetric similarity
                d_ij = interaction_matrix[i, j]
                d_ji = interaction_matrix[j, i]
                
                if np.isnan(d_ij) or np.isnan(d_ji): continue
                
                avg_imbalance = 0.5 * (d_ij + d_ji)
                symmetric_similarity = 1.0 - avg_imbalance
                
                # Penalty: +gamma * similarity
                # If they are identical (sim=1), high penalty.
                J_val = gamma * symmetric_similarity
                
                if key_i not in J_quadratic: J_quadratic[key_i] = {}
                J_quadratic[key_i][key_j] = J_val

        # 3. Solve
        print("  Solving QUBO with SA...")
        try:
            x_vars = {k: pyqubo.Binary(k) for k in keys}
            H = 0.0
            
            for k, val in h_linear.items():
                H += val * x_vars[k]
            
            for k1, partners in J_quadratic.items():
                for k2, val in partners.items():
                    H += val * x_vars[k1] * x_vars[k2]

            model = H.compile()
            Q, offset = model.to_qubo()
            
            sampler = neal.SimulatedAnnealingSampler()
            response = sampler.sample_qubo(Q, num_reads=max(num_solutions*50, 200))
            
            solutions_list = []
            seen = set()
            for record in response.record:
                energy = record['energy'] + offset
                sol_tuple = tuple(record['sample'])
                
                if sol_tuple not in seen:
                    seen.add(sol_tuple)
                    selected = [response.variables[i] for i, bit in enumerate(record['sample']) if bit == 1]
                    solutions_list.append({
                        "energy": float(energy),
                        "size": len(selected),
                        "residues": selected
                    })
                    if len(solutions_list) >= num_solutions: break
            
            return {
                "solutions": solutions_list,
                "matrix_indices": keys,
                # "interaction_matrix": interaction_matrix.tolist() # Optional: save if needed
            }

        except Exception as e:
            print(f"  Error solving QUBO: {e}")
            return {"error": str(e)}