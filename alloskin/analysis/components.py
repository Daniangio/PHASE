"""
Analysis Components and Base Classes.

This module defines the interfaces and common base classes for
the analysis pipeline. It implements the Template Method pattern
to maximize code reuse and extensibility.
"""

import numpy as np
import os
import concurrent.futures
import functools
import multiprocessing
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Callable, List

# --- Optional Imports for BaseQUBO ---
try:
    import MDAnalysis as mda
    from MDAnalysis.core.selection import SelectionError
    from MDAnalysis.core.groups import AtomGroup
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


# --- GOAL 1: Static Reporter Base ---

class BaseStaticReporter(AnalysisComponent):
    """
    Base class for Static Reporter analysis.
    Handles parallelization over N residues.
    """

    @abstractmethod
    def _get_worker_function(self) -> Callable:
        pass

    @abstractmethod
    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def _sort_reverse(self) -> bool:
        pass

    @property
    @abstractmethod
    def _metric_name(self) -> str:
        pass

    def run(self, 
            data: Tuple[FeatureDict, np.ndarray], 
            num_workers: int = None, 
            **kwargs
        ) -> Dict[str, float]:
        
        method_name = self.__class__.__name__
        metric = self._metric_name
        print(f"\n--- Running {method_name} ---")
        
        max_workers = num_workers if num_workers is not None else os.cpu_count()
        print(f"Using max {max_workers or 'all'} workers for analysis.")
            
        all_features_static, labels_Y = data
        scores: Dict[str, float] = {}
        
        if not all_features_static:
            print("  Warning: No features found to analyze.")
            return scores

        n_samples = labels_Y.shape[0]
        worker_params = self._prepare_worker_params(n_samples, **kwargs)
        
        if not worker_params:
            return scores

        print(f"Calculating {metric} for {len(all_features_static)} residues...")
        
        worker_func = functools.partial(
            self._get_worker_function(),
            labels_Y=labels_Y,
            n_samples=n_samples,
            **worker_params
        )
        
        # --- Serial vs Parallel Execution ---
        if max_workers is not None and max_workers <= 1:
            print("  Running in SERIAL mode (num_workers <= 1).")
            # Force single-threaded loop for debugging
            for item in list(all_features_static.items()):
                res_key, score = worker_func(item)
                scores[res_key] = score
                if not np.isnan(score):
                    print(f"  {metric}( {res_key} ) = {score:.4f}")
                else:
                    print(f"  {metric}( {res_key} ) = FAILED")
        else:
            print(f"  Running in PARALLEL mode ({max_workers} workers).")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_key = {executor.submit(worker_func, item): item[0] for item in all_features_static.items()}
                
                # Process as they complete
                for future in concurrent.futures.as_completed(future_to_key):
                    res_key = future_to_key[future]
                    try:
                        key_out, score = future.result()
                        scores[res_key] = score
                        if not np.isnan(score):
                            print(f"  {metric}( {res_key} ) = {score:.4f}")
                        else:
                            print(f"  {metric}( {res_key} ) = FAILED")
                    except Exception as exc:
                        print(f"  {metric}( {res_key} ) generated an exception: {exc}")
                        scores[res_key] = np.nan

        default_val = float('-inf') if self._sort_reverse else float('inf')

        sorted_results = dict(sorted(
            scores.items(),
            key=lambda item: item[1] if not np.isnan(item[1]) else default_val,
            reverse=self._sort_reverse
        ))

        print(f"{method_name} analysis complete.")
        return sorted_results


# --- GOAL 2: QUBO Base ---

class BaseQUBO(AnalysisComponent):
    """
    Base class for QUBO analysis (Feature Selection).
    """
    
    @abstractmethod
    def _get_relevance_worker(self) -> Callable:
        pass
        
    @abstractmethod
    def _get_redundancy_worker(self) -> Callable:
        pass

    @abstractmethod
    def _prepare_worker_params(self, n_samples: int, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def _transform_relevance_to_h(self, raw_score: float) -> float:
        pass
        
    @abstractmethod
    def _transform_redundancy_to_J(self, val_ij: float, val_ji: float, lam: float) -> float:
        pass

    def run(self, 
            data: Tuple[FeatureDict, np.ndarray, Dict[str, str]], 
            num_workers: int = None,
            **kwargs
        ) -> Dict[str, Any]:
        
        print(f"\n--- Running {self.__class__.__name__} ---")
        
        all_features_static, _, mapping = data
        max_workers = num_workers if num_workers is not None else os.cpu_count()
        
        target_sel = kwargs.get('target_selection_string')
        topo_file = kwargs.get('active_topo_file')
        lambda_redundancy = float(kwargs.get('lambda_redundancy', 1.0))
        num_solutions = int(kwargs.get('num_solutions', 5))
        
        if not all_features_static or not mapping: return {}
        if not target_sel or not topo_file: raise ValueError("QUBO requires target and topo.")

        n_samples = next(iter(all_features_static.values())).shape[0]
        worker_params = self._prepare_worker_params(n_samples, **kwargs)
        if not worker_params: return {}

        # 2. Parse Selections
        print(f"Resolving selections using {topo_file}...")
        try:
            u = mda.Universe(topo_file)
            target_ag = u.select_atoms(target_sel)
            if target_ag.n_atoms == 0: raise ValueError("Target selection matched 0 atoms.")
        except Exception as e:
            raise ValueError(f"Selection error: {e}")

        candidate_keys = []
        excluded_keys = set()
        for key, sel_str in mapping.items():
            if key not in all_features_static: continue
            try:
                input_ag = u.select_atoms(sel_str)
                if AtomGroup.intersection(target_ag, input_ag).n_residues > 0:
                    excluded_keys.add(key)
                else:
                    candidate_keys.append(key)
            except:
                continue 
        
        candidate_keys = sorted(candidate_keys)
        target_key = 'qubo_target_selection'
        
        if target_key not in all_features_static:
             raise ValueError(f"Target feature '{target_key}' not found.")
             
        target_S = all_features_static[target_key]
        print(f"  Candidates: {len(candidate_keys)}. Target Shape: {target_S.shape}")

        # 3. Compute Relevance (h_i)
        print("Computing Relevance...")
        h_i_terms = {}
        
        relevance_func = functools.partial(
            self._get_relevance_worker(),
            y_target_3d=target_S,
            n_samples=n_samples,
            **worker_params
        )
        
        items = [(k, all_features_static[k]) for k in candidate_keys]
        
        if max_workers is not None and max_workers <= 1:
            results = map(relevance_func, items)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(relevance_func, items))
        
        for key, score in results:
            h_i = self._transform_relevance_to_h(score)
            h_i_terms[key] = h_i

        # 4. Compute Redundancy (J_ij)
        print("Computing Redundancy...")
        J_ij_terms = {}
        
        redundancy_jobs = []
        for i in range(len(candidate_keys)):
            for j in range(i + 1, len(candidate_keys)):
                redundancy_jobs.append((candidate_keys[i], candidate_keys[j]))
        
        redundancy_func = functools.partial(
            self._get_redundancy_worker(),
            all_features_static=all_features_static,
            n_samples=n_samples,
            **worker_params
        )
        
        if max_workers is not None and max_workers <= 1:
             results = map(redundancy_func, redundancy_jobs)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(redundancy_func, redundancy_jobs))
        
        for key_i, key_j, s_ij, s_ji in results:
            J_val = self._transform_redundancy_to_J(s_ij, s_ji, lambda_redundancy)
            if not np.isnan(J_val):
                if key_i not in J_ij_terms: J_ij_terms[key_i] = {}
                J_ij_terms[key_i][key_j] = J_val

        # 5. Build & Solve Hamiltonian
        print("Solving QUBO...")
        try:
            x_vars = {k: pyqubo.Binary(k) for k in candidate_keys}
            H = 0.0
            
            for k, h_val in h_i_terms.items():
                if not np.isnan(h_val): H += h_val * x_vars[k]
            
            for k_i, partners in J_ij_terms.items():
                for k_j, J_val in partners.items():
                    H += J_val * x_vars[k_i] * x_vars[k_j]
            
            model = H.compile()
            Q, offset = model.to_qubo()
            
            sampler = neal.SimulatedAnnealingSampler()
            response = sampler.sample_qubo(Q, num_reads=max(num_solutions*20, 100))
            
            solutions_list = []
            seen = set()
            for record in response.record:
                energy = record['energy'] + offset
                if energy >= 0 or len(solutions_list) >= num_solutions: continue
                
                sol_tuple = tuple(record['sample'])
                if sol_tuple not in seen:
                    seen.add(sol_tuple)
                    selected = [response.variables[i] for i, bit in enumerate(record['sample']) if bit == 1]
                    solutions_list.append({
                        "energy": energy,
                        "selected_residues": selected,
                        "num_occurrences": int(record['num_occurrences'])
                    })
                    
            print(f"Found {len(solutions_list)} solutions.")
            return {
                "analysis_type": self.__class__.__name__,
                "parameters": kwargs,
                "solutions": solutions_list,
                "hamiltonian": {"h": h_i_terms, "J": J_ij_terms}
            }
            
        except Exception as e:
            print(f"  Error during PyQUBO/Neal solve: {e}")
            return {"error": str(e)}