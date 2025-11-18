"""
Goal 2: The Static Atlas (Set Cover / Dominating Set).
Redefined to use Thresholded Logic and Intersection Penalties.
"""
import numpy as np
import concurrent.futures
from typing import Tuple, List, Dict, Any, Set
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

# --- Top-level worker functions for multiprocessing ---
# CRITICAL: These must be defined at the module level, not inside the class.

def _rank_worker_process(key: str, features_3d: np.ndarray, maxk: int) -> Tuple[str, np.ndarray]:
    """
    Computes distance ranks for a single residue's features.
    Top-level function to be picklable.
    """
    # Flatten features: (frames, 1, dim) -> (frames, dim)
    n_samples = features_3d.shape[0]
    X = features_3d.reshape(n_samples, -1)
    
    # Dadapy calculation
    mc = MetricComparisons(coordinates=X, maxk=maxk, n_jobs=1)
    mc.compute_distances()
    
    return key, mc.dist_indices

def _imbalance_row_process(
    i: int, 
    keys: List[str], 
    features_source: np.ndarray, 
    cached_ranks_target: Dict[str, np.ndarray], 
    maxk: int
) -> Tuple[int, np.ndarray]:
    """
    Computes a single row of the information imbalance matrix.
    Top-level function to be picklable.
    """
    # Flatten source
    n_samples = features_source.shape[0]
    X_source = features_source.reshape(n_samples, -1)
    
    # Re-compute distances for source (Dadapy objects aren't easily picklable)
    mc_source = MetricComparisons(coordinates=X_source, maxk=maxk, n_jobs=1)
    mc_source.compute_distances()
    
    row_vals = []
    # We iterate over all targets to calculate Delta(Source -> Target)
    for j, key_target in enumerate(keys):
        if i == j:
            row_vals.append(0.0)
            continue
        
        target_ranks = cached_ranks_target[key_target]
        
        # Calculate Information Imbalance
        # return_inf_imb... returns [[delta, error], [delta_scaled, error]]
        # We take [1,0] which is the scaled imbalance (or [0,0] for raw)
        # Usually index 1 (scaled) is safer if maxk varies, but index 0 is standard Delta.
        # Let's use index 0 (raw Delta) as per standard definition 
        imb_res = mc_source.return_inf_imb_target_selected_coords(
            target_ranks=target_ranks, 
            coord_list=[list(range(X_source.shape[1]))]
        )
        # imb_res shape is (N_coords, 2). coord_list has 1 element.
        # So we want imb_res[0, 0] (the value)
        row_vals.append(imb_res[0, 0])
        
    return i, np.array(row_vals)

class QUBOMaxCoverage(BaseQUBO):
    
    def _compute_interaction_matrix(self, features: FeatureDict, num_workers: int = None, **kwargs) -> Tuple[List[str], np.ndarray]:
        """
        Computes the raw All-vs-All Information Imbalance matrix.
        """
        if not DADAPY_AVAILABLE:
            raise ImportError("dadapy required for QUBO.")

        keys = sorted(list(features.keys()))
        N = len(keys)
        n_samples = features[keys[0]].shape[0]
        maxk = kwargs.get('maxk', min(n_samples - 1, 100))
        
        matrix = np.full((N, N), np.nan)
        
        print(f"  Pre-computing ranks for {N} candidates (N_samples={n_samples})...")
        
        # 1. Pre-compute Ranks (Heavy Lifting)
        cached_ranks = {}
        
        # Use ProcessPoolExecutor to parallelize
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Prepare arguments for mapping
            # We pass individual feature arrays to avoid pickling the whole dict every time
            features_list = [features[k] for k in keys]
            maxk_list = [maxk] * len(keys)
            
            # executor.map preserves order
            results = executor.map(_rank_worker_process, keys, features_list, maxk_list)
            
            for key, ranks in results:
                cached_ranks[key] = ranks
        
        print("  Calculating All-vs-All Imbalance Matrix...")

        # 2. Compute Imbalances using cached ranks
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(N):
                key_source = keys[i]
                feat_source = features[key_source]
                
                # Submit task
                # Note: We pass the specific source feature and the FULL dictionary of target ranks.
                # Dictionary lookup is fast, pickling a dict of numpy arrays is moderate overhead 
                # but better than re-computing.
                futures.append(
                    executor.submit(
                        _imbalance_row_process, 
                        i, 
                        keys, 
                        feat_source, 
                        cached_ranks, 
                        maxk
                    )
                )
            
            for f in concurrent.futures.as_completed(futures):
                try:
                    i, row = f.result()
                    matrix[i, :] = row
                except Exception as e:
                    print(f"    Error in row {i}: {e}")

        return keys, matrix

    def run(self, data, num_workers=None, **kwargs):
        
        features_dict = data[0]
        
        # 1. Compute Raw Matrix
        keys, raw_matrix = self._compute_interaction_matrix(features_dict, num_workers=num_workers, **kwargs)
        N = len(keys)

        # 2. Pre-process: Thresholding and Set Construction
        ii_threshold = kwargs.get('ii_threshold', 0.4)
        print(f"  Applying hard threshold: II < {ii_threshold:.2f} implies 'Covered'")

        # Domains D(i): The set of indices j that i covers
        domains: Dict[int, Set[int]] = {}
        
        for i in range(N):
            covered_indices = set()
            for j in range(N):
                if i == j: continue
                # Check for NaN
                val = raw_matrix[i, j]
                if np.isnan(val): continue
                
                if val < ii_threshold:
                    covered_indices.add(j)
            domains[i] = covered_indices

        # 3. Compute Switch Scores (JSD)
        state_scores = kwargs.get('candidate_state_scores', {})
        switch_scores = np.zeros(N)
        
        if state_scores:
            print("  Incorporating State Prediction scores...")
            for i, key in enumerate(keys):
                switch_scores[i] = state_scores.get(key, 0.0)

        # 4. Build Hamiltonian
        alpha = kwargs.get('alpha_size', 1.0)       # Cost of adding a residue
        beta_hub = kwargs.get('beta_hub', 1.0)      # Reward per covered child
        beta_switch = kwargs.get('beta_switch', 5.0)# Reward for predicting State
        gamma = kwargs.get('gamma_redundancy', 2.0) # Penalty per shared child
        
        import pyqubo, neal

        h_linear = {}
        J_quadratic = {}

        print("  Building Hamiltonian (Intersection Logic)...")

        for i in range(N):
            key_i = keys[i]
            
            # --- Linear Term (h_i) ---
            h_val = alpha 
            h_val -= beta_switch * switch_scores[i]
            h_val -= beta_hub * len(domains[i])
            
            h_linear[key_i] = h_val

            # --- Quadratic Term (J_ij) ---
            for j in range(i + 1, N):
                key_j = keys[j]
                
                # Intersection of children
                shared_children = domains[i].intersection(domains[j])
                n_shared = len(shared_children)
                
                # Direct redundancy
                direct_redundancy = 0
                if j in domains[i]: direct_redundancy += 1
                if i in domains[j]: direct_redundancy += 1
                
                penalty = gamma * (n_shared + direct_redundancy)
                
                if penalty > 0:
                    if key_i not in J_quadratic: J_quadratic[key_i] = {}
                    J_quadratic[key_i][key_j] = penalty

        # 5. Solve
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
            response = sampler.sample_qubo(Q, num_reads=kwargs.get('num_reads', 1000))
            
            solutions_list = []
            seen = set()
            for record in response.record:
                energy = record['energy'] + offset
                sol_tuple = tuple(record['sample'])
                
                if sol_tuple not in seen:
                    seen.add(sol_tuple)
                    selected = [response.variables[i] for i, bit in enumerate(record['sample']) if bit == 1]
                    
                    # Calculate stats
                    sel_indices = [keys.index(k) for k in selected]
                    total_covered = set()
                    for idx in sel_indices:
                        total_covered.update(domains[idx])
                    
                    solutions_list.append({
                        "energy": float(energy),
                        "size": len(selected),
                        "unique_coverage": len(total_covered),
                        "residues": selected
                    })
                    if len(solutions_list) >= kwargs.get('num_solutions', 5): break
            
            return {
                "solutions": solutions_list,
                "matrix_indices": keys,
                "parameters": {
                    "alpha": alpha, "beta_hub": beta_hub, 
                    "beta_switch": beta_switch, "gamma": gamma,
                    "threshold": ii_threshold
                }
            }

        except Exception as e:
            print(f"  Error solving QUBO: {e}")
            return {"error": str(e)}