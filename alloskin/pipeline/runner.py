"""
Core Analysis Pipeline Runner.
Refactored for "Entropic Decoy" vs "Silent Operator" Logic.
Includes Safe Handling for None/NaN values from JSON.
"""
import json
import numpy as np
from typing import Dict, Any, Optional

from alloskin.io.readers import MDAnalysisReader
from alloskin.features.extraction import FeatureExtractor
from alloskin.pipeline.builder import DatasetBuilder

from alloskin.analysis.static import StaticStateSensitivity
from alloskin.analysis.qubo import QUBOMaxCoverage
from alloskin.analysis.dynamic import TransferEntropy

def _safe_get(d: Dict, key: str, default: float = 0.0) -> float:
    """
    Safely retrieves a float value from a dictionary.
    Handles cases where the key is missing OR the value is None (from JSON null).
    """
    val = d.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def run_analysis(
    analysis_type: str,
    file_paths: Dict[str, str],
    params: Dict[str, Any],
    residue_selections: Optional[Dict[str, str]] = None,
    progress_callback: Optional[callable] = None
):
    def report(msg, pct):
        if progress_callback: progress_callback(msg, pct)
        else: print(f"[{pct}%] {msg}")

    report("Initializing Pipeline", 5)
    
    reader = MDAnalysisReader()
    extractor = FeatureExtractor(residue_selections)
    builder = DatasetBuilder(reader, extractor)

    report(f"Loading Data for {analysis_type}", 15)
    active_slice = params.get("active_slice")
    inactive_slice = params.get("inactive_slice")

    if analysis_type in ['static', 'qubo']:
        features_dict, labels_Y, mapping = builder.prepare_static_analysis_data(
            file_paths['active_traj'], file_paths['active_topo'],
            file_paths['inactive_traj'], file_paths['inactive_topo'],
            active_slice=active_slice, inactive_slice=inactive_slice
        )
    elif analysis_type == 'dynamic':
        features_act, features_inact, mapping = builder.prepare_dynamic_analysis_data(
            file_paths['active_traj'], file_paths['active_topo'],
            file_paths['inactive_traj'], file_paths['inactive_topo'],
             active_slice=active_slice, inactive_slice=inactive_slice
        )

    # --- Execution ---
    
    if analysis_type == 'static':
        report("Running Goal 1: Entropy & State Filter", 30)
        analyzer = StaticStateSensitivity()
        results = analyzer.run((features_dict, labels_Y), **params) 
        # Safe sort
        sorted_keys = sorted(results.keys(), key=lambda k: _safe_get(results[k], 'state_score'), reverse=True)
        final_output = {k: results[k] for k in sorted_keys}
        return final_output, mapping

    elif analysis_type == 'qubo':
        report("Running Goal 2: Hierarchical Dominating Set", 30)
        
        static_results_path = params.get('static_results_path')

        if static_results_path:
            report("Loading pre-computed static analysis results", 35)
            with open(static_results_path, 'r') as f:
                static_stats = json.load(f)
            
            report("Re-building dataset for selected candidates", 40)
            # If loading from file, the mapping might be inside the file structure
            # or we rely on the one we just built.
            # For consistency, we should use the mapping relevant to the loaded keys.
            loaded_mapping = static_stats.get('residue_selections_mapping')
            if loaded_mapping:
                 extractor = FeatureExtractor(residue_selections=loaded_mapping)
            
            # We must rebuild features to get the trajectory data for the QUBO calculation
            builder = DatasetBuilder(reader, extractor)
            features_dict, labels_Y, mapping = builder.prepare_static_analysis_data(
                file_paths['active_traj'], file_paths['active_topo'],
                file_paths['inactive_traj'], file_paths['inactive_topo'],
                active_slice=active_slice, inactive_slice=inactive_slice
            )
        else:
            report("Running Static pre-filter for candidates", 35)
            static_analyzer = StaticStateSensitivity()
            static_stats = static_analyzer.run((features_dict, labels_Y), **params)
        
        # --- DUAL-STREAM FILTERING LOGIC ---
        candidates = []
        candidate_state_scores = {} # JSD for QUBO weights
        
        # Params
        min_id = params.get('filter_min_id', 1.5)      # Discard Rocks
        top_total = params.get('filter_top_total', 100) # Total bandwidth
        top_jsd_guaranteed = params.get('filter_top_jsd', 20) # Guaranteed Switches
        
        report(f"Filtering: Top {top_jsd_guaranteed} JSD + Rest by Entropy (Total {top_total})", 60)
        
        static_results = static_stats['results'] if 'results' in static_stats else static_stats
        
        # 1. Identify Valid "Movers" (Remove Rocks)
        # FIX: Use _safe_get to handle None values from JSON
        movers = [
            k for k, v in static_results.items() 
            if _safe_get(v, 'id', 0.0) >= min_id
        ]
        
        # 2. Sort Movers by JSD (State Sensitivity)
        # FIX: Use _safe_get
        movers_by_jsd = sorted(
            movers, 
            key=lambda k: _safe_get(static_results[k], 'state_score', 0.0), 
            reverse=True
        )
        
        # 3. Select Guaranteed Switches (Stream A)
        selection_set = set()
        for k in movers_by_jsd[:top_jsd_guaranteed]:
            selection_set.add(k)
            
        # 4. Fill remaining slots with High Entropy residues (Stream B)
        # Sort ALL movers by ID to find the "Silent Operators"
        movers_by_id = sorted(
            movers, 
            key=lambda k: _safe_get(static_results[k], 'id', 0.0), 
            reverse=True
        )
        
        slots_remaining = top_total - len(selection_set)
        if slots_remaining > 0:
            for k in movers_by_id:
                if k not in selection_set:
                    selection_set.add(k)
                    slots_remaining -= 1
                    if slots_remaining == 0:
                        break
        
        candidates = list(selection_set)
        for k in candidates:
            # FIX: Use _safe_get
            candidate_state_scores[k] = _safe_get(static_results[k], 'state_score', 0.0)
            
        print(f"  Final Candidate Pool: {len(candidates)} residues.")
        
        if not candidates:
            raise ValueError("No candidate residues found. Try lowering filter_min_id.")

        candidate_features = {k: features_dict[k] for k in candidates}
        
        # 5. Run QUBO
        params['candidate_state_scores'] = candidate_state_scores
        
        report("Running QUBO analysis on filtered candidates", 80)
        qubo_analyzer = QUBOMaxCoverage()
        qubo_results = qubo_analyzer.run((candidate_features, None, None), **params)
        
        # --- POST-PROCESS: CLASSIFICATION ---
        if qubo_results.get('solutions'):
            best_sol = qubo_results['solutions'][0]
            selected_set = set(best_sol['residues'])
            
            classification = {}
            
            # Heuristic threshold for visualization
            vals = list(candidate_state_scores.values())
            max_jsd = max(vals) if vals else 1.0
            switch_threshold = 0.3 * max_jsd 
            
            for k in candidates:
                jsd = candidate_state_scores.get(k, 0.0)
                is_selected = k in selected_set
                
                if is_selected:
                    if jsd > switch_threshold:
                        cat = "Switch"
                    else:
                        cat = "Silent Operator"
                else:
                    cat = "Entropic Decoy"
                    
                classification[k] = {
                    "category": cat,
                    "jsd": jsd,
                    "selected": is_selected
                }
            
            qubo_results['classification'] = classification
            qubo_results['classification_threshold'] = switch_threshold

        return qubo_results, mapping

    elif analysis_type == 'dynamic':
        report("Running Goal 3: Dynamic Causality", 30)
        analyzer = TransferEntropy()
        res = analyzer.run((features_act, features_inact), **params)
        return res, mapping