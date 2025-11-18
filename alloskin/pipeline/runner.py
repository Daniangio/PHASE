"""
Core Analysis Pipeline Runner

This module contains the high-level logic for executing an AllosKin analysis.
It is designed to be called by different entry points, such as the CLI or
a background task worker, ensuring that the core execution path is consistent
and centralized.
"""

from typing import Dict, Any, Optional

from alloskin.io.readers import MDAnalysisReader
from alloskin.features.extraction import FeatureExtractor
from alloskin.pipeline.builder import DatasetBuilder
from alloskin.analysis.static_rf import StaticReportersRF
from alloskin.analysis.static_ii import StaticReportersII
from alloskin.analysis.qubo_rf import QUBOSetRF
from alloskin.analysis.qubo_ii import QUBOSetII
from alloskin.analysis.dynamic import TransferEntropy


def get_builder(
    residue_selections: Optional[Dict[str, str]] = None
) -> DatasetBuilder:
    """
    Initializes the core components (Reader, Extractor, Builder).

    Args:
        residue_selections: A dictionary of residue selections, or None to
                            analyze all protein residues.

    Returns:
        An initialized DatasetBuilder instance.
    """
    if residue_selections:
        print(f"Initializing builder with {len(residue_selections)} custom selections.")
    else:
        print("Initializing builder to analyze all protein residues.")

    reader = MDAnalysisReader()
    extractor = FeatureExtractor(residue_selections)
    builder = DatasetBuilder(reader, extractor)
    return builder


def run_analysis(
    analysis_type: str,
    file_paths: Dict[str, str],
    params: Dict[str, Any],
    residue_selections: Optional[Dict[str, str]] = None,
    progress_callback: Optional[callable] = None
):
    """
    Main analysis execution function.

    Args:
        analysis_type: The type of analysis to run ('static', 'qubo', 'dynamic').
        file_paths: Dictionary of required file paths (e.g., 'active_topo_file').
        params: Dictionary of parameters for the analysis.
        residue_selections: Optional dictionary of residue selections.
        progress_callback: Optional function to report progress (e.g., `lambda msg, pct: print(msg)`).
        static_method
    """
    def report_progress(message, percent):
        if progress_callback:
            progress_callback(message, percent)
        else:
            print(message) # Default to printing if no callback is provided

    # For QUBO analysis, we must ensure that the target selection is included
    # in the feature extraction process from the very beginning.
    if analysis_type == 'qubo' and 'target_selection_string' in params:
        if residue_selections is None:
            residue_selections = {}
        # Add the target selection to the dictionary. The key can be simple;
        # it will be resolved and used by the QUBO module later.
        target_key = 'qubo_target_selection'
        residue_selections[target_key] = params['target_selection_string']
        print(f"QUBO analysis: Added target '{params['target_selection_string']}' to residue selections for feature extraction.")

    report_progress("Initializing Analysis Pipeline", 10)
    builder = get_builder(residue_selections)

    # Prepare data based on analysis type
    report_progress(f"Preparing {analysis_type} dataset", 30)
    active_slice = params.get("active_slice")
    inactive_slice = params.get("inactive_slice")

    if analysis_type in ['static', 'qubo']:
        data, labels, mapping = builder.prepare_static_analysis_data(
            file_paths['active_traj'], file_paths['active_topo'],
            file_paths['inactive_traj'], file_paths['inactive_topo'],
            active_slice=active_slice, inactive_slice=inactive_slice
        )
        analysis_data = (data, labels) if analysis_type == 'static' else (data, labels, mapping)
    elif analysis_type == 'dynamic':
        features_active, features_inactive, mapping = builder.prepare_dynamic_analysis_data(
            file_paths['active_traj'], file_paths['active_topo'],
            file_paths['inactive_traj'], file_paths['inactive_topo'],
            active_slice=active_slice, inactive_slice=inactive_slice
        )
        analysis_data = (features_active, features_inactive)
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    # Run the correct analysis
    report_progress(f"Running {analysis_type} analysis", 60)
    
    if analysis_type == 'static':
        method = params.get('static_method', 'ii').lower()
        if method == 'rf':
            print("Selected Method: Random Forest Classifier (Backup)")
            analyzer = StaticReportersRF()
        else:
            print("Selected Method: Information Imbalance (Default)")
            analyzer = StaticReportersII()
    
    elif analysis_type == 'qubo':
        method = params.get('qubo_method', 'ii').lower()
        
        if method == 'rf':
            print("Selected Method: QUBO via Random Forest (Backup)")
            analyzer = QUBOSetRF()
        else:
            print("Selected Method: QUBO via Information Imbalance (Default)")
            analyzer = QUBOSetII()
            
        # QUBO requires the topology file path in its params
        params['active_topo_file'] = file_paths['active_topo']
        
    elif analysis_type == 'dynamic':
        analyzer = TransferEntropy()

    job_results = analyzer.run(analysis_data, **params)

    report_progress("Analysis complete", 90)
    return job_results, mapping