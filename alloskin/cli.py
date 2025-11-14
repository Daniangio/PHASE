"""
AllosKin: Command-Line Interface (CLI)
"""

# --- 1. Set Environment Variables ---
# This MUST be done before any libraries (numpy, dadapy)
# are imported, to prevent multiprocessing deadlocks.
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import yaml
import sys
from typing import Dict, Any

from alloskin.io.readers import MDAnalysisReader
from alloskin.features.extraction import FeatureExtractor
from alloskin.pipeline.builder import DatasetBuilder
from alloskin.analysis.static import StaticReportersRF
from alloskin.analysis.dynamic import TransferEntropy


def load_config(config_file: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    print(f"Loading configuration from {config_file}...")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            print("Warning: Config file is empty.")
            return {}
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file}", file=sys.stderr)
        sys.exit(1) # Exit if config file is missing
    except Exception as e:
        print(f"Error loading YAML configuration: {e}", file=sys.stderr)
        sys.exit(1) # Exit on YAML parsing error

def main():
    """
    Main function to parse arguments and run the analysis pipeline.
    """
    parser = argparse.ArgumentParser(description="AllosKin GPCR Analysis Pipeline")
    parser.add_argument(
        "analysis", 
        choices=["static", "qubo", "dynamic"], 
        help="The analysis goal to run."
    )
    # File path arguments
    parser.add_argument("--active_traj", required=True, help="Path to active state trajectory.")
    parser.add_argument("--active_topo", required=True, help="Path to active state topology.")
    parser.add_argument("--inactive_traj", required=True, help="Path to inactive state trajectory.")
    parser.add_argument("--inactive_topo", required=True, help="Path to inactive state topology.")
    
    # Config and parameter arguments
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to the residue_selections.yml config file."
    )
    parser.add_argument("--te_lag", type=int, default=10, help="Lag time for TE (in frames). Default: 10.")
    
    # --- New Arguments for Slicing and Multiprocessing ---
    parser.add_argument(
        "--active_slice", 
        type=str, 
        default=None, 
        help="Slice for active trajectory (e.g., '1000:5000:2')."
    )
    parser.add_argument(
        "--inactive_slice", 
        type=str, 
        default=None, 
        help="Slice for inactive trajectory (e.g., '1000:5000:2')."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=None, 
        help="Number of parallel workers for analysis. (Default: all available cores)"
    )
    # --- End New Arguments ---

    args = parser.parse_args()

    # --- 1. Initialization ---
    print("--- Initializing Pipeline ---")
    config = load_config(args.config)
    # Allow residue_selections to be None if not provided in the config.
    residue_selections = config.get('residue_selections', None)
    
    if residue_selections is None:
        print("NOTE: 'residue_selections' not found in config. Will analyze ALL protein residues.")
    elif not any(residue_selections.values()):
        print("Warning: 'residue_selections' is empty. Will analyze ALL protein residues.")

    # Note: extractor is now a 'prototype' holding the config
    reader = MDAnalysisReader()
    extractor = FeatureExtractor(residue_selections)
    builder = DatasetBuilder(reader, extractor)
    print("--- Initialization Complete ---")

    # --- 2. Analysis-Based Execution ---
    
    if args.analysis == "static":
        print("\n--- Preparing Data for Static Analysis ---")
        static_data = builder.prepare_static_analysis_data(
            args.active_traj, args.active_topo,
            args.inactive_traj, args.inactive_topo,
            active_slice=args.active_slice,
            inactive_slice=args.inactive_slice
        )
        print("\n--- Running Static Reporters (Random Forest) ---")
        analyzer = StaticReportersRF()
        results = analyzer.run(static_data, num_workers=args.num_workers)
        print("\n--- Final Static Results (Sorted by best reporter, highest accuracy) ---")
        print(results)

    elif args.analysis == "qubo":
        print("\n--- Preparing Data for QUBO Analysis ---")
        static_data = builder.prepare_static_analysis_data(
            args.active_traj, args.active_topo,
            args.inactive_traj, args.inactive_topo,
            active_slice=args.active_slice,
            inactive_slice=args.inactive_slice
        )
        # analyzer = QUBOSet()
        # results = analyzer.run(static_data, target_switch='res_131', num_workers=args.num_workers)
        print("--- Running Optimal Predictive Set (QUBO) ---")
        print("NOTE: QUBO analyzer is not implemented in this scaffold.")
        print("It would use the same `static_data` as the 'static' analysis.")
        results = "Not Implemented"

    elif args.analysis == "dynamic":
        print("\n--- Preparing Data for Dynamic Analysis ---")
        dynamic_data = builder.prepare_dynamic_analysis_data(
            args.active_traj, args.active_topo,
            args.inactive_traj, args.inactive_topo,
            active_slice=args.active_slice,
            inactive_slice=args.inactive_slice
        )
        print("--- Running Dynamic 'Orchestrated Action' (Transfer Entropy) ---")
        analyzer = TransferEntropy()
        results = analyzer.run(dynamic_data, lag=args.te_lag, num_workers=args.num_workers)
        print("\n--- Final Dynamic Results ---")
        print(results)

if __name__ == "__main__":
    main()