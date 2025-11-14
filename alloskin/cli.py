"""
AllosKin: Command-Line Interface (CLI)
"""
import argparse
import yaml
import sys
from typing import Dict, Any

from alloskin.io.readers import MDAnalysisReader
from alloskin.features.extraction import FeatureExtractor
from alloskin.pipeline.builder import DatasetBuilder
from alloskin.analysis.static import StaticReportersRF
from alloskin.analysis.qubo import QUBOSet
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
    
    # --- Slicing and Multiprocessing ---
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

    # --- QUBO Arguments ---
    parser.add_argument(
        '--target_residues', 
        nargs='+', 
        help='REQUIRED for QUBO. List of target residue keys (e.g., res_131 res_140)'
    )
    parser.add_argument(
        '--qubo_lambda', 
        type=float, 
        default=1.0, 
        help='QUBO redundancy penalty (lambda). Default: 1.0'
    )
    parser.add_argument(
        '--qubo_solutions', 
        type=int, 
        default=5, 
        help='Number of optimal solutions to find. Default: 5'
    )
    parser.add_argument(
        '--qubo_cv_folds', 
        type=int, 
        default=3, 
        help='CV folds for QUBO RF regressors. Default: 3'
    )
    parser.add_argument(
        '--qubo_n_estimators', 
        type=int, 
        default=50, 
        help='Number of trees for QUBO RF regressors. Default: 50'
    )

    # --- Dynamic Arguments ---
    parser.add_argument("--te_lag", type=int, default=10, help="Lag time for TE (in frames). Default: 10.")
    

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
        # prepare_static_analysis_data returns (all_features_static, labels_Y, mapping)
        static_data, labels_Y, mapping = builder.prepare_static_analysis_data(
            args.active_traj, args.active_topo,
            args.inactive_traj, args.inactive_topo,
            active_slice=args.active_slice,
            inactive_slice=args.inactive_slice
        )
        print("\n--- Running Static Reporters (Random Forest) ---")
        analyzer = StaticReportersRF()
        # The analyzer only needs the first two items
        results = analyzer.run(
            (static_data, labels_Y), 
            num_workers=args.num_workers
        )
        print("\n--- Final Static Results (Sorted by best reporter, highest accuracy) ---")
        print(results)

    elif args.analysis == "qubo":
        print("\n--- Preparing Data for QUBO Analysis ---")
        
        if not args.target_residues:
            print("Error: --target_residues argument is required for QUBO analysis.", file=sys.stderr)
            print("Example: --target_residues res_131 res_140", file=sys.stderr)
            sys.exit(1)

        # QUBO also uses the static dataset
        static_data, labels_Y, mapping = builder.prepare_static_analysis_data(
            args.active_traj, args.active_topo,
            args.inactive_traj, args.inactive_topo,
            active_slice=args.active_slice,
            inactive_slice=args.inactive_slice
        )
        
        # Pass all data and kwargs to the analyzer
        qubo_data_tuple = (static_data, labels_Y, mapping)
        qubo_kwargs = {
            "target_residues": args.target_residues,
            "lambda_redundancy": args.qubo_lambda,
            "num_solutions": args.qubo_solutions,
            "qubo_cv_folds": args.qubo_cv_folds,
            "qubo_n_estimators": args.qubo_n_estimators
        }

        print("--- Running Optimal Predictive Set (QUBO) ---")
        analyzer = QUBOSet()
        results = analyzer.run(
            qubo_data_tuple, 
            num_workers=args.num_workers, 
            **qubo_kwargs
        )
        
        print("\n--- Final QUBO Results ---")
        # Pretty-print the dictionary results
        import json
        print(json.dumps(results, indent=2))


    elif args.analysis == "dynamic":
        print("\n--- Preparing Data for Dynamic Analysis ---")
        # prepare_dynamic_analysis_data returns (features_active, features_inactive, mapping)
        features_active, features_inactive, mapping = builder.prepare_dynamic_analysis_data(
            args.active_traj, args.active_topo,
            args.inactive_traj, args.inactive_topo,
            active_slice=args.active_slice,
            inactive_slice=args.inactive_slice
        )
        print("--- Running Dynamic 'Orchestrated Action' (Transfer Entropy) ---")
        analyzer = TransferEntropy()
        # The analyzer only needs the first two items
        results = analyzer.run(
            (features_active, features_inactive), 
            lag=args.te_lag, 
            num_workers=args.num_workers
        )
        print("\n--- Final Dynamic Results ---")
        print(results)

if __name__ == "__main":
    main()