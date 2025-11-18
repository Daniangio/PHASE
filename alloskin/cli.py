"""
AllosKin: Command-Line Interface (CLI)

This script has been refactored to align with the new
analysis logic (e.g., passing a selection string and
topology path for QUBO).
"""
import argparse, yaml, sys, json
from typing import Dict, Any
from alloskin.pipeline.runner import run_analysis


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
        help="Path to the residue_selections.yml config file. If not provided, will analyze all protein residues."
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

    # --- Static Analysis Arguments ---
    parser.add_argument(
        "--static_method",
        choices=["ii", "rf"],
        default="rf",
        help="Method for static reporter detection: 'ii' (Information Imbalance, Default) or 'rf' (Random Forest)."
    )

    # --- QUBO Analysis Arguments ---
    parser.add_argument(
        "--qubo_method",
        choices=["ii", "rf"],
        default="ii",
        help="Method for QUBO feature selection: 'ii' (Information Imbalance, Default) or 'rf' (Random Forest)."
    )
    
    parser.add_argument(
        '--target_selection', 
        type=str,
        help='REQUIRED for QUBO. MDAnalysis selection string (e.g., "resid 131 140")'
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

    # --- Dynamic Analysis Arguments ---
    parser.add_argument("--te_lag", type=int, default=10, help="Lag time for TE (in frames). Default: 10.")
    

    args = parser.parse_args()

    # --- 1. Initialization ---
    print("--- Initializing Pipeline ---")
    
    residue_selections = None
    if args.config:
        config = load_config(args.config)
        residue_selections = config.get('residue_selections', None)
    
    if residue_selections is None:
        print("NOTE: No config or 'residue_selections' not found. Will analyze ALL protein residues.")
    elif not any(residue_selections.values()):
        print("Warning: 'residue_selections' is empty. Will analyze ALL protein residues.")

    # --- 2. Prepare arguments for the runner ---
    if args.analysis == "qubo":
        if not args.target_selection:
            print("Error: --target_selection argument is required for QUBO analysis.", file=sys.stderr)
            print("Example: --target_selection \"resid 131 140\"", file=sys.stderr)
            sys.exit(1)

    file_paths = {
        'active_traj': args.active_traj,
        'active_topo': args.active_topo,
        'inactive_traj': args.inactive_traj,
        'inactive_topo': args.inactive_topo,
    }

    # Collect all other parameters into a single dict
    params = {
        "active_slice": args.active_slice,
        "inactive_slice": args.inactive_slice,
        "num_workers": args.num_workers,
        # Static
        "static_method": args.static_method,
        # QUBO
        "qubo_method": args.qubo_method,
        "target_selection_string": args.target_selection,
        "lambda_redundancy": args.qubo_lambda,
        "num_solutions": args.qubo_solutions,
        "qubo_cv_folds": args.qubo_cv_folds,
        "qubo_n_estimators": args.qubo_n_estimators,
        # Dynamic
        "lag": args.te_lag, # Renamed from te_lag for consistency
    }

    # --- 3. Delegate to the core runner ---
    try:
        results, mapping = run_analysis(
            analysis_type=args.analysis,
            file_paths=file_paths,
            params=params,
            residue_selections=residue_selections
        )
        print(f"\n--- Final {args.analysis.upper()} Results ---")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"\nFATAL ERROR during analysis: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()