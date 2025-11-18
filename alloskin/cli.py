"""
CLI for AllosKin.
"""
import argparse, sys, json
from alloskin.pipeline.runner import run_analysis

def main():
    parser = argparse.ArgumentParser(description="AllosKin: Hierarchical Information Atlas")
    parser.add_argument("analysis", choices=["static", "qubo", "dynamic"])
    
    # Paths
    parser.add_argument("--active_traj", required=True)
    parser.add_argument("--active_topo", required=True)
    parser.add_argument("--inactive_traj", required=True)
    parser.add_argument("--inactive_topo", required=True)
    
    # Common
    parser.add_argument("--config", help="YAML config for residues")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--active_slice", default=None)
    parser.add_argument("--inactive_slice", default=None)
    
    # Goal 1 Params
    parser.add_argument("--maxk", type=int, default=100)
    
    # Goal 2 Params (Hierarchical QUBO)
    parser.add_argument("--static_results_path", help="Path to a JSON from a previous static analysis to use as input for QUBO")
    parser.add_argument("--alpha_size", type=float, default=1.0, help="Cost of adding a residue")
    parser.add_argument("--beta_hub", type=float, default=2.0, help="Reward per covered downstream residue")
    parser.add_argument("--beta_switch", type=float, default=5.0, help="Reward for predicting global state")
    parser.add_argument("--gamma_redundancy", type=float, default=3.0, help="Penalty for overlapping coverage")
    parser.add_argument("--ii_threshold", type=float, default=0.4, help="Threshold for 'Prediction' (Delta < T)")
    parser.add_argument("--filter_top_n", type=int, default=80)
    
    args = parser.parse_args()

    params = vars(args)
    
    # Map CLI args to internal names where they differ
    params['beta_coverage'] = args.beta_hub # Map CLI beta_hub to internal beta_coverage

    try:
        results, mapping = run_analysis(
            analysis_type=args.analysis,
            file_paths={
                'active_traj': args.active_traj, 'active_topo': args.active_topo,
                'inactive_traj': args.inactive_traj, 'inactive_topo': args.inactive_topo
            },
            params=params,
            residue_selections=None # Loader handles config parsing inside runner/builder
        )
        print(json.dumps(results, indent=2, default=str))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()