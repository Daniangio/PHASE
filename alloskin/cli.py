"""
CLI for AllosKin.
"""
import argparse, sys, json
from alloskin.pipeline.runner import run_analysis

def main():
    parser = argparse.ArgumentParser(description="AllosKin: Hierarchical Information Atlas (local debug CLI)")
    parser.add_argument("analysis", choices=["static"])

    # Descriptor inputs (preferred for current pipeline)
    parser.add_argument("--active_descriptors", help="Path to active descriptors NPZ")
    parser.add_argument("--inactive_descriptors", help="Path to inactive descriptors NPZ")
    parser.add_argument("--descriptor_keys", help="Comma-separated keys to load from NPZ; defaults to all")
    parser.add_argument("--residue_mapping", help="Path to JSON with residue mapping (optional)")
    parser.add_argument("--n_frames_active", type=int, help="Override frame count for active descriptors")
    parser.add_argument("--n_frames_inactive", type=int, help="Override frame count for inactive descriptors")

    # Raw trajectory inputs (fallback)
    parser.add_argument("--active_traj")
    parser.add_argument("--active_topo")
    parser.add_argument("--inactive_traj")
    parser.add_argument("--inactive_topo")

    # Common
    parser.add_argument("--config", help="YAML/JSON config for residues")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--active_slice", default=None)
    parser.add_argument("--inactive_slice", default=None)

    # Clustering Params
    parser.add_argument("--maxk", type=int, default=100)

    
    args = parser.parse_args()

    params = vars(args)
    # Build file_paths
    file_paths = {}
    if args.active_descriptors and args.inactive_descriptors:
        desc_keys = []
        if args.descriptor_keys:
            desc_keys = [k.strip() for k in args.descriptor_keys.split(",") if k.strip()]
        else:
            import numpy as np
            data = np.load(args.active_descriptors, allow_pickle=True)
            desc_keys = list(data.keys())

        file_paths.update({
            "active_descriptors": args.active_descriptors,
            "inactive_descriptors": args.inactive_descriptors,
            "descriptor_keys": desc_keys,
        })
        if args.n_frames_active:
            file_paths["n_frames_active"] = args.n_frames_active
        if args.n_frames_inactive:
            file_paths["n_frames_inactive"] = args.n_frames_inactive
        if args.residue_mapping:
            import json
            with open(args.residue_mapping, "r") as fh:
                file_paths["residue_mapping"] = json.load(fh)
    else:
        # Require trajectories
        required = ["active_traj", "active_topo", "inactive_traj", "inactive_topo"]
        missing = [r for r in required if not getattr(args, r)]
        if missing:
            parser.error(f"Missing required trajectory args: {', '.join(missing)}")
        file_paths.update({
            "active_traj": args.active_traj,
            "active_topo": args.active_topo,
            "inactive_traj": args.inactive_traj,
            "inactive_topo": args.inactive_topo,
        })

    try:
        results, mapping = run_analysis(
            analysis_type=args.analysis,
            file_paths=file_paths,
            params=params,
            residue_selections=None # Loader handles config parsing inside runner/builder
        )
        print(json.dumps(results, indent=2, default=str))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
