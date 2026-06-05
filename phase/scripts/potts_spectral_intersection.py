from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.spectral_analysis import upsert_spectral_intersection_analysis


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute allosteric piston set-intersections between single-state and pair spectral communities."
    )
    ap.add_argument("--root", required=True, help="PHASE data root")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--single-analysis-id", required=True, help="hamiltonian_spectral_single analysis id")
    ap.add_argument("--pair-analysis-id", required=True, help="hamiltonian_spectral_pair analysis id")
    ap.add_argument("--min-group-size", type=int, default=3, help="Minimum residue count for an allosteric piston")
    ap.add_argument("--overwrite", action="store_true", help="Recompute existing intersection analysis")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    os.environ["PHASE_DATA_ROOT"] = str(root)
    out = upsert_spectral_intersection_analysis(
        project_id=args.project_id,
        system_id=args.system_id,
        cluster_id=args.cluster_id,
        single_analysis_id=args.single_analysis_id,
        pair_analysis_id=args.pair_analysis_id,
        min_group_size=int(args.min_group_size),
        overwrite=bool(args.overwrite),
    )
    meta = out.get("analysis") or {}
    summary = meta.get("summary") or {}
    print(f"[spectral_intersection] analysis_id={meta.get('analysis_id')}")
    print(f"[spectral_intersection] created={out.get('created')}")
    print(f"[spectral_intersection] pistons={summary.get('n_pistons')} residues={summary.get('piston_residues')}")
    print(f"[spectral_intersection] scaffolds={summary.get('structural_scaffold_residues')} transient_switches={summary.get('transient_switch_residues')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
