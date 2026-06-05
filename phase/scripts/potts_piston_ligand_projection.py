from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.spectral_analysis import upsert_piston_ligand_projection_analysis


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compute masked ligand/short-MD projection scores for allosteric pistons.")
    ap.add_argument("--root", required=True, help="PHASE data root")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--intersection-analysis-id", required=True)
    ap.add_argument("--sample-ids", required=True, help="Comma-separated ligand/MD sample ids")
    ap.add_argument("--label-mode", default="assigned", choices=["assigned", "halo"])
    ap.add_argument("--keep-invalid", action="store_true", help="Keep invalid sampled rows when present")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    os.environ["PHASE_DATA_ROOT"] = str(root)
    sample_ids = [s.strip() for s in str(args.sample_ids or "").split(",") if s.strip()]
    if not sample_ids:
        raise SystemExit("No samples selected.")
    out = upsert_piston_ligand_projection_analysis(
        project_id=args.project_id,
        system_id=args.system_id,
        cluster_id=args.cluster_id,
        intersection_analysis_id=args.intersection_analysis_id,
        sample_ids=sample_ids,
        label_mode=args.label_mode,
        drop_invalid=not bool(args.keep_invalid),
        overwrite=bool(args.overwrite),
    )
    meta = out.get("analysis") or {}
    summary = meta.get("summary") or {}
    print(f"[piston_ligand_projection] analysis_id={meta.get('analysis_id')}")
    print(f"[piston_ligand_projection] created={out.get('created')}")
    print(f"[piston_ligand_projection] samples={summary.get('n_samples')} pistons={summary.get('n_pistons')}")
    print(f"[piston_ligand_projection] score_range={summary.get('min_score')}..{summary.get('max_score')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
