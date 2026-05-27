from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.orchestration import run_potts_nn_mapping_local


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run Potts-weighted nearest-neighbor mapping from one discrete sample ensemble to one MD ensemble "
            "in cluster-label space, using Potts field/coupling gaps as mismatch weights."
        )
    )
    ap.add_argument("--root", default="", help="PHASE data root (defaults to $PHASE_DATA_ROOT or ./data).")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--model-id", default="")
    ap.add_argument("--model-path", default="")
    ap.add_argument(
        "--sample-id",
        action="append",
        required=True,
        help=(
            "Sample ensemble to map onto MD nearest neighbors. "
            "Repeat this option or pass comma-separated ids to create one analysis per sample."
        ),
    )
    ap.add_argument("--md-sample-id", required=True, help="Reference MD sample (must be md_eval).")
    ap.add_argument("--md-label-mode", default="assigned", choices=["assigned", "halo"])
    ap.add_argument("--keep-invalid", action="store_true", help="Keep invalid rows instead of dropping them.")
    ap.add_argument("--no-unique", action="store_true", help="Disable unique-sequence compression.")
    ap.add_argument("--no-normalize", action="store_true", help="Disable normalization by max Potts gap weights.")
    ap.add_argument("--no-per-residue", action="store_true", help="Skip per-residue node/edge outputs.")
    ap.add_argument("--alpha", type=float, default=0.75, help="Default UI alpha for node/edge per-residue blending.")
    ap.add_argument("--beta-node", type=float, default=1.0)
    ap.add_argument("--beta-edge", type=float, default=1.0)
    ap.add_argument("--top-k-candidates", type=int, default=0, help="Optional node-only prefilter candidate count.")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument(
        "--distance-threshold",
        action="append",
        default=[],
        help="Distance threshold for summary coverage (repeatable). Defaults to 0.05,0.1,0.2.",
    )
    ap.add_argument("--workers", type=int, default=0, help="Optional cap for local worker fan-out (0 uses all unique rows).")
    ap.add_argument("--progress", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    root = (args.root or os.getenv("PHASE_DATA_ROOT") or "").strip()
    if not root:
        root = str((Path(__file__).resolve().parents[2] / "data").resolve())
    os.environ["PHASE_DATA_ROOT"] = root

    if not (str(args.model_id or "").strip() or str(args.model_path or "").strip()):
        raise SystemExit("Provide --model-id or --model-path.")

    thresholds = [float(v) for v in (args.distance_threshold or [])] or [0.05, 0.1, 0.2]
    sample_ids: list[str] = []
    for raw in args.sample_id or []:
        for sid in str(raw).split(","):
            sid = sid.strip()
            if sid and sid not in sample_ids:
                sample_ids.append(sid)
    if not sample_ids:
        raise SystemExit("Provide at least one --sample-id.")

    def progress_cb(message: str, current: int, total: int) -> None:
        if not args.progress:
            return
        print(f"[potts_nn_mapping] {message}: {current}/{total}")

    for idx, sample_id in enumerate(sample_ids, start=1):
        if len(sample_ids) > 1:
            print(f"[potts_nn_mapping] running {idx}/{len(sample_ids)} sample_id={sample_id}")
        out = run_potts_nn_mapping_local(
            project_id=str(args.project_id),
            system_id=str(args.system_id),
            cluster_id=str(args.cluster_id),
            model_id=(str(args.model_id).strip() or None),
            model_path=(str(args.model_path).strip() or None),
            sample_id=str(sample_id),
            md_sample_id=str(args.md_sample_id),
            md_label_mode=str(args.md_label_mode),
            keep_invalid=bool(args.keep_invalid),
            use_unique=not bool(args.no_unique),
            normalize=not bool(args.no_normalize),
            compute_per_residue=not bool(args.no_per_residue),
            alpha=float(args.alpha),
            beta_node=float(args.beta_node),
            beta_edge=float(args.beta_edge),
            top_k_candidates=(int(args.top_k_candidates) if int(args.top_k_candidates) > 0 else None),
            chunk_size=int(args.chunk_size),
            distance_thresholds=thresholds,
            n_workers=(int(args.workers) if int(args.workers) > 0 else None),
            progress_callback=progress_cb,
        )
        meta = out.get("metadata") or {}
        print(f"[potts_nn_mapping] analysis_id={out.get('analysis_id')}")
        print(f"[potts_nn_mapping] analysis_npz={out.get('analysis_npz')}")
        print(
            "[potts_nn_mapping] "
            f"sample={meta.get('sample_name') or meta.get('sample_id')} "
            f"-> md={meta.get('md_sample_name') or meta.get('md_sample_id')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
