from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.analysis_run import run_gibbs_relaxation_analysis


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run Gibbs relaxation analysis from random starting frames of one sample "
            "under a selected Potts Hamiltonian."
        )
    )
    ap.add_argument("--root", default="", help="PHASE data root (defaults to $PHASE_DATA_ROOT or ./data).")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)

    ap.add_argument("--start-sample-id", required=True, help="Sample id used as starting ensemble (typically md_eval).")
    ap.add_argument("--model-id", default="", help="Target Potts model id on this cluster.")
    ap.add_argument("--model-path", default="", help="Target Potts model NPZ path (alternative to --model-id).")

    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n-start-frames", type=int, default=100)
    ap.add_argument("--gibbs-sweeps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="Worker processes (0 = all cpus).")
    ap.add_argument("--start-label-mode", default="assigned", choices=["assigned", "halo"])
    ap.add_argument("--keep-invalid", action="store_true")
    ap.add_argument("--progress", action="store_true")

    args = ap.parse_args(argv)

    model_ref = str(args.model_id or "").strip() or str(args.model_path or "").strip()
    if not model_ref:
        raise SystemExit("Provide --model-id or --model-path.")

    root = (args.root or os.getenv("PHASE_DATA_ROOT") or "").strip()
    if not root:
        root = str((Path(__file__).resolve().parents[2] / "data").resolve())
    os.environ["PHASE_DATA_ROOT"] = root

    last_pct = {"value": -1}

    def progress_cb(message: str, current: int, total: int):
        if not args.progress or total <= 0:
            return
        pct = int(100.0 * float(current) / float(total))
        if pct == last_pct["value"]:
            return
        last_pct["value"] = pct
        print(f"[gibbs_relax] {message} {current}/{total} ({pct}%)")

    out = run_gibbs_relaxation_analysis(
        project_id=str(args.project_id),
        system_id=str(args.system_id),
        cluster_id=str(args.cluster_id),
        start_sample_id=str(args.start_sample_id),
        model_ref=model_ref,
        beta=float(args.beta),
        n_start_frames=int(args.n_start_frames),
        gibbs_sweeps=int(args.gibbs_sweeps),
        seed=int(args.seed),
        start_label_mode=str(args.start_label_mode),
        drop_invalid=not bool(args.keep_invalid),
        n_workers=int(args.workers),
        progress_callback=progress_cb if bool(args.progress) else None,
    )

    meta = out.get("metadata") or {}
    print(f"[gibbs_relax] analysis_id: {meta.get('analysis_id')}")
    print(f"[gibbs_relax] analysis_npz: {out.get('analysis_npz')}")
    print(f"[gibbs_relax] analysis_dir: {out.get('analysis_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

