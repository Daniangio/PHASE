from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.analysis_run import upsert_delta_energy_analysis


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compute delta-energy distributions, ΔE = E_model_A - E_model_B, for selected clustered trajectories."
    )
    ap.add_argument("--root", default="", help="PHASE data root (defaults to $PHASE_DATA_ROOT or ./data).")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--model-a-id", required=True)
    ap.add_argument("--model-b-id", required=True)
    ap.add_argument("--sample-ids", required=True, help="Comma-separated sample ids to analyze.")
    ap.add_argument("--md-label-mode", default="assigned", choices=["assigned", "halo"])
    ap.add_argument("--keep-invalid", action="store_true", help="Keep invalid frames instead of dropping them.")
    ap.add_argument("--frame-limit", action="append", default=[], help="Per-sample random limit as sample_id:n_frames. Omit or 0 = all frames.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--energy-bins", type=int, default=80)
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers for local analysis (0=auto).")
    ap.add_argument("--progress", action="store_true", help="Show a local progress bar while processing samples.")
    return ap


def _parse_frame_limits(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items or []:
        raw = str(item or "").strip()
        if not raw:
            continue
        if ":" not in raw:
            raise SystemExit(f"Invalid --frame-limit '{raw}'. Expected sample_id:n_frames.")
        sid, value = raw.split(":", 1)
        sid = sid.strip()
        if not sid:
            raise SystemExit(f"Invalid --frame-limit '{raw}'. Empty sample_id.")
        out[sid] = max(0, int(value.strip() or "0"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    root = (args.root or os.getenv("PHASE_DATA_ROOT") or "").strip()
    if not root:
        root = str((Path(__file__).resolve().parents[2] / "data").resolve())
    os.environ["PHASE_DATA_ROOT"] = root

    sample_ids = [s.strip() for s in str(args.sample_ids or "").split(",") if s.strip()]
    if not sample_ids:
        raise SystemExit("Provide at least one sample id via --sample-ids.")
    frame_limits = _parse_frame_limits(args.frame_limit)

    progress_bar = None

    def progress_cb(message: str, current: int, total: int) -> None:
        nonlocal progress_bar
        if not args.progress:
            return
        total_i = max(0, int(total))
        current_i = max(0, int(current))
        if progress_bar is None:
            try:
                from tqdm import tqdm  # type: ignore

                progress_bar = tqdm(total=total_i, desc=message, unit="sample")
            except Exception:
                progress_bar = False
        if progress_bar is False:
            print(f"[delta_energy] {message}: {current_i}/{max(1, total_i)}")
            return
        progress_bar.set_description_str(message)
        progress_bar.total = total_i
        progress_bar.n = min(current_i, total_i)
        progress_bar.refresh()

    try:
        out = upsert_delta_energy_analysis(
            project_id=str(args.project_id),
            system_id=str(args.system_id),
            cluster_id=str(args.cluster_id),
            model_a_ref=str(args.model_a_id),
            model_b_ref=str(args.model_b_id),
            sample_ids=sample_ids,
            md_label_mode=str(args.md_label_mode),
            drop_invalid=not bool(args.keep_invalid),
            frame_limits=frame_limits,
            seed=int(args.seed),
            energy_bins=int(args.energy_bins),
            n_workers=(int(args.workers) if int(args.workers) > 0 else None),
            progress_callback=progress_cb if args.progress else None,
        )
    finally:
        if progress_bar is not None and progress_bar is not False:
            progress_bar.close()
    meta = out.get("metadata") or {}
    print(f"[delta_energy] analysis_id={meta.get('analysis_id')}")
    print(f"[delta_energy] analysis_npz={out.get('analysis_npz')}")
    print(f"[delta_energy] samples={len(sample_ids)} workers={meta.get('summary', {}).get('workers_used')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
