from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.spectral_analysis import upsert_hamiltonian_spectral_batch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute incremental single-state and pair Hamiltonian spectral analyses from state-associated Potts models."
    )
    ap.add_argument("--root", required=True, help="PHASE data root")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--state-ids", required=True, help="Comma-separated state ids")
    ap.add_argument("--top-k", type=int, default=20, help="Number of eigenvectors to store")
    ap.add_argument("--overwrite", action="store_true", help="Recompute existing single and pair analyses")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    os.environ["PHASE_DATA_ROOT"] = str(root)
    state_ids = [s.strip() for s in str(args.state_ids or "").split(",") if s.strip()]
    if not state_ids:
        raise SystemExit("No states selected.")

    def progress(message: str, current: int, total: int):
        if args.progress:
            print(f"[hamiltonian_spectral] {message}: {current}/{max(1, total)}", flush=True)

    out = upsert_hamiltonian_spectral_batch(
        project_id=args.project_id,
        system_id=args.system_id,
        cluster_id=args.cluster_id,
        state_ids=state_ids,
        top_k=int(args.top_k),
        overwrite=bool(args.overwrite),
        progress_callback=progress,
    )
    print(f"[hamiltonian_spectral] requested_states={','.join(out.get('requested_state_ids') or [])}")
    print(f"[hamiltonian_spectral] single_written_or_reused={out.get('single_count')}")
    print(f"[hamiltonian_spectral] pair_created={out.get('pair_count')}")
    skipped = out.get("skipped_states") or {}
    if skipped:
        print("[hamiltonian_spectral] skipped states:")
        for sid, reason in skipped.items():
            print(f"  - {sid}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
