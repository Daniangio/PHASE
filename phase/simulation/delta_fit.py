from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from phase.io.data import load_npz
from phase.simulation.potts_model import (
    add_potts_models,
    fit_potts_delta_pseudolikelihood_torch,
    load_potts_model,
    save_potts_model,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fit delta Potts models on top of a frozen base model.",
    )
    ap.add_argument("--base-model", required=True, help="Base Potts model NPZ (shared core).")
    ap.add_argument("--npz", help="Cluster NPZ containing frame_state_ids.")
    ap.add_argument("--active-state-id", help="State ID for active frames (used with --npz).")
    ap.add_argument("--inactive-state-id", help="State ID for inactive frames (used with --npz).")
    ap.add_argument("--active-npz", help="Optional NPZ with active frames (overrides --npz).")
    ap.add_argument("--inactive-npz", help="Optional NPZ with inactive frames (overrides --npz).")
    ap.add_argument("--results-dir", required=True, help="Output directory for delta models.")

    ap.add_argument("--unassigned-policy", default="drop_frames", choices=["drop_frames", "treat_as_state", "error"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "none"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument("--delta-l2", type=float, default=0.0, help="L2 weight on delta parameters.")
    ap.add_argument("--delta-group-h", type=float, default=0.0, help="Group sparsity on delta fields.")
    ap.add_argument("--delta-group-j", type=float, default=0.0, help="Group sparsity on delta couplings.")

    ap.add_argument("--no-combined", action="store_true", help="Do not save base+delta combined models.")
    return ap.parse_args(argv)


def _load_labels_from_npz(path: Path, *, state_id: Optional[str], unassigned_policy: str) -> np.ndarray:
    ds = load_npz(str(path), unassigned_policy=unassigned_policy, allow_missing_edges=True)
    labels = ds.labels
    if state_id is None:
        return labels
    if ds.frame_state_ids is None:
        raise ValueError("frame_state_ids missing in NPZ; cannot filter by state.")
    frame_ids = np.asarray(ds.frame_state_ids).astype(str)
    mask = frame_ids == str(state_id)
    if not np.any(mask):
        raise ValueError(f"No frames matched state_id='{state_id}'.")
    return labels[mask]


def _device_arg(raw: str) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value or value == "auto":
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    base_model = load_potts_model(args.base_model)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.active_npz and args.inactive_npz:
        active_labels = _load_labels_from_npz(Path(args.active_npz), state_id=None, unassigned_policy=args.unassigned_policy)
        inactive_labels = _load_labels_from_npz(Path(args.inactive_npz), state_id=None, unassigned_policy=args.unassigned_policy)
    else:
        if not args.npz:
            raise ValueError("Provide --npz or both --active-npz/--inactive-npz.")
        if not args.active_state_id or not args.inactive_state_id:
            raise ValueError("Provide --active-state-id and --inactive-state-id when using --npz.")
        npz_path = Path(args.npz)
        active_labels = _load_labels_from_npz(npz_path, state_id=args.active_state_id, unassigned_policy=args.unassigned_policy)
        inactive_labels = _load_labels_from_npz(npz_path, state_id=args.inactive_state_id, unassigned_policy=args.unassigned_policy)

    if active_labels.shape[1] != len(base_model.h):
        raise ValueError("Active labels do not match base model size.")
    if inactive_labels.shape[1] != len(base_model.h):
        raise ValueError("Inactive labels do not match base model size.")

    device = _device_arg(args.device)

    print("[delta] fitting active delta model...")
    delta_active = fit_potts_delta_pseudolikelihood_torch(
        base_model,
        active_labels,
        l2=args.delta_l2,
        lambda_h=args.delta_group_h,
        lambda_J=args.delta_group_j,
        lr=args.lr,
        lr_min=args.lr_min,
        lr_schedule=args.lr_schedule,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    print("[delta] fitting inactive delta model...")
    delta_inactive = fit_potts_delta_pseudolikelihood_torch(
        base_model,
        inactive_labels,
        l2=args.delta_l2,
        lambda_h=args.delta_group_h,
        lambda_J=args.delta_group_j,
        lr=args.lr,
        lr_min=args.lr_min,
        lr_schedule=args.lr_schedule,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    meta = {
        "base_model": str(Path(args.base_model)),
        "active_state_id": args.active_state_id,
        "inactive_state_id": args.inactive_state_id,
        "active_npz": args.active_npz,
        "inactive_npz": args.inactive_npz,
        "npz": args.npz,
        "fit_params": {
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_min": args.lr_min,
            "lr_schedule": args.lr_schedule,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "delta_l2": args.delta_l2,
            "delta_group_h": args.delta_group_h,
            "delta_group_j": args.delta_group_j,
        },
    }

    active_delta_path = results_dir / "delta_active.npz"
    inactive_delta_path = results_dir / "delta_inactive.npz"
    save_potts_model(delta_active, active_delta_path, metadata={**meta, "kind": "delta_active"})
    save_potts_model(delta_inactive, inactive_delta_path, metadata={**meta, "kind": "delta_inactive"})
    print(f"[delta] saved {active_delta_path}")
    print(f"[delta] saved {inactive_delta_path}")

    if not args.no_combined:
        active_model = add_potts_models(base_model, delta_active)
        inactive_model = add_potts_models(base_model, delta_inactive)
        active_model_path = results_dir / "model_active.npz"
        inactive_model_path = results_dir / "model_inactive.npz"
        save_potts_model(active_model, active_model_path, metadata={**meta, "kind": "model_active"})
        save_potts_model(inactive_model, inactive_model_path, metadata={**meta, "kind": "model_inactive"})
        print(f"[delta] saved {active_model_path}")
        print(f"[delta] saved {inactive_model_path}")

    meta_path = results_dir / "delta_fit_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[delta] metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
