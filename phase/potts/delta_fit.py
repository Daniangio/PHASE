from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from phase.io.data import load_npz
from phase.potts.potts_model import (
    add_potts_models,
    fit_potts_delta_pseudolikelihood_torch,
    load_potts_model,
    load_potts_model_metadata,
    save_potts_model,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fit delta Potts models on top of a frozen base model.",
    )
    ap.add_argument("--base-model", required=True, help="Base Potts model NPZ (shared core).")
    ap.add_argument("--npz", help="Cluster NPZ containing frame_state_ids.")
    ap.add_argument(
        "--state-ids",
        help="Comma-separated state IDs to include in the delta fit (used with --npz).",
    )
    ap.add_argument("--state-label", help="Optional label for the selected state set.")
    ap.add_argument("--active-state-id", help="Legacy: state ID for active frames (used with --npz).")
    ap.add_argument("--inactive-state-id", help="Legacy: state ID for inactive frames (used with --npz).")
    ap.add_argument("--active-npz", help="Legacy: NPZ with active frames (overrides --npz).")
    ap.add_argument("--inactive-npz", help="Legacy: NPZ with inactive frames (overrides --npz).")
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
    ap.add_argument("--resume-model", help="Resume delta fit from an existing delta model NPZ.")

    ap.add_argument("--no-combined", action="store_true", help="Do not save base+delta combined models.")
    return ap.parse_args(argv)


def _load_labels_from_npz(
    path: Path,
    *,
    state_ids: Optional[list[str]] = None,
    unassigned_policy: str,
) -> np.ndarray:
    ds = load_npz(str(path), unassigned_policy=unassigned_policy, allow_missing_edges=True)
    labels = ds.labels
    if not state_ids:
        return labels
    if ds.frame_state_ids is None:
        raise ValueError("frame_state_ids missing in NPZ; cannot filter by state.")
    frame_ids = np.asarray(ds.frame_state_ids).astype(str)
    state_set = {str(state_id) for state_id in state_ids if state_id is not None}
    mask = np.isin(frame_ids, list(state_set))
    if not np.any(mask):
        raise ValueError(f"No frames matched state_ids={sorted(state_set)}.")
    return labels[mask]


def _device_arg(raw: str) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value or value == "auto":
        return None
    return value


def _coerce_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta_metrics(model: object) -> dict:
    metrics = {}
    best_loss = getattr(model, "best_delta_loss", None)
    last_loss = getattr(model, "last_delta_loss", None)
    best_epoch = getattr(model, "best_delta_epoch", None)
    if best_loss is not None:
        metrics["delta_best_loss"] = float(best_loss)
    if last_loss is not None:
        metrics["delta_last_loss"] = float(last_loss)
    if best_epoch is not None:
        metrics["delta_best_epoch"] = int(best_epoch)
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    base_model = load_potts_model(args.base_model)
    resume_model = None
    resume_meta = None
    start_best_loss = None
    if args.resume_model:
        resume_model = load_potts_model(args.resume_model)
        resume_meta = load_potts_model_metadata(args.resume_model)
        if resume_meta:
            start_best_loss = _coerce_float(resume_meta.get("delta_best_loss"))
            if start_best_loss is None:
                start_best_loss = _coerce_float(resume_meta.get("plm_best_loss"))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    state_ids = []
    if args.state_ids:
        state_ids = [s.strip() for s in str(args.state_ids).split(",") if s.strip()]

    single_mode = bool(state_ids)

    if single_mode:
        if not args.npz:
            raise ValueError("Provide --npz when using --state-ids.")
        npz_path = Path(args.npz)
        labels = _load_labels_from_npz(npz_path, state_ids=state_ids, unassigned_policy=args.unassigned_policy)
    elif args.active_npz and args.inactive_npz:
        active_labels = _load_labels_from_npz(
            Path(args.active_npz),
            state_ids=None,
            unassigned_policy=args.unassigned_policy,
        )
        inactive_labels = _load_labels_from_npz(
            Path(args.inactive_npz),
            state_ids=None,
            unassigned_policy=args.unassigned_policy,
        )
    else:
        if not args.npz:
            raise ValueError("Provide --npz or both --active-npz/--inactive-npz.")
        if not args.active_state_id or not args.inactive_state_id:
            raise ValueError("Provide --state-ids or both --active-state-id/--inactive-state-id when using --npz.")
        npz_path = Path(args.npz)
        active_labels = _load_labels_from_npz(
            npz_path,
            state_ids=[args.active_state_id],
            unassigned_policy=args.unassigned_policy,
        )
        inactive_labels = _load_labels_from_npz(
            npz_path,
            state_ids=[args.inactive_state_id],
            unassigned_policy=args.unassigned_policy,
        )

    if single_mode:
        if labels.shape[1] != len(base_model.h):
            raise ValueError("Labels do not match base model size.")
    else:
        if active_labels.shape[1] != len(base_model.h):
            raise ValueError("Active labels do not match base model size.")
        if inactive_labels.shape[1] != len(base_model.h):
            raise ValueError("Inactive labels do not match base model size.")

    device = _device_arg(args.device)

    meta = {
        "base_model": str(Path(args.base_model)),
        "resume_model": str(Path(args.resume_model)) if args.resume_model else None,
        "state_ids": state_ids or None,
        "state_label": args.state_label,
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

    if args.resume_model and not single_mode:
        print("[delta] warning: --resume-model is only supported with --state-ids; ignoring resume.")
        resume_model = None
        start_best_loss = None

    if single_mode:
        label = args.state_label or "delta_patch"
        print(f"[delta] fitting delta model for states={state_ids}...")
        delta_path = results_dir / "delta_model.npz"
        combined_path = results_dir / "model_combined.npz"
        delta_meta = {**meta, "kind": "delta_patch", "label": label}
        combined_meta = {**meta, "kind": "model_patch", "label": label}
        delta_model = fit_potts_delta_pseudolikelihood_torch(
            base_model,
            labels,
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
            init_model=resume_model,
            start_best_loss=start_best_loss,
            best_model_path=str(delta_path),
            best_model_metadata=delta_meta,
            best_combined_path=None if args.no_combined else str(combined_path),
            best_combined_metadata=None if args.no_combined else combined_meta,
        )
        metrics = _delta_metrics(delta_model)
        if metrics:
            meta.update(metrics)
        if not delta_path.exists():
            save_potts_model(delta_model, delta_path, metadata={**delta_meta, **metrics})
        print(f"[delta] saved {delta_path}")
        if not args.no_combined:
            if not combined_path.exists():
                combined_model = add_potts_models(base_model, delta_model)
                save_potts_model(combined_model, combined_path, metadata={**combined_meta, **metrics})
            print(f"[delta] saved {combined_path}")
    else:
        active_delta_path = results_dir / "delta_active.npz"
        inactive_delta_path = results_dir / "delta_inactive.npz"
        active_model_path = results_dir / "model_active.npz"
        inactive_model_path = results_dir / "model_inactive.npz"

        print("[delta] fitting active delta model...")
        active_meta = {**meta, "kind": "delta_active"}
        active_combined_meta = {**meta, "kind": "model_active"}
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
            best_model_path=str(active_delta_path),
            best_model_metadata=active_meta,
            best_combined_path=None if args.no_combined else str(active_model_path),
            best_combined_metadata=None if args.no_combined else active_combined_meta,
        )
        active_metrics = _delta_metrics(delta_active)
        if active_metrics:
            meta["delta_active_metrics"] = active_metrics
        if not active_delta_path.exists():
            save_potts_model(delta_active, active_delta_path, metadata={**active_meta, **active_metrics})
        print(f"[delta] saved {active_delta_path}")

        print("[delta] fitting inactive delta model...")
        inactive_meta = {**meta, "kind": "delta_inactive"}
        inactive_combined_meta = {**meta, "kind": "model_inactive"}
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
            best_model_path=str(inactive_delta_path),
            best_model_metadata=inactive_meta,
            best_combined_path=None if args.no_combined else str(inactive_model_path),
            best_combined_metadata=None if args.no_combined else inactive_combined_meta,
        )
        inactive_metrics = _delta_metrics(delta_inactive)
        if inactive_metrics:
            meta["delta_inactive_metrics"] = inactive_metrics
        if not inactive_delta_path.exists():
            save_potts_model(delta_inactive, inactive_delta_path, metadata={**inactive_meta, **inactive_metrics})
        print(f"[delta] saved {inactive_delta_path}")

        if not args.no_combined:
            if not active_model_path.exists():
                active_model = add_potts_models(base_model, delta_active)
                save_potts_model(active_model, active_model_path, metadata={**active_combined_meta, **active_metrics})
            if not inactive_model_path.exists():
                inactive_model = add_potts_models(base_model, delta_inactive)
                save_potts_model(inactive_model, inactive_model_path, metadata={**inactive_combined_meta, **inactive_metrics})
            print(f"[delta] saved {active_model_path}")
            print(f"[delta] saved {inactive_model_path}")

    meta_path = results_dir / "delta_fit_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[delta] metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
