from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import uuid

from phase.services.project_store import ProjectStore
from phase.potts.delta_fit import _device_arg, _load_labels_from_npz  # keep CLI behavior consistent
from phase.potts.potts_model import (
    add_potts_models,
    fit_potts_delta_pseudolikelihood_torch,
    load_potts_model,
    load_potts_model_metadata,
    save_potts_model,
)
from phase.scripts.potts_utils import sanitize_model_filename


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fit delta Potts models (offline runner that persists model_metadata.json after each new best).",
    )
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)

    ap.add_argument("--base-model", required=True, help="Base Potts model NPZ (frozen core).")
    ap.add_argument("--npz", required=True, help="Cluster NPZ containing frame_state_ids.")
    ap.add_argument("--state-ids", required=True, help="Comma-separated state IDs to include in the delta fit.")
    ap.add_argument("--state-label", default="", help="Optional label for the selected state set.")

    ap.add_argument("--model-name", default="", help="Base display name used for saved delta models.")
    ap.add_argument("--model-source", default="offline_delta_fit")
    ap.add_argument("--unassigned-policy", default="drop_frames", choices=["drop_frames", "treat_as_state", "error"])

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--lr-schedule", type=str, default="cosine", choices=["cosine", "none"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")

    ap.add_argument("--delta-l2", type=float, default=0.0)
    ap.add_argument("--delta-group-h", type=float, default=0.0)
    ap.add_argument("--delta-group-j", type=float, default=0.0)

    ap.add_argument("--resume-model", default="", help="Resume delta fit from an existing delta model NPZ (single-mode only).")
    ap.add_argument("--no-combined", action="store_true", help="Do not save base+delta combined models.")

    # Back-compat: older wrapper passed --results-dir; keep it accepted but ignored.
    ap.add_argument("--results-dir", default="", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _relativize_path(value: str, system_dir: Path) -> str:
    try:
        p = Path(value)
    except Exception:
        return value
    if not p.is_absolute():
        return value
    try:
        return str(p.relative_to(system_dir))
    except Exception:
        return value


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    project_id = str(args.project_id).strip()
    system_id = str(args.system_id).strip()
    cluster_id = str(args.cluster_id).strip()

    data_root = Path(os.getenv("PHASE_DATA_ROOT", "/app/data"))
    store = ProjectStore(base_dir=data_root / "projects")
    cluster_dirs = store.ensure_cluster_directories(project_id, system_id, cluster_id)
    system_dir = cluster_dirs["system_dir"]
    potts_models_dir = cluster_dirs["potts_models_dir"]

    system_meta = store.get_system(project_id, system_id)
    clusters = system_meta.metastable_clusters or []
    cluster_entry = next((c for c in clusters if c.get("cluster_id") == cluster_id), None)
    if not isinstance(cluster_entry, dict):
        raise SystemExit(f"Cluster '{cluster_id}' not found in system metadata.")

    base_model_path = Path(str(args.base_model))
    if not base_model_path.is_absolute():
        base_model_path = store.resolve_path(project_id, system_id, str(args.base_model))
    if not base_model_path.exists():
        raise SystemExit(f"Base model not found: {base_model_path}")
    base_model = load_potts_model(str(base_model_path))

    resume_model = None
    resume_path = None
    start_best_loss = None
    resume_raw = str(args.resume_model or "").strip()
    if resume_raw:
        resume_path = Path(resume_raw)
        if not resume_path.is_absolute():
            resume_path = store.resolve_path(project_id, system_id, resume_raw)
        if not resume_path.exists():
            raise SystemExit(f"Resume model not found: {resume_path}")
        resume_model = load_potts_model(str(resume_path))
        resume_meta = load_potts_model_metadata(str(resume_path))
        if resume_meta:
            start_best_loss = _coerce_float(resume_meta.get("delta_best_loss"))
            if start_best_loss is None:
                start_best_loss = _coerce_float(resume_meta.get("plm_best_loss"))

    state_ids = [s.strip() for s in str(args.state_ids or "").split(",") if s.strip()]
    if not state_ids:
        raise SystemExit("Select at least one state id (--state-ids).")

    # Load labels for the selected state set (single-mode only, by design)
    npz_path = Path(str(args.npz))
    if not npz_path.is_absolute():
        npz_path = store.resolve_path(project_id, system_id, str(args.npz))
    if not npz_path.exists():
        raise SystemExit(f"Cluster NPZ not found: {npz_path}")

    labels = _load_labels_from_npz(
        npz_path,
        state_ids=state_ids,
        unassigned_policy=str(args.unassigned_policy),
    )
    if labels.shape[1] != len(base_model.h):
        raise SystemExit("Labels do not match base model size.")

    device = _device_arg(str(args.device))

    # ------------------------------------------------------------------
    # Allocate model buckets up-front (so we always write into a stable folder).
    # ------------------------------------------------------------------
    base_label = (str(args.model_name or "").strip() or f"{base_model_path.stem} Delta")
    base_label = f"{base_label} ({','.join(state_ids)})"

    def _ensure_model_entry(
        *,
        model_id: str,
        display_name: str,
        kind: str,
        model_path_override: Path | None = None,
    ) -> tuple[Path, dict]:
        model_dir = potts_models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        if model_path_override is not None:
            model_path = Path(model_path_override)
        else:
            filename = f"{sanitize_model_filename(display_name)}.npz"
            model_path = model_dir / filename

        rel_model_path = _relativize_path(str(model_path), system_dir)

        params = {
            "fit_mode": "delta",
            "delta_kind": kind,
            "base_model": _relativize_path(str(base_model_path), system_dir),
            "resume_model": _relativize_path(str(resume_path), system_dir) if resume_path else None,
            "npz": _relativize_path(str(npz_path), system_dir),
            "state_ids": state_ids,
            "state_label": (str(args.state_label).strip() or None),
            "unassigned_policy": str(args.unassigned_policy),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "lr_min": float(args.lr_min),
            "lr_schedule": str(args.lr_schedule),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "device": str(args.device),
            "delta_l2": float(args.delta_l2),
            "delta_group_h": float(args.delta_group_h),
            "delta_group_j": float(args.delta_group_j),
            # Updated after each new best:
            "delta_best_loss": None,
            "delta_best_epoch": None,
            "delta_last_loss": None,
        }
        # Remove Nones for a cleaner metadata file.
        params = {k: v for k, v in params.items() if v is not None}

        model_entry = {
            "model_id": model_id,
            "name": display_name,
            "path": rel_model_path,
            "created_at": datetime.utcnow().isoformat(),
            "source": str(args.model_source or "offline_delta_fit"),
            "params": params,
        }

        models = cluster_entry.get("potts_models")
        if not isinstance(models, list):
            models = []
        existing = next((m for m in models if m.get("model_id") == model_id), None)
        if existing is not None:
            existing.update(model_entry)
        else:
            models.append(model_entry)
        cluster_entry["potts_models"] = models
        system_meta.metastable_clusters = clusters
        store.save_system(system_meta)
        return model_path, model_entry

    def _model_id_from_path(p: Path) -> str | None:
        try:
            if p.resolve().is_relative_to(potts_models_dir.resolve()):
                return p.parent.name
        except Exception:
            return None
        return None

    # Delta model bucket: reuse when resuming.
    delta_model_id = _model_id_from_path(resume_path) if resume_path else None
    if not delta_model_id:
        delta_model_id = str(uuid.uuid4())
    delta_path_override = resume_path if resume_path is not None else None
    delta_path, delta_entry = _ensure_model_entry(
        model_id=delta_model_id,
        display_name=f"{base_label} (delta)",
        kind="delta_patch",
        model_path_override=delta_path_override,
    )

    combined_path = None
    combined_entry = None
    if not bool(args.no_combined):
        combined_id = str(uuid.uuid4())
        combined_path, combined_entry = _ensure_model_entry(
            model_id=combined_id,
            display_name=f"{base_label} (combined)",
            kind="model_patch",
        )

    # ------------------------------------------------------------------
    # Best-save callback updates model_metadata.json (via system sync) every time loss improves.
    # ------------------------------------------------------------------
    def _update_best_metrics(*, best_loss: float, best_epoch: int, last_loss: float) -> None:
        system_meta_local = store.get_system(project_id, system_id)
        clusters_local = system_meta_local.metastable_clusters or []
        cluster_local = next((c for c in clusters_local if c.get("cluster_id") == cluster_id), None)
        if not isinstance(cluster_local, dict):
            return
        models_local = cluster_local.get("potts_models")
        if not isinstance(models_local, list):
            return

        def _apply(model_id: str) -> None:
            item = next((m for m in models_local if m.get("model_id") == model_id), None)
            if not isinstance(item, dict):
                return
            params = item.get("params")
            if not isinstance(params, dict):
                params = {}
            params["delta_best_loss"] = float(best_loss)
            params["delta_best_epoch"] = int(best_epoch)
            params["delta_last_loss"] = float(last_loss)
            item["params"] = params

        _apply(delta_model_id)
        if combined_entry is not None and combined_path is not None:
            _apply(str(combined_entry.get("model_id")))

        cluster_local["potts_models"] = models_local
        system_meta_local.metastable_clusters = clusters_local
        store.save_system(system_meta_local)

    # Fit
    delta_meta_for_npz = dict(delta_entry.get("params") or {})
    delta_meta_for_npz.update({"kind": "delta_patch", "label": str(args.state_label or "")})
    combined_meta_for_npz = None
    if combined_entry is not None:
        combined_meta_for_npz = dict(combined_entry.get("params") or {})
        combined_meta_for_npz.update({"kind": "model_patch", "label": str(args.state_label or "")})

    delta_model = fit_potts_delta_pseudolikelihood_torch(
        base_model,
        labels,
        l2=float(args.delta_l2),
        lambda_h=float(args.delta_group_h),
        lambda_J=float(args.delta_group_j),
        lr=float(args.lr),
        lr_min=float(args.lr_min),
        lr_schedule=str(args.lr_schedule),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=device,
        init_model=resume_model,
        start_best_loss=start_best_loss,
        best_model_path=str(delta_path),
        best_model_metadata=delta_meta_for_npz,
        best_combined_path=None if combined_path is None else str(combined_path),
        best_combined_metadata=combined_meta_for_npz,
        best_save_callback=lambda ep, best, last: _update_best_metrics(best_loss=best, best_epoch=ep, last_loss=last),
    )

    # If, for some reason, no best was ever saved (should be rare), at least write the final model once.
    if not delta_path.exists():
        save_potts_model(delta_model, str(delta_path), metadata=delta_meta_for_npz)
    if combined_path is not None and not combined_path.exists():
        save_potts_model(add_potts_models(base_model, delta_model), str(combined_path), metadata=combined_meta_for_npz)

    print(f"[potts_delta_fit] saved delta model: {str(delta_path.relative_to(system_dir))}")
    if combined_path is not None:
        print(f"[potts_delta_fit] saved combined model: {str(combined_path.relative_to(system_dir))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
