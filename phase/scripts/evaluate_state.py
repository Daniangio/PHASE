from __future__ import annotations

import argparse
from pathlib import Path

from phase.workflows.clustering import evaluate_state_with_models
from phase.services.project_store import ProjectStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Offline data root (contains projects/)")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)
    ap.add_argument("--state-id", required=True)
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve() / "projects"
    store = ProjectStore(base_dir=root)
    _ = store.get_system(args.project_id, args.system_id)
    sample_entry = evaluate_state_with_models(
        args.project_id,
        args.system_id,
        args.cluster_id,
        args.state_id,
        store=store,
    )
    system_meta = store.get_system(args.project_id, args.system_id)
    clusters = system_meta.metastable_clusters or []
    entry = next((c for c in clusters if c.get("cluster_id") == args.cluster_id), None)
    if not entry:
        raise SystemExit("Cluster not found.")
    samples = entry.get("samples")
    if not isinstance(samples, list):
        samples = []
    samples.append(sample_entry)
    entry["samples"] = samples
    store.save_system(system_meta)
    print(f"[evaluate] sample saved: {sample_entry.get('path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
