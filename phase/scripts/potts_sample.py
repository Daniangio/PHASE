from __future__ import annotations

from pathlib import Path

from phase.potts import pipeline as sim_main
from phase.scripts.potts_utils import persist_sample


def main(argv: list[str] | None = None) -> int:
    parser = sim_main._build_arg_parser()
    parser.add_argument("--project-id", default="")
    parser.add_argument("--system-id", default="")
    parser.add_argument("--cluster-id", default="")
    parser.add_argument("--sample-id", default="")
    parser.add_argument("--sample-name", default="")
    args = parser.parse_args(argv)
    args.fit_only = False
    if not getattr(args, "plot_only", False):
        args.no_plots = True
    args.no_save_model = True
    try:
        results = sim_main.run_pipeline(args, parser=parser)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 1
    project_id = (args.project_id or "").strip()
    system_id = (args.system_id or "").strip()
    cluster_id = (args.cluster_id or "").strip()
    if project_id and system_id and cluster_id and results and not getattr(args, "plot_only", False):
        summary_path = results.get("summary_path")
        meta_path = results.get("metadata_path")
        summary = Path(summary_path) if summary_path is not None else None
        meta = Path(meta_path) if meta_path is not None else None
        if summary is not None:
            model_paths = []
            for raw in getattr(args, "model_npz", []) or []:
                model_paths.append(Path(str(raw)))
            persist_sample(
                project_id=project_id,
                system_id=system_id,
                cluster_id=cluster_id,
                summary_path=summary,
                metadata_path=meta,
                sample_name=args.sample_name or None,
                sample_type="potts_sampling",
                method=str(args.sampling_method) if getattr(args, "sampling_method", None) else None,
                params=vars(args) if hasattr(args, "__dict__") else {},
                model_paths=model_paths,
                sample_id=args.sample_id or None,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
