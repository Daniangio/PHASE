from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def _normalize_model_list(raw: object) -> List[str]:
    """Normalize a model list coming from CLI or run_metadata.json."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out: List[str] = []
        for item in raw:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(raw).strip()
    if not s:
        return []
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _load_run_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_index(out_dir: Path, items: List[tuple[str, str]]) -> Path:
    links = "\n".join([f'<li><a href="{fname}">{label}</a></li>' for label, fname in items if fname])
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>PHASE Potts plots</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 18px; }}
    ul {{ line-height: 1.6; }}
    code {{ background: #f3f3f3; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h2>PHASE Potts plots</h2>
  <p>Open any report below. All HTML files are written next to this index.</p>
  <ul>
    {links}
  </ul>
</body>
</html>
"""
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate offline Plotly HTML reports for a PHASE Potts sampling run.")
    ap.add_argument("--results-dir", required=True, help="Directory containing run_summary.npz and run_metadata.json")
    ap.add_argument("--summary-file", default="", help="Optional explicit path to run_summary.npz")
    ap.add_argument("--metadata-file", default="", help="Optional explicit path to run_metadata.json")
    ap.add_argument("--offline", action="store_true", help="Inline plotly.js into HTML so it works without internet.")
    ap.add_argument("--model-npz", default="", help="Optional comma-separated list: base,delta1,delta2,... (for Experiment C).")
    ap.add_argument("--no-index", action="store_true", help="Do not write index.html")
    ap.add_argument("--delta-top-res", type=int, default=40)
    ap.add_argument("--delta-top-edge", type=int, default=60)
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {results_dir}")

    summary_path = Path(args.summary_file).expanduser().resolve() if args.summary_file else results_dir / "run_summary.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    metadata_path = Path(args.metadata_file).expanduser().resolve() if args.metadata_file else results_dir / "run_metadata.json"
    meta = _load_run_metadata(metadata_path)

    from phase.potts.plotting import (
        plot_marginal_summary_from_npz,
        plot_sampling_report_from_npz,
        plot_cross_likelihood_report_from_npz,
        plot_beta_scan_curve,
        plot_delta_terms_report_from_models,
    )

    written: List[tuple[str, str]] = []

    marg_path = plot_marginal_summary_from_npz(
        summary_path=summary_path,
        out_path=results_dir / "marginals.html",
        annotate=False,
        offline=bool(args.offline),
    )
    written.append(("Marginals dashboard", marg_path.name))

    samp_path = plot_sampling_report_from_npz(
        summary_path=summary_path,
        out_path=results_dir / "sampling_report.html",
        offline=bool(args.offline),
    )
    written.append(("Sampling report", samp_path.name))

    xlik_path = plot_cross_likelihood_report_from_npz(
        summary_path=summary_path,
        out_path=results_dir / "cross_likelihood_report.html",
        offline=bool(args.offline),
    )
    written.append(("Cross-likelihood report", xlik_path.name))

    data = np.load(summary_path, allow_pickle=True)
    beta_eff_grid = data["beta_eff_grid"] if "beta_eff_grid" in data else np.array([], dtype=float)
    beta_eff_dist = data["beta_eff_distances_by_schedule"] if "beta_eff_distances_by_schedule" in data else np.array([], dtype=float)
    sa_schedule_labels = data["sa_schedule_labels"] if "sa_schedule_labels" in data else np.array([], dtype=str)

    if beta_eff_grid.size and beta_eff_dist.size:
        outp = plot_beta_scan_curve(
            betas=[float(x) for x in beta_eff_grid.tolist()],
            distances=beta_eff_dist.tolist(),
            labels=[str(x) for x in sa_schedule_labels.tolist()] if sa_schedule_labels.size else None,
            out_path=results_dir / "beta_scan.html",
            offline=bool(args.offline),
        )
        written.append(("Beta-eff scan", outp.name))

    model_paths = _normalize_model_list(args.model_npz)
    if not model_paths:
        run_args = meta.get("args") if isinstance(meta, dict) else None
        if isinstance(run_args, dict):
            model_paths = _normalize_model_list(run_args.get("model_npz") or run_args.get("source_model"))

    if len(model_paths) >= 2:
        outp = plot_delta_terms_report_from_models(
            summary_path=summary_path,
            model_paths=model_paths,
            out_path=results_dir / "delta_terms.html",
            top_res=int(args.delta_top_res),
            top_edges=int(args.delta_top_edge),
            offline=bool(args.offline),
        )
        written.append(("Experiment C: delta term magnitudes", outp.name))

    if not args.no_index:
        idx = _write_index(results_dir, written)
        print(f"[plots] wrote index: {idx}")

    for label, fname in written:
        print(f"[plots] {label}: {results_dir / fname}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
