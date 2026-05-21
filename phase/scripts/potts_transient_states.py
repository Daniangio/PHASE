from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.transient_analysis import run_transient_state_analysis


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value or '').split(',') if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Run transient cluster-state enrichment analysis.')
    ap.add_argument('--root', default='', help='PHASE data root (defaults to $PHASE_DATA_ROOT or ./data).')
    ap.add_argument('--project-id', required=True)
    ap.add_argument('--system-id', required=True)
    ap.add_argument('--cluster-id', required=True)
    ap.add_argument('--sample-ids', required=True, help='Comma-separated sample ids to compare.')
    ap.add_argument('--md-label-mode', default='assigned', choices=['assigned', 'halo'])
    ap.add_argument('--keep-invalid', action='store_true')
    ap.add_argument('--p-min', type=float, default=0.005)
    ap.add_argument('--p-max', type=float, default=0.05)
    ap.add_argument('--enrichment-min', type=float, default=1.0, help='Minimum log2 enrichment over leave-one-out background.')
    ap.add_argument('--epsilon', type=float, default=1e-9)
    ap.add_argument('--top-k-nodes', type=int, default=500)
    ap.add_argument('--no-edges', action='store_true')
    ap.add_argument('--edge-mode', default='cluster', choices=['cluster', 'all_vs_all'])
    ap.add_argument('--edge-p-min', type=float, default=None)
    ap.add_argument('--edge-p-max', type=float, default=None)
    ap.add_argument('--edge-enrichment-min', type=float, default=None)
    ap.add_argument('--delta-pmi-min', type=float, default=None)
    ap.add_argument('--top-k-edges', type=int, default=1000)
    ap.add_argument('--progress', action='store_true')
    args = ap.parse_args(argv)

    root = (args.root or os.getenv('PHASE_DATA_ROOT') or '').strip()
    if not root:
        root = str((Path(__file__).resolve().parents[2] / 'data').resolve())
    os.environ['PHASE_DATA_ROOT'] = root

    sample_ids = _parse_csv(args.sample_ids)
    if len(sample_ids) < 2:
        raise SystemExit('Select at least two --sample-ids entries.')

    def progress(message: str, current: int, total: int):
        if args.progress:
            print(f'[transient_states] {message}: {current}/{max(1, total)}', flush=True)

    out = run_transient_state_analysis(
        project_id=str(args.project_id),
        system_id=str(args.system_id),
        cluster_id=str(args.cluster_id),
        sample_ids=sample_ids,
        md_label_mode=str(args.md_label_mode),
        drop_invalid=not bool(args.keep_invalid),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        enrichment_min=float(args.enrichment_min),
        epsilon=float(args.epsilon),
        top_k_nodes=int(args.top_k_nodes),
        include_edges=not bool(args.no_edges),
        edge_mode=str(args.edge_mode),
        edge_p_min=args.edge_p_min,
        edge_p_max=args.edge_p_max,
        edge_enrichment_min=args.edge_enrichment_min,
        delta_pmi_min=args.delta_pmi_min,
        top_k_edges=int(args.top_k_edges),
        progress_callback=progress if args.progress else None,
    )
    meta = out.get('metadata') or {}
    summary = meta.get('summary') or {}
    print(f"[transient_states] analysis_id: {meta.get('analysis_id')}")
    print(f"[transient_states] analysis_npz: {out.get('analysis_npz')}")
    print(f"[transient_states] analysis_dir: {out.get('analysis_dir')}")
    print(f"[transient_states] node_hits={summary.get('n_node_hits')} edge_hits={summary.get('n_edge_hits')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
