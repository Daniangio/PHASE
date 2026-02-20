from __future__ import annotations

import argparse
import os
from pathlib import Path

from phase.potts.analysis_run import upsert_delta_js_analysis


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run incremental delta-JS A/B/Other analysis (model-pair optional)."
    )
    ap.add_argument("--root", default="", help="PHASE data root (defaults to $PHASE_DATA_ROOT or ./data).")
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--system-id", required=True)
    ap.add_argument("--cluster-id", required=True)

    ap.add_argument("--model-a-id", default="", help="Optional model A id (must be paired with --model-b-id).")
    ap.add_argument("--model-b-id", default="", help="Optional model B id (must be paired with --model-a-id).")
    ap.add_argument("--sample-ids", required=True, help="Comma-separated sample ids to compute/append.")
    ap.add_argument("--ref-a-sample-ids", default="", help="Optional comma-separated reference sample ids for side A.")
    ap.add_argument("--ref-b-sample-ids", default="", help="Optional comma-separated reference sample ids for side B.")

    ap.add_argument("--md-label-mode", default="assigned", choices=["assigned", "halo"])
    ap.add_argument("--keep-invalid", action="store_true")
    ap.add_argument("--top-k-residues", type=int, default=20)
    ap.add_argument("--top-k-edges", type=int, default=30)
    ap.add_argument("--node-edge-alpha", type=float, default=0.5)
    ap.add_argument(
        "--edge-mode",
        default="",
        choices=["", "cluster", "all_vs_all", "contact"],
        help="Required in model-free mode: cluster | all_vs_all | contact.",
    )
    ap.add_argument("--contact-state-ids", default="", help="Comma-separated state IDs for contact PDB lookup.")
    ap.add_argument("--contact-pdbs", default="", help="Comma-separated PDB paths for contact edges.")
    ap.add_argument("--contact-cutoff", type=float, default=10.0)
    ap.add_argument("--contact-atom-mode", default="CA", choices=["CA", "CM"])
    args = ap.parse_args(argv)

    root = (args.root or os.getenv("PHASE_DATA_ROOT") or "").strip()
    if not root:
        root = str((Path(__file__).resolve().parents[2] / "data").resolve())
    os.environ["PHASE_DATA_ROOT"] = root

    sample_ids = _parse_csv(args.sample_ids)
    if not sample_ids:
        raise SystemExit("Provide at least one --sample-ids entry.")
    ref_a = _parse_csv(args.ref_a_sample_ids)
    ref_b = _parse_csv(args.ref_b_sample_ids)
    model_a_id = str(args.model_a_id or "").strip()
    model_b_id = str(args.model_b_id or "").strip()
    using_models = bool(model_a_id or model_b_id)
    edge_mode = str(args.edge_mode or "").strip().lower()
    contact_state_ids = _parse_csv(args.contact_state_ids)
    contact_pdbs = _parse_csv(args.contact_pdbs)
    if using_models and (not model_a_id or not model_b_id):
        raise SystemExit("Provide both --model-a-id and --model-b-id, or neither.")
    if using_models and model_a_id == model_b_id:
        raise SystemExit("--model-a-id and --model-b-id must be different.")
    if not using_models and (not ref_a or not ref_b):
        raise SystemExit(
            "When models are not provided, pass both --ref-a-sample-ids and --ref-b-sample-ids."
        )
    if not using_models and not edge_mode:
        raise SystemExit("When models are not provided, --edge-mode is required.")
    if edge_mode == "contact" and not (contact_state_ids or contact_pdbs):
        raise SystemExit("--edge-mode contact requires --contact-state-ids and/or --contact-pdbs.")

    out = upsert_delta_js_analysis(
        project_id=str(args.project_id),
        system_id=str(args.system_id),
        cluster_id=str(args.cluster_id),
        model_a_ref=(model_a_id or None),
        model_b_ref=(model_b_id or None),
        sample_ids=sample_ids,
        reference_sample_ids_a=ref_a or None,
        reference_sample_ids_b=ref_b or None,
        md_label_mode=str(args.md_label_mode),
        drop_invalid=not bool(args.keep_invalid),
        top_k_residues=int(args.top_k_residues),
        top_k_edges=int(args.top_k_edges),
        node_edge_alpha=float(args.node_edge_alpha),
        edge_mode=(edge_mode or None),
        contact_state_ids=contact_state_ids or None,
        contact_pdbs=contact_pdbs or None,
        contact_cutoff=float(args.contact_cutoff),
        contact_atom_mode=str(args.contact_atom_mode).upper(),
    )
    meta = out.get("metadata") or {}
    print(f"[delta_js] analysis_id: {meta.get('analysis_id')}")
    print(f"[delta_js] analysis_npz: {out.get('analysis_npz')}")
    print(f"[delta_js] analysis_dir: {out.get('analysis_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
