from __future__ import annotations

from pathlib import Path
from typing import Sequence
import tempfile

import numpy as np

from phase.potts.plotting import plot_sampling_report_from_npz


def _normalize_id_list(raw: Sequence[object] | str | None) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return parts
    out: list[str] = []
    for item in raw:
        if item is None:
            continue
        value = str(item).strip()
        if value:
            out.append(value)
    return out


def _resolve_indices(all_ids: np.ndarray, keep_ids: Sequence[object] | str | None, *, label: str) -> np.ndarray | None:
    if keep_ids is None:
        return None
    keep_list = _normalize_id_list(keep_ids) or []
    if not keep_list:
        raise ValueError(f"No {label} ids provided.")
    id_to_idx = {str(v): i for i, v in enumerate(all_ids.tolist())}
    missing = [src_id for src_id in keep_list if src_id not in id_to_idx]
    if missing:
        raise ValueError(f"Unknown {label} ids: {missing}")
    return np.asarray([id_to_idx[src_id] for src_id in keep_list], dtype=int)


def list_sampling_sources(summary_path: str | Path) -> dict[str, list[dict[str, object]]]:
    summary_path = Path(summary_path)
    with np.load(summary_path, allow_pickle=False) as data:
        md_ids = data["md_source_ids"] if "md_source_ids" in data else np.array([], dtype=str)
        md_labels = data["md_source_labels"] if "md_source_labels" in data else np.array([], dtype=str)
        md_types = data["md_source_types"] if "md_source_types" in data else np.array([], dtype=str)
        md_counts = data["md_source_counts"] if "md_source_counts" in data else np.array([], dtype=int)

        sample_ids = data["sample_source_ids"] if "sample_source_ids" in data else np.array([], dtype=str)
        sample_labels = data["sample_source_labels"] if "sample_source_labels" in data else np.array([], dtype=str)
        sample_types = data["sample_source_types"] if "sample_source_types" in data else np.array([], dtype=str)
        sample_counts = data["sample_source_counts"] if "sample_source_counts" in data else np.array([], dtype=int)

    md_sources: list[dict[str, object]] = []
    for idx, src_id in enumerate(md_ids.tolist()):
        md_sources.append(
            {
                "id": str(src_id),
                "label": str(md_labels[idx]) if idx < len(md_labels) else str(src_id),
                "type": str(md_types[idx]) if idx < len(md_types) else "",
                "count": int(md_counts[idx]) if idx < len(md_counts) else 0,
            }
        )

    sample_sources: list[dict[str, object]] = []
    for idx, src_id in enumerate(sample_ids.tolist()):
        sample_sources.append(
            {
                "id": str(src_id),
                "label": str(sample_labels[idx]) if idx < len(sample_labels) else str(src_id),
                "type": str(sample_types[idx]) if idx < len(sample_types) else "",
                "count": int(sample_counts[idx]) if idx < len(sample_counts) else 0,
            }
        )

    return {"md_sources": md_sources, "sample_sources": sample_sources}


def build_filtered_summary(
    summary_path: str | Path,
    *,
    md_source_ids: Sequence[object] | str | None = None,
    sample_source_ids: Sequence[object] | str | None = None,
    out_path: str | Path | None = None,
) -> Path:
    summary_path = Path(summary_path)
    with np.load(summary_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    md_ids = arrays.get("md_source_ids")
    sample_ids = arrays.get("sample_source_ids")
    if md_ids is None or sample_ids is None:
        raise ValueError("Summary file is missing md_source_ids/sample_source_ids.")

    md_idx = _resolve_indices(np.asarray(md_ids), md_source_ids, label="md_source")
    sample_idx = _resolve_indices(np.asarray(sample_ids), sample_source_ids, label="sample_source")

    def _slice_first_dim(key: str, idx: np.ndarray | None) -> None:
        if idx is None:
            return
        arr = arrays.get(key)
        if arr is None or getattr(arr, "size", 0) == 0:
            return
        arrays[key] = arr[idx]

    def _slice_md_sample(key: str) -> None:
        arr = arrays.get(key)
        if arr is None or getattr(arr, "size", 0) == 0:
            return
        md_sel = md_idx if md_idx is not None else np.arange(arr.shape[0])
        sample_sel = sample_idx if sample_idx is not None else np.arange(arr.shape[1])
        arrays[key] = arr[np.ix_(md_sel, sample_sel)]

    _slice_first_dim("md_source_ids", md_idx)
    _slice_first_dim("md_source_labels", md_idx)
    _slice_first_dim("md_source_types", md_idx)
    _slice_first_dim("md_source_counts", md_idx)
    _slice_first_dim("p_md_by_source", md_idx)
    _slice_first_dim("energy_hist_md", md_idx)
    _slice_first_dim("energy_cdf_md", md_idx)

    _slice_first_dim("sample_source_ids", sample_idx)
    _slice_first_dim("sample_source_labels", sample_idx)
    _slice_first_dim("sample_source_types", sample_idx)
    _slice_first_dim("sample_source_counts", sample_idx)
    _slice_first_dim("p_sample_by_source", sample_idx)
    _slice_first_dim("js_gibbs_sample", sample_idx)
    _slice_first_dim("js2_gibbs_sample", sample_idx)
    _slice_first_dim("energy_hist_sample", sample_idx)
    _slice_first_dim("energy_cdf_sample", sample_idx)

    _slice_md_sample("js_md_sample")
    _slice_md_sample("js2_md_sample")
    _slice_md_sample("nn_cdf_sample_to_md")
    _slice_md_sample("nn_cdf_md_to_sample")

    out_path = Path(out_path) if out_path else Path(tempfile.mkdtemp()) / "run_summary_filtered.npz"
    np.savez_compressed(out_path, **arrays)
    return out_path


def sampling_report_html(
    summary_path: str | Path,
    *,
    md_source_ids: Sequence[object] | str | None = None,
    sample_source_ids: Sequence[object] | str | None = None,
    offline: bool = False,
) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        filtered_summary = build_filtered_summary(
            summary_path,
            md_source_ids=md_source_ids,
            sample_source_ids=sample_source_ids,
            out_path=tmpdir_path / "run_summary_filtered.npz",
        )
        html_path = tmpdir_path / "sampling_report.html"
        plot_sampling_report_from_npz(
            summary_path=filtered_summary,
            out_path=html_path,
            offline=offline,
        )
        return html_path.read_text(encoding="utf-8")


def show_sampling_report(
    summary_path: str | Path,
    *,
    md_source_ids: Sequence[object] | str | None = None,
    sample_source_ids: Sequence[object] | str | None = None,
    offline: bool = False,
) -> str:
    html = sampling_report_html(
        summary_path,
        md_source_ids=md_source_ids,
        sample_source_ids=sample_source_ids,
        offline=offline,
    )
    try:
        from IPython.display import HTML, display

        display(HTML(html))
    except Exception:
        pass
    return html
