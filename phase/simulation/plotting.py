from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def _to_matrix(marginals: Sequence[np.ndarray]) -> np.ndarray:
    """
    Pads a list of 1D marginal arrays into a 2D matrix (residues x states).
    Missing states are filled with NaN so they render as masked cells.
    """
    if len(marginals) == 0:
        raise ValueError("No marginals provided.")
    max_k = max(len(p) for p in marginals)
    mat = np.full((len(marginals), max_k), np.nan, dtype=float)
    for i, p in enumerate(marginals):
        mat[i, : len(p)] = p
    return mat


def _coerce_labels(residue_keys: Iterable[object], n: int) -> List[str]:
    labels = [str(k) for k in residue_keys]
    if len(labels) != n:
        labels = [str(i) for i in range(n)]
    return labels


def _ensure_matrix(marginals: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(marginals, np.ndarray):
        if marginals.ndim != 2:
            raise ValueError("Expected 2D array for marginals.")
        return marginals.astype(float)
    return _to_matrix(marginals)


def _html_template(fig_layout: str, payload: str, div_id: str = "marginal-fig") -> str:
    """
    Compose an interactive HTML page with a residue selector that updates Plotly subplots.
    """
    template = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Marginal comparison</title>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
    .wrap { display: flex; flex-direction: row; height: 100vh; }
    .controls { width: 380px; padding: 14px; border-right: 1px solid #ccc; box-sizing: border-box; overflow-y: auto; }
    .controls h2 { margin-top: 0; font-size: 18px; }
    .controls button { margin: 4px 4px 4px 0; padding: 6px 10px; }
    .control-row { margin: 6px 0; display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
    .control-row select { max-width: 280px; }
    #residue-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 6px; max-height: 60vh; overflow-y: auto; border: 1px solid #ccc; padding: 6px; }
    #residue-grid label { display: flex; align-items: center; gap: 6px; font-size: 13px; padding: 2px 4px; }
    #status { margin-top: 8px; font-size: 12px; color: #444; }
    .figure { flex: 1; padding: 8px; box-sizing: border-box; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="controls">
      <h2>Residue filter</h2>
      <p>Select residues to display in the heatmaps/bar plot. Leave empty to show all.</p>
      <div class="control-row">
        <button id="btn-all" type="button">Select all</button>
        <button id="btn-clear" type="button">Clear</button>
      </div>
      <div class="control-row">
        <label for="sampler-select">Sampler:</label>
        <select id="sampler-select"></select>
      </div>
      <div class="control-row">
        <label for="top-n">Top N:</label>
        <input id="top-n" type="number" min="1" value="30" style="width:70px;" />
        <label for="top-metric">by</label>
        <select id="top-metric">
          <option value="js">JS divergence (max)</option>
          <option value="err">|Sample - MD|</option>
        </select>
        <button id="btn-top" type="button">Select top</button>
      </div>
      <div id="residue-grid"></div>
      <div id="status"></div>
    </div>
    <div class="figure">
      <div id="__DIV_ID__"></div>
    </div>
  </div>

  <script>
    const payload = __PAYLOAD__;
    const baseLayout = __LAYOUT__;
    const stateLabels = Array.from({length: payload.stateCount}, (_, i) => i.toString());
    const allIdx = payload.labels.map((_, i) => i);
    const statusEl = document.getElementById("status");
    const figId = "__DIV_ID__";
    const grid = document.getElementById("residue-grid");
    const topNInput = document.getElementById("top-n");
    const topMetric = document.getElementById("top-metric");
    const samplerSelect = document.getElementById("sampler-select");

    // populate selector as checkboxes in a responsive grid
    const checkboxes = [];
    payload.labels.forEach((lbl, idx) => {
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = idx.toString();
      label.appendChild(cb);
      const txt = document.createElement("span");
      txt.textContent = `${idx} — ${lbl}`;
      label.appendChild(txt);
      grid.appendChild(label);
      checkboxes.push(cb);
    });

    payload.sources.forEach((src, idx) => {
      const opt = document.createElement("option");
      opt.value = src.id;
      opt.textContent = src.label;
      if (idx === 0) opt.selected = true;
      samplerSelect.appendChild(opt);
    });

    function sliceRows(mat, idxs) {
      return idxs.map(i => mat[i]);
    }
    function sliceVec(vec, idxs) {
      return idxs.map(i => vec[i]);
    }

    function getActiveSource() {
      const id = samplerSelect.value || (payload.sources[0] && payload.sources[0].id);
      return payload.sources.find(src => src.id === id) || payload.sources[0];
    }

    function buildTraces(idxs) {
      const yLabels = idxs.map(i => payload.labels[i]);
      const md = sliceRows(payload.md, idxs);
      const active = getActiveSource();
      const sample = sliceRows(active.matrix, idxs);
      const errSample = sliceRows(active.err, idxs);
      const traces = [
        {type: "heatmap", x: stateLabels, y: yLabels, z: md, xaxis: "x1", yaxis: "y1", zmin: 0, zmax: 1, colorscale: "Viridis", colorbar: {title: "p(state)", len: 0.35, y: 0.82, thickness: 12} , name: "MD"},
        {type: "heatmap", x: stateLabels, y: yLabels, z: sample, xaxis: "x2", yaxis: "y2", zmin: 0, zmax: 1, colorscale: "Viridis", showscale: false, name: active.label},
        {type: "heatmap", x: stateLabels, y: yLabels, z: errSample, xaxis: "x3", yaxis: "y3", zmin: 0, zmax: payload.vmax_err, colorscale: "Magma", colorbar: {title: "|Sample-MD|", len: 0.25, y: 0.45, thickness: 12} , name: "Error"},
      ];

      payload.js_series.forEach((series, idx) => {
        const jsVals = sliceVec(series.values, idxs);
        traces.push({
          type: "bar",
          x: yLabels,
          y: jsVals,
          xaxis: "x4",
          yaxis: "y4",
          name: series.label,
          marker: {color: series.color || undefined},
        });
      });
      return traces;
    }

    function setStatus(idxs) {
      statusEl.textContent = `Showing ${idxs.length} / ${allIdx.length} residues`;
    }

    function applySelection(idxs) {
      const useIdxs = idxs.length ? idxs : allIdx;
      const traces = buildTraces(useIdxs);
      Plotly.react(figId, traces, baseLayout, {responsive: true, displaylogo: false});
      setStatus(useIdxs);
    }

    function setSelected(idxs) {
      const toSelect = new Set(idxs.map(String));
      checkboxes.forEach(cb => {
        cb.checked = toSelect.has(cb.value);
      });
      applySelection(idxs);
    }

    grid.addEventListener("change", () => {
      const idxs = checkboxes.filter(cb => cb.checked).map(cb => parseInt(cb.value, 10));
      applySelection(idxs);
    });
    samplerSelect.addEventListener("change", () => {
      const idxs = checkboxes.filter(cb => cb.checked).map(cb => parseInt(cb.value, 10));
      applySelection(idxs);
    });

    document.getElementById("btn-all").addEventListener("click", () => setSelected(allIdx));
    document.getElementById("btn-clear").addEventListener("click", () => setSelected([]));
    document.getElementById("btn-top").addEventListener("click", () => {
      const n = Math.max(1, parseInt(topNInput.value || "30", 10));
      const metric = topMetric.value;
      let scores;
      if (metric === "err") {
        const active = getActiveSource();
        scores = active.err_max;
      } else {
        scores = payload.js_max;
      }
      const ranked = allIdx.slice().sort((a, b) => scores[b] - scores[a]);
      const pick = ranked.slice(0, Math.min(n, ranked.length));
      setSelected(pick);
    });

    // initial render
    Plotly.newPlot(figId, buildTraces(allIdx), baseLayout, {responsive: true, displaylogo: false});
    setStatus(allIdx);
  </script>
</body>
</html>
"""
    return template.replace("__DIV_ID__", div_id).replace("__PAYLOAD__", payload).replace("__LAYOUT__", fig_layout)


def plot_marginal_summary(
    *,
    p_md: Sequence[np.ndarray],
    p_gibbs: Sequence[np.ndarray],
    p_sa: Sequence[np.ndarray],
    js_gibbs: np.ndarray,
    js_sa: np.ndarray,
    betas: Sequence[float] | None = None,
    p_gibbs_by_beta: np.ndarray | None = None,
    js_gibbs_by_beta: np.ndarray | None = None,
    sa_schedule_labels: Sequence[str] | None = None,
    p_sa_by_schedule: np.ndarray | None = None,
    js_sa_by_schedule: np.ndarray | None = None,
    residue_labels: Iterable[object],
    out_path: str | Path,
    annotate: bool = True,  # kept for backward compatibility; unused
) -> Path:
    """
    Save an interactive Plotly HTML comparing marginals from MD vs sampled models.
    Includes a multi-select to filter which residues are shown.
    """
    md_mat = _ensure_matrix(p_md)
    res_labels = _coerce_labels(residue_labels, md_mat.shape[0])

    sources = []
    js_series = []

    def _add_source(src_id: str, label: str, matrix: np.ndarray, js_vec: np.ndarray, color: str | None = None) -> None:
        err = np.abs(matrix - md_mat)
        sources.append({
            "id": src_id,
            "label": label,
            "matrix": matrix.tolist(),
            "err": err.tolist(),
            "err_max": np.nanmax(err, axis=1).tolist(),
            "js": np.asarray(js_vec, dtype=float).tolist(),
        })
        if color:
            js_series.append({"label": label, "values": np.asarray(js_vec, dtype=float).tolist(), "color": color})
        else:
            js_series.append({"label": label, "values": np.asarray(js_vec, dtype=float).tolist()})

    has_sa_grid = (
        p_sa_by_schedule is not None
        and js_sa_by_schedule is not None
        and sa_schedule_labels is not None
        and len(sa_schedule_labels)
        and np.size(p_sa_by_schedule) > 0
        and np.size(js_sa_by_schedule) > 0
    )
    if has_sa_grid:
        sa_palette = [
            "#f97316",
            "#fb923c",
            "#f59e0b",
            "#facc15",
            "#fbbf24",
        ]
        for i, label in enumerate(sa_schedule_labels):
            if i >= len(p_sa_by_schedule) or i >= len(js_sa_by_schedule):
                break
            color = sa_palette[i % len(sa_palette)]
            _add_source(
                f"sa_{i}",
                str(label),
                np.asarray(p_sa_by_schedule[i], dtype=float),
                js_sa_by_schedule[i],
                color=color,
            )
    else:
        sa_mat = _ensure_matrix(p_sa)
        _add_source("sa", "SA-QUBO", sa_mat, js_sa, color="#ed7d31")

    has_beta_grid = (
        p_gibbs_by_beta is not None
        and js_gibbs_by_beta is not None
        and betas is not None
        and len(betas)
        and np.size(p_gibbs_by_beta) > 0
        and np.size(js_gibbs_by_beta) > 0
    )
    if has_beta_grid:
        beta_palette = [
            "#2563eb",
            "#16a34a",
            "#f59e0b",
            "#db2777",
            "#06b6d4",
            "#7c3aed",
            "#f97316",
            "#0ea5e9",
            "#84cc16",
            "#e11d48",
        ]
        for i, b in enumerate(betas):
            if i >= len(p_gibbs_by_beta) or i >= len(js_gibbs_by_beta):
                break
            label = f"Gibbs β={float(b):g}"
            color = beta_palette[i % len(beta_palette)]
            _add_source(
                f"gibbs_{i}",
                label,
                np.asarray(p_gibbs_by_beta[i], dtype=float),
                js_gibbs_by_beta[i],
                color=color,
            )
    else:
        g_mat = _ensure_matrix(p_gibbs)
        label = "Gibbs"
        if betas:
            label = f"Gibbs β={float(betas[0]):g}"
        _add_source("gibbs", label, g_mat, js_gibbs, color="#4472c4")

    js_max = np.zeros(md_mat.shape[0], dtype=float)
    for src in sources:
        js_max = np.maximum(js_max, np.asarray(src["js"], dtype=float))

    payload = {
        "labels": res_labels,
        "stateCount": int(md_mat.shape[1]),
        "md": md_mat.tolist(),
        "sources": sources,
        "js_series": js_series,
        "js_max": js_max.tolist(),
        "vmax_err": float(np.nanmax([np.nanmax(np.asarray(src["err"], dtype=float)) for src in sources])),
    }

    # Custom layout with 3 rows: top row (MD + selected sampler), middle row (error), bottom row (JS bars)
    layout = {
        "title": {"text": "Marginal comparison: MD vs sampled models", "x": 0.5},
        "height": 920,
        "margin": {"l": 70, "r": 20, "t": 60, "b": 60},
        "barmode": "group",
        # Row 1 domains
        "xaxis": {"domain": [0.0, 0.48], "anchor": "y", "title": "state"},
        "yaxis": {"domain": [0.67, 1.0], "title": "residue", "automargin": True},
        "xaxis2": {"domain": [0.52, 1.0], "anchor": "y2", "title": "state"},
        "yaxis2": {"domain": [0.67, 1.0], "showticklabels": False},
        # Row 2 domains
        "xaxis3": {"domain": [0.0, 1.0], "anchor": "y3", "title": "state"},
        "yaxis3": {"domain": [0.34, 0.64], "title": "residue", "automargin": True},
        # Row 3 (JS bars across full width)
        "xaxis4": {"domain": [0.0, 1.0], "anchor": "y4", "title": "residue"},
        "yaxis4": {"domain": [0.0, 0.28], "title": "JS divergence", "automargin": True},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
    }

    html = _html_template(fig_layout=json.dumps(layout), payload=json.dumps(payload))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path



def plot_marginal_summary_from_npz(
    *,
    summary_path: str | Path,
    out_path: str | Path,
    annotate: bool = True,  # unused, kept for symmetry with plot_marginal_summary
) -> Path:
    """
    Convenience loader: read a run_summary.npz bundle and render the marginal dashboard from it.
    """
    with np.load(summary_path, allow_pickle=False) as data:
        required = ["p_md", "p_gibbs", "p_sa", "js_gibbs", "js_sa", "residue_labels"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing keys in summary file {summary_path}: {missing}")

        p_md = data["p_md"]
        p_gibbs = data["p_gibbs"]
        p_sa = data["p_sa"]
        js_gibbs = data["js_gibbs"]
        js_sa = data["js_sa"]
        residue_labels = data["residue_labels"]
        betas = data["betas"] if "betas" in data else []
        p_gibbs_by_beta = data["p_gibbs_by_beta"] if "p_gibbs_by_beta" in data else None
        js_gibbs_by_beta = data["js_gibbs_by_beta"] if "js_gibbs_by_beta" in data else None
        sa_schedule_labels = data["sa_schedule_labels"] if "sa_schedule_labels" in data else None
        p_sa_by_schedule = data["p_sa_by_schedule"] if "p_sa_by_schedule" in data else None
        js_sa_by_schedule = data["js_sa_by_schedule"] if "js_sa_by_schedule" in data else None

        if isinstance(sa_schedule_labels, np.ndarray) and sa_schedule_labels.size == 0:
            sa_schedule_labels = None
        if isinstance(p_sa_by_schedule, np.ndarray) and p_sa_by_schedule.size == 0:
            p_sa_by_schedule = None
        if isinstance(js_sa_by_schedule, np.ndarray) and js_sa_by_schedule.size == 0:
            js_sa_by_schedule = None

    return plot_marginal_summary(
        p_md=p_md,
        p_gibbs=p_gibbs,
        p_sa=p_sa,
        js_gibbs=js_gibbs,
        js_sa=js_sa,
        betas=betas,
        p_gibbs_by_beta=p_gibbs_by_beta,
        js_gibbs_by_beta=js_gibbs_by_beta,
        sa_schedule_labels=sa_schedule_labels,
        p_sa_by_schedule=p_sa_by_schedule,
        js_sa_by_schedule=js_sa_by_schedule,
        residue_labels=residue_labels,
        out_path=out_path,
        annotate=annotate,
    )


def plot_beta_scan_curve(
    *,
    betas: Sequence[float],
    distances: Sequence[float] | Sequence[Sequence[float]],
    out_path: str | Path,
    title: str = "Effective temperature calibration: distance vs beta",
    labels: Sequence[str] | None = None,
) -> Path:
    """
    Save a small interactive HTML plot of D(beta), used to pick beta_eff.
    Supports multiple curves (one per SA schedule).

    This keeps dependencies minimal by reusing the same Plotly-in-HTML pattern
    used by plot_marginal_summary.
    """
    betas = [float(b) for b in betas]
    if len(distances) and isinstance(distances[0], (list, tuple, np.ndarray)):
        series = [list(map(float, seq)) for seq in distances]  # type: ignore[arg-type]
    else:
        series = [list(map(float, distances))]  # type: ignore[list-item]

    if labels is None or len(labels) != len(series):
        labels = [f"SA {i + 1}" for i in range(len(series))]

    payload = {
        "betas": betas,
        "series": series,
        "labels": list(labels),
        "title": title,
    }

    # Tiny HTML template (standalone). Uses Plotly CDN like the marginal dashboard.
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 0; padding: 0; }}
    .wrap {{ padding: 12px; }}
    #plot {{ width: 100%; height: 520px; }}
    .note {{ color: #444; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h2>{title}</h2>
    <div class="note">We pick \\(\\beta_\\mathrm{{eff}}\\) per schedule as the minimizer of each distance curve.</div>
    <div id="plot"></div>
  </div>

  <script>
    const payload = {json.dumps(payload)};
    const x = payload.betas;
    const traces = payload.series.map((y, idx) => ({{
      x: x,
      y: y,
      mode: "lines+markers",
      name: payload.labels[idx] || ("SA " + (idx + 1))
    }}));

    const layout = {{
      xaxis: {{ title: "beta", type: "linear" }},
      yaxis: {{ title: "distance", type: "linear" }},
      margin: {{ l: 60, r: 20, t: 30, b: 50 }},
    }};

    Plotly.newPlot("plot", traces, layout, {{responsive: true}});
  </script>
</body>
</html>
"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
