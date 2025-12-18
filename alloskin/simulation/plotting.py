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
        <label for="top-n">Top N:</label>
        <input id="top-n" type="number" min="1" value="30" style="width:70px;" />
        <label for="top-metric">by</label>
        <select id="top-metric">
          <option value="js">JS divergence</option>
          <option value="err_g">|Gibbs - MD|</option>
          <option value="err_sa">|SA-QUBO - MD|</option>
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

    function sliceRows(mat, idxs) {
      return idxs.map(i => mat[i]);
    }
    function sliceVec(vec, idxs) {
      return idxs.map(i => vec[i]);
    }

    function buildTraces(idxs) {
      const yLabels = idxs.map(i => payload.labels[i]);
      const md = sliceRows(payload.md, idxs);
      const gibbs = sliceRows(payload.gibbs, idxs);
      const sa = sliceRows(payload.sa, idxs);
      const errG = sliceRows(payload.err_g, idxs);
      const errSA = sliceRows(payload.err_sa, idxs);
      const jsG = sliceVec(payload.js_g, idxs);
      const jsSA = sliceVec(payload.js_sa, idxs);

      return [
        {type: "heatmap", x: stateLabels, y: yLabels, z: md, xaxis: "x1", yaxis: "y1", zmin: 0, zmax: 1, colorscale: "Viridis", colorbar: {title: "p(state)", len: 0.25, y: 0.83, thickness: 12} , name: "MD"},
        {type: "heatmap", x: stateLabels, y: yLabels, z: gibbs, xaxis: "x2", yaxis: "y2", zmin: 0, zmax: 1, colorscale: "Viridis", showscale: false, name: "Gibbs"},
        {type: "heatmap", x: stateLabels, y: yLabels, z: sa, xaxis: "x3", yaxis: "y3", zmin: 0, zmax: 1, colorscale: "Viridis", showscale: false, name: "SA-QUBO"},
        {type: "heatmap", x: stateLabels, y: yLabels, z: errG, xaxis: "x4", yaxis: "y4", zmin: 0, zmax: payload.vmax_err, colorscale: "Magma", colorbar: {title: "|Gibbs-MD|", len: 0.22, y: 0.48, thickness: 12} , name: "Err Gibbs"},
        {type: "heatmap", x: stateLabels, y: yLabels, z: errSA, xaxis: "x5", yaxis: "y5", zmin: 0, zmax: payload.vmax_err, colorscale: "Magma", showscale: false, name: "Err SA-QUBO"},
        {type: "bar", x: yLabels, y: jsG, xaxis: "x6", yaxis: "y6", name: "JS Gibbs", marker: {color: "#4472c4"}},
        {type: "bar", x: yLabels, y: jsSA, xaxis: "x6", yaxis: "y6", name: "JS SA-QUBO", marker: {color: "#ed7d31"}},
      ];
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

    document.getElementById("btn-all").addEventListener("click", () => setSelected(allIdx));
    document.getElementById("btn-clear").addEventListener("click", () => setSelected([]));
    document.getElementById("btn-top").addEventListener("click", () => {
      const n = Math.max(1, parseInt(topNInput.value || "30", 10));
      const metric = topMetric.value;
      let scores;
      if (metric === "err_g") {
        scores = payload.err_g_max;
      } else if (metric === "err_sa") {
        scores = payload.err_sa_max;
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
    residue_labels: Iterable[object],
    out_path: str | Path,
    annotate: bool = True,  # kept for backward compatibility; unused
) -> Path:
    """
    Save an interactive Plotly HTML comparing marginals from MD vs sampled models.
    Includes a multi-select to filter which residues are shown.
    """
    md_mat = _to_matrix(p_md)
    g_mat = _to_matrix(p_gibbs)
    sa_mat = _to_matrix(p_sa)
    err_g = np.abs(g_mat - md_mat)
    err_sa = np.abs(sa_mat - md_mat)

    res_labels = _coerce_labels(residue_labels, md_mat.shape[0])
    vmax_err = float(np.nanmax([err_g, err_sa]))

    payload = {
        "labels": res_labels,
        "stateCount": int(md_mat.shape[1]),
        "md": md_mat.tolist(),
        "gibbs": g_mat.tolist(),
        "sa": sa_mat.tolist(),
        "err_g": err_g.tolist(),
        "err_sa": err_sa.tolist(),
        "err_g_max": np.nanmax(err_g, axis=1).tolist(),
        "err_sa_max": np.nanmax(err_sa, axis=1).tolist(),
        "js_g": np.asarray(js_gibbs, dtype=float).tolist(),
        "js_sa": np.asarray(js_sa, dtype=float).tolist(),
        "js_max": np.maximum(js_gibbs, js_sa).tolist(),
        "vmax_err": vmax_err,
    }

    # Custom layout with 3 rows: top row (MD/Gibbs/SA), middle row (errors), bottom row (JS bars)
    layout = {
        "title": {"text": "Marginal comparison: MD vs sampled models", "x": 0.5},
        "height": 980,
        "margin": {"l": 70, "r": 20, "t": 60, "b": 60},
        "barmode": "group",
        # Row 1 domains
        "xaxis": {"domain": [0.0, 0.32], "anchor": "y", "title": "state"},
        "yaxis": {"domain": [0.67, 1.0], "title": "residue", "automargin": True},
        "xaxis2": {"domain": [0.34, 0.66], "anchor": "y2", "title": "state"},
        "yaxis2": {"domain": [0.67, 1.0], "showticklabels": False},
        "xaxis3": {"domain": [0.68, 1.0], "anchor": "y3", "title": "state"},
        "yaxis3": {"domain": [0.67, 1.0], "showticklabels": False},
        # Row 2 domains
        "xaxis4": {"domain": [0.0, 0.32], "anchor": "y4", "title": "state"},
        "yaxis4": {"domain": [0.34, 0.64], "title": "residue", "automargin": True},
        "xaxis5": {"domain": [0.34, 0.66], "anchor": "y5", "title": "state"},
        "yaxis5": {"domain": [0.34, 0.64], "showticklabels": False},
        # Row 3 (JS bars across full width)
        "xaxis6": {"domain": [0.0, 1.0], "anchor": "y6", "title": "residue"},
        "yaxis6": {"domain": [0.0, 0.28], "title": "JS divergence", "automargin": True},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
    }

    html = _html_template(fig_layout=json.dumps(layout), payload=json.dumps(payload))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path



def plot_beta_scan_curve(
    *,
    betas: Sequence[float],
    distances: Sequence[float],
    out_path: str | Path,
    title: str = "Effective temperature calibration: distance vs beta",
) -> Path:
    """
    Save a small interactive HTML plot of D(beta), used to pick beta_eff.

    This keeps dependencies minimal by reusing the same Plotly-in-HTML pattern
    used by plot_marginal_summary.
    """
    betas = [float(b) for b in betas]
    distances = [float(d) for d in distances]
    payload = {
        "betas": betas,
        "distances": distances,
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
    <div class="note">We pick \\(\\beta_\\mathrm{{eff}}\\) as the minimizer of the distance curve.</div>
    <div id="plot"></div>
  </div>

  <script>
    const payload = {json.dumps(payload)};
    const x = payload.betas;
    const y = payload.distances;

    const trace = {{
      x: x,
      y: y,
      mode: "lines+markers",
      name: "D(beta)"
    }};

    const layout = {{
      xaxis: {{ title: "beta", type: "linear" }},
      yaxis: {{ title: "distance", type: "linear" }},
      margin: {{ l: 60, r: 20, t: 30, b: 50 }},
    }};

    Plotly.newPlot("plot", [trace], layout, {{responsive: true}});
  </script>
</body>
</html>
"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
