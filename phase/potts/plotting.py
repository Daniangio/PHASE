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
      <div class="control-row">
        <label><input id="js-normalize" type="checkbox" /> Normalize JS by log(K)</label>
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
    const jsNormalize = document.getElementById("js-normalize");

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
    function getKList() {
      return payload.md.map(row => row.reduce((acc, v) => acc + (Number.isFinite(v) ? 1 : 0), 0));
    }
    function normalizeByLogK(values, idxs, kList) {
      return values.map((v, i) => {
        const k = kList[idxs[i]];
        const denom = k > 1 ? Math.log(k) : 0;
        return denom > 0 ? v / denom : 0;
      });
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
      const kList = getKList();
      const useNormalize = jsNormalize.checked;
      const traces = [
        {type: "heatmap", x: stateLabels, y: yLabels, z: md, xaxis: "x1", yaxis: "y1", zmin: 0, zmax: 1, colorscale: "Viridis", colorbar: {title: "p(state)", len: 0.35, y: 0.82, thickness: 12} , name: "MD"},
        {type: "heatmap", x: stateLabels, y: yLabels, z: sample, xaxis: "x2", yaxis: "y2", zmin: 0, zmax: 1, colorscale: "Viridis", showscale: false, name: active.label},
        {type: "heatmap", x: stateLabels, y: yLabels, z: errSample, xaxis: "x3", yaxis: "y3", zmin: 0, zmax: payload.vmax_err, colorscale: "Magma", colorbar: {title: "|Sample-MD|", len: 0.25, y: 0.45, thickness: 12} , name: "Error"},
      ];

      payload.js_series.forEach((series, idx) => {
        const jsVals = sliceVec(series.values, idxs);
        const jsNorm = normalizeByLogK(jsVals, idxs, kList);
        const yVals = useNormalize ? jsNorm : jsVals;
        const customData = jsVals.map((v, i) => [v, jsNorm[i], kList[idxs[i]]]);
        traces.push({
          type: "bar",
          x: yLabels,
          y: yVals,
          xaxis: "x4",
          yaxis: "y4",
          name: series.label,
          marker: {color: series.color || undefined},
          customdata: customData,
          hovertemplate: "Residue %{x}<br>JS: %{customdata[0]:.3f}<br>JS/logK: %{customdata[1]:.3f}<br>K=%{customdata[2]}<extra></extra>",
        });
      });
      return traces;
    }

    function setStatus(idxs) {
      const mode = jsNormalize.checked ? "normalized JS" : "raw JS";
      statusEl.textContent = `Showing ${idxs.length} / ${allIdx.length} residues | ${mode}`;
    }

    function applySelection(idxs) {
      const useIdxs = idxs.length ? idxs : allIdx;
      const traces = buildTraces(useIdxs);
      const layout = {
        ...baseLayout,
        yaxis4: {
          ...baseLayout.yaxis4,
          title: jsNormalize.checked ? "JS / ln(K)" : "JS divergence",
        },
      };
      Plotly.react(figId, traces, layout, {responsive: true, displaylogo: false});
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
    jsNormalize.addEventListener("change", () => {
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
        if (!jsNormalize.checked) {
          scores = payload.js_max;
        } else {
          const kList = getKList();
          scores = payload.js_series.reduce((acc, series) => {
            series.values.forEach((v, i) => {
              const denom = kList[i] > 1 ? Math.log(kList[i]) : 0;
              const score = denom > 0 ? v / denom : 0;
              if (score > acc[i]) acc[i] = score;
            });
            return acc;
          }, new Array(kList.length).fill(0));
        }
      }
      const ranked = allIdx.slice().sort((a, b) => scores[b] - scores[a]);
      const pick = ranked.slice(0, Math.min(n, ranked.length));
      setSelected(pick);
    });

    // initial render
    applySelection(allIdx);
  </script>
</body>
</html>
"""
    return template.replace("__DIV_ID__", div_id).replace("__PAYLOAD__", payload).replace("__LAYOUT__", fig_layout)


def _html_template_multi(fig_layout: str, payload: str, div_id: str = "marginal-fig") -> str:
    template = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Marginal comparison</title>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
    .wrap { display: flex; flex-direction: row; height: 100vh; }
    .controls { width: 420px; padding: 14px; border-right: 1px solid #ccc; box-sizing: border-box; overflow-y: auto; }
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
        <label for="md-select">MD source:</label>
        <select id="md-select"></select>
      </div>
      <div class="control-row">
        <label for="sample-select">Sampler:</label>
        <select id="sample-select"></select>
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
      <div class="control-row">
        <label><input id="js-normalize" type="checkbox" /> Normalize JS by log(K)</label>
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
    const mdSelect = document.getElementById("md-select");
    const sampleSelect = document.getElementById("sample-select");
    const jsNormalize = document.getElementById("js-normalize");

    const palette = [
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
    ];

    payload.md_sources.forEach((src, idx) => {
      const opt = document.createElement("option");
      opt.value = src.id;
      opt.textContent = src.label;
      if (idx === 0) opt.selected = true;
      mdSelect.appendChild(opt);
    });

    payload.sample_sources.forEach((src, idx) => {
      const opt = document.createElement("option");
      opt.value = src.id;
      opt.textContent = src.label;
      if (idx === 0) opt.selected = true;
      sampleSelect.appendChild(opt);
    });

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
    function getKList() {
      return getActiveMd().src.matrix.map(row => row.reduce((acc, v) => acc + (Number.isFinite(v) ? 1 : 0), 0));
    }
    function normalizeByLogK(values, idxs, kList) {
      return values.map((v, i) => {
        const k = kList[idxs[i]];
        const denom = k > 1 ? Math.log(k) : 0;
        return denom > 0 ? v / denom : 0;
      });
    }

    function getActiveMd() {
      const id = mdSelect.value || (payload.md_sources[0] && payload.md_sources[0].id);
      const idx = payload.md_sources.findIndex(src => src.id === id);
      return { idx: idx < 0 ? 0 : idx, src: payload.md_sources[idx < 0 ? 0 : idx] };
    }

    function getActiveSample() {
      const id = sampleSelect.value || (payload.sample_sources[0] && payload.sample_sources[0].id);
      const idx = payload.sample_sources.findIndex(src => src.id === id);
      return { idx: idx < 0 ? 0 : idx, src: payload.sample_sources[idx < 0 ? 0 : idx] };
    }

    function rowMax(row) {
      let max = -Infinity;
      for (const v of row) {
        if (Number.isFinite(v) && v > max) max = v;
      }
      return max === -Infinity ? 0 : max;
    }

    function buildTraces(idxs) {
      const yLabels = idxs.map(i => payload.labels[i]);
      const md = sliceRows(getActiveMd().src.matrix, idxs);
      const sample = sliceRows(getActiveSample().src.matrix, idxs);
      const err = sample.map((row, rIdx) => row.map((v, cIdx) => Math.abs(v - md[rIdx][cIdx])));
      const kList = getKList();
      const useNormalize = jsNormalize.checked;

      const traces = [
        {type: "heatmap", x: stateLabels, y: yLabels, z: md, xaxis: "x1", yaxis: "y1", zmin: 0, zmax: 1, colorscale: "Viridis", colorbar: {title: "p(state)", len: 0.35, y: 0.82, thickness: 12} , name: "MD"},
        {type: "heatmap", x: stateLabels, y: yLabels, z: sample, xaxis: "x2", yaxis: "y2", zmin: 0, zmax: 1, colorscale: "Viridis", showscale: false, name: getActiveSample().src.label},
        {type: "heatmap", x: stateLabels, y: yLabels, z: err, xaxis: "x3", yaxis: "y3", zmin: 0, zmax: payload.vmax_err, colorscale: "Magma", colorbar: {title: "|Sample-MD|", len: 0.25, y: 0.45, thickness: 12} , name: "Error"},
      ];

      payload.sample_sources.forEach((src, idx) => {
        const jsVals = sliceVec(payload.js_md_sample[getActiveMd().idx][idx], idxs);
        const jsNorm = normalizeByLogK(jsVals, idxs, kList);
        const yVals = useNormalize ? jsNorm : jsVals;
        const customData = jsVals.map((v, i) => [v, jsNorm[i], kList[idxs[i]]]);
        traces.push({
          type: "bar",
          x: yLabels,
          y: yVals,
          xaxis: "x4",
          yaxis: "y4",
          name: src.label,
          marker: {color: palette[idx % palette.length]},
          customdata: customData,
          hovertemplate: "Residue %{x}<br>JS: %{customdata[0]:.3f}<br>JS/logK: %{customdata[1]:.3f}<br>K=%{customdata[2]}<extra></extra>",
        });
      });
      return { traces, err };
    }

    function setStatus(idxs) {
      const mode = jsNormalize.checked ? "normalized JS" : "raw JS";
      statusEl.textContent = `Showing ${idxs.length} / ${allIdx.length} residues | ${mode}`;
    }

    function applySelection(idxs) {
      const useIdxs = idxs.length ? idxs : allIdx;
      const result = buildTraces(useIdxs);
      const layout = {
        ...baseLayout,
        yaxis4: {
          ...baseLayout.yaxis4,
          title: jsNormalize.checked ? "JS / ln(K)" : "JS divergence",
        },
      };
      Plotly.react(figId, result.traces, layout, {responsive: true, displaylogo: false});
      setStatus(useIdxs);
      return result;
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
    mdSelect.addEventListener("change", () => {
      const idxs = checkboxes.filter(cb => cb.checked).map(cb => parseInt(cb.value, 10));
      applySelection(idxs);
    });
    sampleSelect.addEventListener("change", () => {
      const idxs = checkboxes.filter(cb => cb.checked).map(cb => parseInt(cb.value, 10));
      applySelection(idxs);
    });
    jsNormalize.addEventListener("change", () => {
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
        const useIdxs = checkboxes.filter(cb => cb.checked).map(cb => parseInt(cb.value, 10));
        const result = buildTraces(useIdxs.length ? useIdxs : allIdx);
        scores = result.err.map(rowMax);
      } else {
        if (!jsNormalize.checked) {
          scores = payload.js_md_sample[getActiveMd().idx][getActiveSample().idx];
        } else {
          const raw = payload.js_md_sample[getActiveMd().idx][getActiveSample().idx];
          const kList = getKList();
          scores = raw.map((v, i) => {
            const denom = kList[i] > 1 ? Math.log(kList[i]) : 0;
            return denom > 0 ? v / denom : 0;
          });
        }
      }
      const ranked = allIdx.slice().sort((a, b) => scores[b] - scores[a]);
      const pick = ranked.slice(0, Math.min(n, ranked.length));
      setSelected(pick);
    });

    applySelection(allIdx);
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


def plot_marginal_dashboard(
    *,
    md_sources: Sequence[dict],
    sample_sources: Sequence[dict],
    js_md_sample: np.ndarray,
    residue_labels: Iterable[object],
    out_path: str | Path,
) -> Path:
    if not md_sources or not sample_sources:
        raise ValueError("Need at least one MD source and one sampler source.")

    md_mat = np.asarray(md_sources[0]["matrix"], dtype=float)
    res_labels = _coerce_labels(residue_labels, md_mat.shape[0])

    vmax_err = 0.0
    for md in md_sources:
        md_arr = np.asarray(md["matrix"], dtype=float)
        for sample in sample_sources:
            sample_arr = np.asarray(sample["matrix"], dtype=float)
            if md_arr.shape != sample_arr.shape:
                continue
            vmax_err = max(vmax_err, float(np.nanmax(np.abs(sample_arr - md_arr))))

    payload = {
        "labels": res_labels,
        "stateCount": int(md_mat.shape[1]),
        "md_sources": [
            {"id": src["id"], "label": src["label"], "matrix": np.asarray(src["matrix"], dtype=float).tolist()}
            for src in md_sources
        ],
        "sample_sources": [
            {"id": src["id"], "label": src["label"], "matrix": np.asarray(src["matrix"], dtype=float).tolist()}
            for src in sample_sources
        ],
        "js_md_sample": np.asarray(js_md_sample, dtype=float).tolist(),
        "vmax_err": float(vmax_err) if vmax_err > 0 else 1.0,
    }

    layout = {
        "title": {"text": "Marginal comparison: MD vs sampled models", "x": 0.5},
        "height": 920,
        "margin": {"l": 70, "r": 20, "t": 60, "b": 60},
        "barmode": "group",
        "xaxis": {"domain": [0.0, 0.48], "anchor": "y", "title": "state"},
        "yaxis": {"domain": [0.67, 1.0], "title": "residue", "automargin": True},
        "xaxis2": {"domain": [0.52, 1.0], "anchor": "y2", "title": "state"},
        "yaxis2": {"domain": [0.67, 1.0], "showticklabels": False},
        "xaxis3": {"domain": [0.0, 1.0], "anchor": "y3", "title": "state"},
        "yaxis3": {"domain": [0.34, 0.64], "title": "residue", "automargin": True},
        "xaxis4": {"domain": [0.0, 1.0], "anchor": "y4", "title": "residue"},
        "yaxis4": {"domain": [0.0, 0.28], "title": "JS divergence", "automargin": True},
        "legend": {"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
    }

    html = _html_template_multi(fig_layout=json.dumps(layout), payload=json.dumps(payload))

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

        residue_labels = data["residue_labels"]
        md_source_ids = data["md_source_ids"] if "md_source_ids" in data else None
        md_source_labels = data["md_source_labels"] if "md_source_labels" in data else None
        p_md_by_source = data["p_md_by_source"] if "p_md_by_source" in data else None
        sample_source_ids = data["sample_source_ids"] if "sample_source_ids" in data else None
        sample_source_labels = data["sample_source_labels"] if "sample_source_labels" in data else None
        p_sample_by_source = data["p_sample_by_source"] if "p_sample_by_source" in data else None
        js_md_sample = data["js_md_sample"] if "js_md_sample" in data else None

        use_multi = (
            isinstance(md_source_ids, np.ndarray)
            and md_source_ids.size > 0
            and isinstance(sample_source_ids, np.ndarray)
            and sample_source_ids.size > 0
            and isinstance(p_md_by_source, np.ndarray)
            and p_md_by_source.size > 0
            and isinstance(p_sample_by_source, np.ndarray)
            and p_sample_by_source.size > 0
            and isinstance(js_md_sample, np.ndarray)
            and js_md_sample.size > 0
        )

        if use_multi:
            md_sources = []
            for idx, src_id in enumerate(md_source_ids):
                label = md_source_labels[idx] if md_source_labels is not None and idx < len(md_source_labels) else src_id
                md_sources.append(
                    {
                        "id": str(src_id),
                        "label": str(label),
                        "matrix": np.asarray(p_md_by_source[idx], dtype=float),
                    }
                )

            sample_sources = []
            for idx, src_id in enumerate(sample_source_ids):
                label = (
                    sample_source_labels[idx]
                    if sample_source_labels is not None and idx < len(sample_source_labels)
                    else src_id
                )
                sample_sources.append(
                    {
                        "id": str(src_id),
                        "label": str(label),
                        "matrix": np.asarray(p_sample_by_source[idx], dtype=float),
                    }
                )

            return plot_marginal_dashboard(
                md_sources=md_sources,
                sample_sources=sample_sources,
                js_md_sample=js_md_sample,
                residue_labels=residue_labels,
                out_path=out_path,
            )

        p_md = data["p_md"]
        p_gibbs = data["p_gibbs"]
        p_sa = data["p_sa"]
        js_gibbs = data["js_gibbs"]
        js_sa = data["js_sa"]
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


def plot_sampling_report_from_npz(
    *,
    summary_path: str | Path,
    out_path: str | Path,
) -> Path:
    with np.load(summary_path, allow_pickle=False) as data:
        residue_labels = data["residue_labels"] if "residue_labels" in data else np.array([])
        K = data["K"] if "K" in data else np.array([], dtype=int)
        edges = data["edges"] if "edges" in data else np.zeros((0, 2), dtype=int)

        md_source_ids = data["md_source_ids"] if "md_source_ids" in data else np.array([], dtype=str)
        md_source_labels = data["md_source_labels"] if "md_source_labels" in data else np.array([], dtype=str)
        md_source_types = data["md_source_types"] if "md_source_types" in data else np.array([], dtype=str)
        md_source_counts = data["md_source_counts"] if "md_source_counts" in data else np.array([], dtype=int)
        p_md_by_source = data["p_md_by_source"] if "p_md_by_source" in data else np.zeros((0, 0, 0), dtype=float)

        sample_source_ids = data["sample_source_ids"] if "sample_source_ids" in data else np.array([], dtype=str)
        sample_source_labels = data["sample_source_labels"] if "sample_source_labels" in data else np.array([], dtype=str)
        sample_source_types = data["sample_source_types"] if "sample_source_types" in data else np.array([], dtype=str)
        sample_source_counts = data["sample_source_counts"] if "sample_source_counts" in data else np.array([], dtype=int)
        p_sample_by_source = data["p_sample_by_source"] if "p_sample_by_source" in data else np.zeros((0, 0, 0), dtype=float)

        js_md_sample = data["js_md_sample"] if "js_md_sample" in data else np.zeros((0, 0, 0), dtype=float)
        js2_md_sample = data["js2_md_sample"] if "js2_md_sample" in data else np.zeros((0, 0, 0), dtype=float)
        js_gibbs_sample = data["js_gibbs_sample"] if "js_gibbs_sample" in data else np.zeros((0, 0), dtype=float)
        js2_gibbs_sample = data["js2_gibbs_sample"] if "js2_gibbs_sample" in data else np.zeros((0, 0), dtype=float)

        energy_bins = data["energy_bins"] if "energy_bins" in data else np.array([], dtype=float)
        energy_hist_md = data["energy_hist_md"] if "energy_hist_md" in data else np.zeros((0, 0), dtype=float)
        energy_cdf_md = data["energy_cdf_md"] if "energy_cdf_md" in data else np.zeros((0, 0), dtype=float)
        energy_hist_sample = data["energy_hist_sample"] if "energy_hist_sample" in data else np.zeros((0, 0), dtype=float)
        energy_cdf_sample = data["energy_cdf_sample"] if "energy_cdf_sample" in data else np.zeros((0, 0), dtype=float)

        nn_bins = data["nn_bins"] if "nn_bins" in data else np.array([], dtype=int)
        nn_cdf_sample_to_md = data["nn_cdf_sample_to_md"] if "nn_cdf_sample_to_md" in data else np.zeros((0, 0, 0), dtype=float)
        nn_cdf_md_to_sample = data["nn_cdf_md_to_sample"] if "nn_cdf_md_to_sample" in data else np.zeros((0, 0, 0), dtype=float)

        edge_strength = data["edge_strength"] if "edge_strength" in data else np.array([], dtype=float)
        xlik_delta_active = data["xlik_delta_active"] if "xlik_delta_active" in data else np.array([], dtype=float)
        xlik_delta_inactive = data["xlik_delta_inactive"] if "xlik_delta_inactive" in data else np.array([], dtype=float)
        xlik_auc = data["xlik_auc"] if "xlik_auc" in data else np.array([], dtype=float)
        xlik_active_state_ids = data["xlik_active_state_ids"] if "xlik_active_state_ids" in data else np.array([], dtype=str)
        xlik_inactive_state_ids = data["xlik_inactive_state_ids"] if "xlik_inactive_state_ids" in data else np.array([], dtype=str)
        xlik_active_state_labels = data["xlik_active_state_labels"] if "xlik_active_state_labels" in data else np.array([], dtype=str)
        xlik_inactive_state_labels = data["xlik_inactive_state_labels"] if "xlik_inactive_state_labels" in data else np.array([], dtype=str)
        xlik_delta_fit_by_other = data["xlik_delta_fit_by_other"] if "xlik_delta_fit_by_other" in data else np.array([], dtype=float)
        xlik_delta_other_by_other = data["xlik_delta_other_by_other"] if "xlik_delta_other_by_other" in data else np.array([], dtype=float)
        xlik_delta_other_counts = data["xlik_delta_other_counts"] if "xlik_delta_other_counts" in data else np.array([], dtype=int)
        xlik_other_state_ids = data["xlik_other_state_ids"] if "xlik_other_state_ids" in data else np.array([], dtype=str)
        xlik_other_state_labels = data["xlik_other_state_labels"] if "xlik_other_state_labels" in data else np.array([], dtype=str)
        xlik_auc_by_other = data["xlik_auc_by_other"] if "xlik_auc_by_other" in data else np.array([], dtype=float)
        xlik_roc_fpr_by_other = data["xlik_roc_fpr_by_other"] if "xlik_roc_fpr_by_other" in data else np.array([], dtype=float)
        xlik_roc_tpr_by_other = data["xlik_roc_tpr_by_other"] if "xlik_roc_tpr_by_other" in data else np.array([], dtype=float)
        xlik_roc_counts = data["xlik_roc_counts"] if "xlik_roc_counts" in data else np.array([], dtype=int)
        xlik_score_fit_by_other = data["xlik_score_fit_by_other"] if "xlik_score_fit_by_other" in data else np.array([], dtype=float)
        xlik_score_other_by_other = data["xlik_score_other_by_other"] if "xlik_score_other_by_other" in data else np.array([], dtype=float)
        xlik_score_other_counts = data["xlik_score_other_counts"] if "xlik_score_other_counts" in data else np.array([], dtype=int)
        xlik_auc_score_by_other = data["xlik_auc_score_by_other"] if "xlik_auc_score_by_other" in data else np.array([], dtype=float)
        xlik_score_roc_fpr_by_other = data["xlik_score_roc_fpr_by_other"] if "xlik_score_roc_fpr_by_other" in data else np.array([], dtype=float)
        xlik_score_roc_tpr_by_other = data["xlik_score_roc_tpr_by_other"] if "xlik_score_roc_tpr_by_other" in data else np.array([], dtype=float)
        xlik_score_roc_counts = data["xlik_score_roc_counts"] if "xlik_score_roc_counts" in data else np.array([], dtype=int)

    md_sources = []
    for idx, src_id in enumerate(md_source_ids):
        label = md_source_labels[idx] if idx < len(md_source_labels) else src_id
        src_type = md_source_types[idx] if idx < len(md_source_types) else ""
        count = int(md_source_counts[idx]) if idx < len(md_source_counts) else 0
        md_sources.append(
            {
                "id": str(src_id),
                "label": str(label),
                "type": str(src_type),
                "count": count,
            }
        )

    sample_sources = []
    for idx, src_id in enumerate(sample_source_ids):
        label = sample_source_labels[idx] if idx < len(sample_source_labels) else src_id
        src_type = sample_source_types[idx] if idx < len(sample_source_types) else ""
        count = int(sample_source_counts[idx]) if idx < len(sample_source_counts) else 0
        sample_sources.append(
            {
                "id": str(src_id),
                "label": str(label),
                "type": str(src_type),
                "count": count,
            }
        )

    gibbs_index = 0
    for idx, src in enumerate(sample_sources):
        if src.get("id") == "gibbs":
            gibbs_index = idx
            break

    payload = {
        "residue_labels": [str(v) for v in residue_labels.tolist()] if hasattr(residue_labels, "tolist") else [],
        "K": K.tolist() if hasattr(K, "tolist") else [],
        "edges": edges.tolist() if hasattr(edges, "tolist") else [],
        "md_sources": md_sources,
        "sample_sources": sample_sources,
        "p_md_by_source": p_md_by_source.tolist() if p_md_by_source.size else [],
        "p_sample_by_source": p_sample_by_source.tolist() if p_sample_by_source.size else [],
        "js_md_sample": js_md_sample.tolist() if js_md_sample.size else [],
        "js2_md_sample": js2_md_sample.tolist() if js2_md_sample.size else [],
        "js_gibbs_sample": js_gibbs_sample.tolist() if js_gibbs_sample.size else [],
        "js2_gibbs_sample": js2_gibbs_sample.tolist() if js2_gibbs_sample.size else [],
        "energy_bins": energy_bins.tolist() if energy_bins.size else [],
        "energy_hist_md": energy_hist_md.tolist() if energy_hist_md.size else [],
        "energy_cdf_md": energy_cdf_md.tolist() if energy_cdf_md.size else [],
        "energy_hist_sample": energy_hist_sample.tolist() if energy_hist_sample.size else [],
        "energy_cdf_sample": energy_cdf_sample.tolist() if energy_cdf_sample.size else [],
        "nn_bins": nn_bins.tolist() if nn_bins.size else [],
        "nn_cdf_sample_to_md": nn_cdf_sample_to_md.tolist() if nn_cdf_sample_to_md.size else [],
        "nn_cdf_md_to_sample": nn_cdf_md_to_sample.tolist() if nn_cdf_md_to_sample.size else [],
        "edge_strength": edge_strength.tolist() if edge_strength.size else [],
        "xlik_delta_active": xlik_delta_active.tolist() if xlik_delta_active.size else [],
        "xlik_delta_inactive": xlik_delta_inactive.tolist() if xlik_delta_inactive.size else [],
        "xlik_auc": float(xlik_auc[0]) if xlik_auc.size else None,
        "xlik_active_state_ids": [str(v) for v in xlik_active_state_ids.tolist()] if xlik_active_state_ids.size else [],
        "xlik_inactive_state_ids": [str(v) for v in xlik_inactive_state_ids.tolist()] if xlik_inactive_state_ids.size else [],
        "xlik_active_state_labels": [str(v) for v in xlik_active_state_labels.tolist()] if xlik_active_state_labels.size else [],
        "xlik_inactive_state_labels": [str(v) for v in xlik_inactive_state_labels.tolist()] if xlik_inactive_state_labels.size else [],
        "xlik_delta_fit_by_other": xlik_delta_fit_by_other.tolist() if xlik_delta_fit_by_other.size else [],
        "xlik_delta_other_by_other": xlik_delta_other_by_other.tolist() if xlik_delta_other_by_other.size else [],
        "xlik_delta_other_counts": xlik_delta_other_counts.tolist() if xlik_delta_other_counts.size else [],
        "xlik_other_state_ids": [str(v) for v in xlik_other_state_ids.tolist()] if xlik_other_state_ids.size else [],
        "xlik_other_state_labels": [str(v) for v in xlik_other_state_labels.tolist()] if xlik_other_state_labels.size else [],
        "xlik_auc_by_other": xlik_auc_by_other.tolist() if xlik_auc_by_other.size else [],
        "xlik_roc_fpr_by_other": xlik_roc_fpr_by_other.tolist() if xlik_roc_fpr_by_other.size else [],
        "xlik_roc_tpr_by_other": xlik_roc_tpr_by_other.tolist() if xlik_roc_tpr_by_other.size else [],
        "xlik_roc_counts": xlik_roc_counts.tolist() if xlik_roc_counts.size else [],
        "xlik_score_fit_by_other": xlik_score_fit_by_other.tolist() if xlik_score_fit_by_other.size else [],
        "xlik_score_other_by_other": xlik_score_other_by_other.tolist() if xlik_score_other_by_other.size else [],
        "xlik_score_other_counts": xlik_score_other_counts.tolist() if xlik_score_other_counts.size else [],
        "xlik_auc_score_by_other": xlik_auc_score_by_other.tolist() if xlik_auc_score_by_other.size else [],
        "xlik_score_roc_fpr_by_other": xlik_score_roc_fpr_by_other.tolist() if xlik_score_roc_fpr_by_other.size else [],
        "xlik_score_roc_tpr_by_other": xlik_score_roc_tpr_by_other.tolist() if xlik_score_roc_tpr_by_other.size else [],
        "xlik_score_roc_counts": xlik_score_roc_counts.tolist() if xlik_score_roc_counts.size else [],
        "gibbs_index": gibbs_index,
    }

    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Potts sampling report</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { padding: 18px 20px 32px; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    h2 { margin: 18px 0 8px; font-size: 18px; }
    h3 { margin: 12px 0 6px; font-size: 15px; }
    .controls { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0 18px; align-items: center; }
    .control { display: flex; flex-direction: column; gap: 4px; min-width: 220px; }
    label { font-size: 12px; color: #94a3b8; }
    select { background: #1e293b; border: 1px solid #334155; color: #e2e8f0; padding: 6px 8px; border-radius: 6px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 12px; }
    .plot { height: 360px; }
    .plot.tall { height: 420px; }
    .plot.short { height: 260px; }
    .tables { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #1f2937; }
    th { color: #93c5fd; font-weight: 600; }
    .meta { font-size: 12px; color: #94a3b8; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Potts sampling report</h1>
    <p class="meta">Compare Potts sampling outputs against selected MD states or metastable subsets.</p>
    <div class="controls">
      <div class="control">
        <label for="md-select">MD source</label>
        <select id="md-select"></select>
      </div>
      <div class="control">
        <label for="sample-select">Sampler source</label>
        <select id="sample-select"></select>
      </div>
      <div class="meta" id="selection-meta"></div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Residue barcode</h2>
        <div id="residue-barcode" class="plot tall"></div>
      </div>
      <div class="card">
        <h2>Edge barcode</h2>
        <div id="edge-barcode" class="plot tall"></div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Edge heatmap</h2>
        <div class="meta">
          <label><input id="edge-js-normalize" type="checkbox" /> Normalize edge JS2 by ln(Kx*Ky)</label>
        </div>
        <div id="edge-heatmap" class="plot tall"></div>
      </div>
    </div>

    <div class="tables">
      <div class="card">
        <h3>Top residues</h3>
        <div id="top-residues"></div>
      </div>
      <div class="card">
        <h3>Top edges</h3>
        <div id="top-edges"></div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Edge mismatch network</h2>
        <div id="edge-network" class="plot"></div>
      </div>
      <div class="card">
        <h2>Edge mismatch vs strength</h2>
        <div id="edge-strength" class="plot"></div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Energy histogram</h2>
        <div id="energy-hist" class="plot short"></div>
        <div id="energy-cdf" class="plot short"></div>
      </div>
      <div class="card">
        <h2>Nearest-neighbor CDFs</h2>
        <div id="nn-cdf" class="plot short"></div>
      </div>
    </div>
  </div>

  <script>
    const payload = __PAYLOAD__;
    const mdSelect = document.getElementById("md-select");
    const sampleSelect = document.getElementById("sample-select");
    const selectionMeta = document.getElementById("selection-meta");
    const xlikMeta = document.getElementById("xlik-meta");
    const edgeNormalize = document.getElementById("edge-js-normalize");

    function populateSelect(select, items) {
      select.innerHTML = "";
      items.forEach((item, idx) => {
        const opt = document.createElement("option");
        opt.value = item.id;
        opt.textContent = item.label;
        if (idx === 0) opt.selected = true;
        select.appendChild(opt);
      });
    }

    populateSelect(mdSelect, payload.md_sources || []);
    populateSelect(sampleSelect, payload.sample_sources || []);

    function getIndexById(items, id) {
      return items.findIndex((item) => item.id === id);
    }

    function getMdIndex() {
      const id = mdSelect.value;
      const idx = getIndexById(payload.md_sources, id);
      return idx < 0 ? 0 : idx;
    }

    function getSampleIndex() {
      const id = sampleSelect.value;
      const idx = getIndexById(payload.sample_sources, id);
      return idx < 0 ? 0 : idx;
    }

    function edgeStatsFor(edge) {
      const kList = payload.K || [];
      const r = edge[0];
      const s = edge[1];
      const kr = Number.isFinite(kList[r]) ? kList[r] : null;
      const ks = Number.isFinite(kList[s]) ? kList[s] : null;
      const kprod = kr !== null && ks !== null ? kr * ks : null;
      const denom = kprod && kprod > 1 ? Math.log(kprod) : 0;
      return { kr, ks, kprod, denom };
    }

    function normalizeEdgeValues(values, edges) {
      return values.map((v, idx) => {
        if (!Number.isFinite(v)) return v;
        const edge = edges[idx] || [];
        if (edge.length < 2) return v;
        const stats = edgeStatsFor(edge);
        if (stats.denom > 0) return v / stats.denom;
        if (stats.kprod === null) return v;
        return 0;
      });
    }

    function edgeNormalizedValue(raw, stats) {
      if (!Number.isFinite(raw)) return raw;
      if (stats.denom > 0) return raw / stats.denom;
      if (stats.kprod === null) return raw;
      return 0;
    }

    function setMeta() {
      const md = payload.md_sources[getMdIndex()];
      const sample = payload.sample_sources[getSampleIndex()];
      selectionMeta.textContent = `MD frames: ${md ? md.count : 0} · Samples: ${sample ? sample.count : 0}`;
    }

    function sortIndices(values) {
      return values.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map((pair) => pair[1]);
    }

    function buildResidueBarcode() {
      if (!payload.js_md_sample || !payload.js_md_sample.length) {
        document.getElementById("residue-barcode").innerHTML = "<div class='meta'>Residue metrics unavailable.</div>";
        return;
      }
      const mdIdx = getMdIndex();
      const sampleIdx = getSampleIndex();
      const gibbsIdx = payload.gibbs_index || 0;
      const jsMdGibbs = (payload.js_md_sample[mdIdx] || [])[gibbsIdx] || [];
      const jsMdSample = (payload.js_md_sample[mdIdx] || [])[sampleIdx] || [];
      const jsGibbsSample = (payload.js_gibbs_sample || [])[sampleIdx] || [];
      const order = sortIndices(jsMdSample);
      const labels = order.map((i) => payload.residue_labels[i] || i.toString());
      const z = [
        order.map((i) => jsMdGibbs[i] || 0),
        order.map((i) => jsMdSample[i] || 0),
        order.map((i) => jsGibbsSample[i] || 0),
      ];
      const y = ["JS(MD,Gibbs)", "JS(MD,Sample)", "JS(Gibbs,Sample)"];
      Plotly.react(
        "residue-barcode",
        [
          {
            type: "heatmap",
            x: labels,
            y: y,
            z: z,
            colorscale: "Viridis",
            hovertemplate: "Residue %{x}<br>%{y}: %{z:.3f}<extra></extra>",
          },
        ],
        {
          margin: { l: 120, r: 10, t: 10, b: 40 },
          xaxis: { showticklabels: false },
          yaxis: { automargin: true },
        },
        { responsive: true }
      );
    }

    function buildEdgeBarcode() {
      if (!payload.js2_md_sample || !payload.js2_md_sample.length) {
        document.getElementById("edge-barcode").innerHTML = "<div class='meta'>Edge metrics unavailable.</div>";
        return;
      }
      const mdIdx = getMdIndex();
      const sampleIdx = getSampleIndex();
      const gibbsIdx = payload.gibbs_index || 0;
      const edges = payload.edges || [];
      const useNormalize = edgeNormalize && edgeNormalize.checked;
      const jsMdSampleRaw = (payload.js2_md_sample[mdIdx] || [])[sampleIdx] || [];
      const jsMdGibbsRaw = (payload.js2_md_sample[mdIdx] || [])[gibbsIdx] || [];
      const jsGibbsSampleRaw = (payload.js2_gibbs_sample || [])[sampleIdx] || [];
      const jsMdSample = useNormalize ? normalizeEdgeValues(jsMdSampleRaw, edges) : jsMdSampleRaw;
      const jsMdGibbs = useNormalize ? normalizeEdgeValues(jsMdGibbsRaw, edges) : jsMdGibbsRaw;
      const jsGibbsSample = useNormalize ? normalizeEdgeValues(jsGibbsSampleRaw, edges) : jsGibbsSampleRaw;
      const order = sortIndices(jsMdSample);
      const edgeLabels = order.map((i) => {
        const edge = edges[i] || [];
        const r = edge[0];
        const s = edge[1];
        const rLabel = payload.residue_labels[r] || r;
        const sLabel = payload.residue_labels[s] || s;
        return `${rLabel}-${sLabel}`;
      });
      const edgeKprod = order.map((i) => {
        const edge = edges[i] || [];
        const stats = edge.length >= 2 ? edgeStatsFor(edge) : { kprod: null };
        return stats.kprod;
      });
      const z = [
        order.map((i) => jsMdGibbs[i] || 0),
        order.map((i) => jsMdSample[i] || 0),
        order.map((i) => jsGibbsSample[i] || 0),
      ];
      const customData = [
        order.map((i, idx) => [jsMdGibbsRaw[i] || 0, jsMdGibbs[i] || 0, edgeKprod[idx]]),
        order.map((i, idx) => [jsMdSampleRaw[i] || 0, jsMdSample[i] || 0, edgeKprod[idx]]),
        order.map((i, idx) => [jsGibbsSampleRaw[i] || 0, jsGibbsSample[i] || 0, edgeKprod[idx]]),
      ];
      const suffix = useNormalize ? "/ln(Kx*Ky)" : "";
      const y = [`JS2${suffix}(MD,Gibbs)`, `JS2${suffix}(MD,Sample)`, `JS2${suffix}(Gibbs,Sample)`];
      Plotly.react(
        "edge-barcode",
        [
          {
            type: "heatmap",
            x: edgeLabels,
            y: y,
            z: z,
            colorscale: "Viridis",
            customdata: customData,
            hovertemplate:
              "Edge %{x}<br>%{y}: %{z:.3f}" +
              "<br>JS2(raw): %{customdata[0]:.3f}" +
              "<br>JS2/ln(Kx*Ky): %{customdata[1]:.3f}" +
              "<br>Kx*Ky=%{customdata[2]}" +
              "<extra></extra>",
          },
        ],
        {
          margin: { l: 140, r: 10, t: 10, b: 40 },
          xaxis: { showticklabels: false },
          yaxis: { automargin: true },
        },
        { responsive: true }
      );
    }

    function buildEdgeHeatmap() {
      const container = document.getElementById("edge-heatmap");
      if (!payload.js2_md_sample || !payload.js2_md_sample.length) {
        container.innerHTML = "<div class='meta'>Edge heatmap unavailable.</div>";
        return;
      }
      const mdIdx = getMdIndex();
      const sampleIdx = getSampleIndex();
      const js2Raw = (payload.js2_md_sample[mdIdx] || [])[sampleIdx] || [];
      const edges = payload.edges || [];
      const n = (payload.residue_labels || []).length;
      if (!js2Raw.length || !edges.length || !n) {
        container.innerHTML = "<div class='meta'>Edge heatmap unavailable.</div>";
        return;
      }

      const useNormalize = edgeNormalize && edgeNormalize.checked;
      const order = Array.from({ length: n }, (_, i) => i);
      const labels = order.map((i) => payload.residue_labels[i] || i.toString());
      const z = Array.from({ length: n }, () => Array.from({ length: n }, () => null));
      const customdata = Array.from({ length: n }, () => Array.from({ length: n }, () => null));
      let maxVal = 0;
      edges.forEach((edge, idx) => {
        const r = edge[0];
        const s = edge[1];
        if (r === undefined || s === undefined) return;
        const raw = js2Raw[idx] || 0;
        const stats = edgeStatsFor(edge);
        const norm = edgeNormalizedValue(raw, stats);
        const val = useNormalize ? norm : raw;
        z[r][s] = val;
        z[s][r] = val;
        customdata[r][s] = [raw, norm, stats.kr, stats.ks, stats.kprod];
        customdata[s][r] = [raw, norm, stats.kr, stats.ks, stats.kprod];
        if (val > maxVal) maxVal = val;
      });

      Plotly.react(
        "edge-heatmap",
        [
          {
            type: "heatmap",
            x: labels,
            y: labels,
            z: z,
            colorscale: "Viridis",
            zmin: 0,
            zmax: maxVal || 1,
            customdata: customdata,
            hoverongaps: false,
            hovertemplate:
              "Residues %{x}-%{y}<br>JS2(MD,Sample): %{z:.3f}" +
              "<br>JS2(raw): %{customdata[0]:.3f}" +
              "<br>JS2/ln(Kx*Ky): %{customdata[1]:.3f}" +
              "<br>Kx=%{customdata[2]} Ky=%{customdata[3]} Kx*Ky=%{customdata[4]}" +
              "<extra></extra>",
          },
        ],
        {
          margin: { l: 60, r: 20, t: 10, b: 40 },
          xaxis: { showticklabels: false, constrain: "domain" },
          yaxis: { showticklabels: false, autorange: "reversed", scaleanchor: "x", scaleratio: 1, constrain: "domain" },
        },
        { responsive: true }
      );
    }

    function maxAbsDiff(a, b) {
      let max = 0;
      for (let i = 0; i < a.length; i++) {
        const av = a[i];
        const bv = b[i];
        if (!Number.isFinite(av) || !Number.isFinite(bv)) continue;
        const diff = Math.abs(av - bv);
        if (diff > max) max = diff;
      }
      return max;
    }

    function buildTopTables() {
      if (!payload.p_md_by_source || !payload.p_md_by_source.length) {
        document.getElementById("top-residues").innerHTML = "<div class='meta'>Top residues unavailable.</div>";
        document.getElementById("top-edges").innerHTML = "<div class='meta'>Top edges unavailable.</div>";
        return;
      }
      const mdIdx = getMdIndex();
      const sampleIdx = getSampleIndex();
      const gibbsIdx = payload.gibbs_index || 0;
      const jsMdSample = (payload.js_md_sample[mdIdx] || [])[sampleIdx] || [];
      const jsMdGibbs = (payload.js_md_sample[mdIdx] || [])[gibbsIdx] || [];
      const pMd = payload.p_md_by_source[mdIdx] || [];
      const pSample = payload.p_sample_by_source[sampleIdx] || [];
      const pGibbs = payload.p_sample_by_source[gibbsIdx] || [];

      const rankedResidues = jsMdSample.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 10);
      const residueRows = rankedResidues.map(([val, idx]) => {
        const label = payload.residue_labels[idx] || idx.toString();
        const maxSample = maxAbsDiff(pMd[idx] || [], pSample[idx] || []);
        const maxGibbs = maxAbsDiff(pMd[idx] || [], pGibbs[idx] || []);
        return `<tr><td>${label}</td><td>${val.toFixed(3)}</td><td>${(jsMdGibbs[idx] || 0).toFixed(3)}</td><td>${maxSample.toFixed(3)}</td><td>${maxGibbs.toFixed(3)}</td></tr>`;
      }).join("");
      document.getElementById("top-residues").innerHTML = `<table><thead><tr><th>Residue</th><th>JS(MD,Sample)</th><th>JS(MD,Gibbs)</th><th>max|Sample-MD|</th><th>max|Gibbs-MD|</th></tr></thead><tbody>${residueRows}</tbody></table>`;

      const edges = payload.edges || [];
      const useNormalize = edgeNormalize && edgeNormalize.checked;
      const js2MdSampleRaw = (payload.js2_md_sample[mdIdx] || [])[sampleIdx] || [];
      const js2MdGibbsRaw = (payload.js2_md_sample[mdIdx] || [])[gibbsIdx] || [];
      const js2MdSample = useNormalize ? normalizeEdgeValues(js2MdSampleRaw, edges) : js2MdSampleRaw;
      const js2MdGibbs = useNormalize ? normalizeEdgeValues(js2MdGibbsRaw, edges) : js2MdGibbsRaw;
      const rankedEdges = js2MdSample.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 10);
      const edgeRows = rankedEdges.map(([val, idx]) => {
        const edge = edges[idx] || [];
        const r = edge[0];
        const s = edge[1];
        const rLabel = payload.residue_labels[r] || r;
        const sLabel = payload.residue_labels[s] || s;
        return `<tr><td>${rLabel}-${sLabel}</td><td>${val.toFixed(3)}</td><td>${(js2MdGibbs[idx] || 0).toFixed(3)}</td></tr>`;
      }).join("");
      const edgeSuffix = useNormalize ? "/ln(Kx*Ky)" : "";
      document.getElementById("top-edges").innerHTML = `<table><thead><tr><th>Edge</th><th>JS2${edgeSuffix}(MD,Sample)</th><th>JS2${edgeSuffix}(MD,Gibbs)</th></tr></thead><tbody>${edgeRows}</tbody></table>`;
    }

    function buildEdgeNetwork() {
      if (!payload.js2_md_sample || !payload.js2_md_sample.length) {
        document.getElementById("edge-network").innerHTML = "<div class='meta'>Edge network unavailable.</div>";
        return;
      }
      const mdIdx = getMdIndex();
      const sampleIdx = getSampleIndex();
      const edges = payload.edges || [];
      const useNormalize = edgeNormalize && edgeNormalize.checked;
      const js2Raw = (payload.js2_md_sample[mdIdx] || [])[sampleIdx] || [];
      const js2 = useNormalize ? normalizeEdgeValues(js2Raw, edges) : js2Raw;
      const n = payload.residue_labels.length || 1;
      const coords = Array.from({ length: n }, (_, i) => {
        const angle = (2 * Math.PI * i) / n;
        return { x: Math.cos(angle), y: Math.sin(angle) };
      });
      const ranked = js2.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 200);
      const xs = [];
      const ys = [];
      let maxVal = ranked.length ? ranked[0][0] : 1;
      ranked.forEach(([val, idx]) => {
        const edge = edges[idx] || [];
        const r = edge[0];
        const s = edge[1];
        if (r === undefined || s === undefined) return;
        xs.push(coords[r].x, coords[s].x, null);
        ys.push(coords[r].y, coords[s].y, null);
      });
      Plotly.react(
        "edge-network",
        [
          {
            type: "scatter",
            mode: "lines",
            x: xs,
            y: ys,
            line: { color: "#38bdf8", width: maxVal ? 1.5 : 1 },
            hoverinfo: "skip",
          },
        ],
        {
          margin: { l: 10, r: 10, t: 10, b: 10 },
          xaxis: { visible: false },
          yaxis: { visible: false },
        },
        { responsive: true }
      );
    }

    function buildEdgeStrengthPlot() {
      const strength = payload.edge_strength || [];
      if (!strength.length) {
        document.getElementById("edge-strength").innerHTML = "<div class='meta'>No coupling strengths available.</div>";
        return;
      }
      const mdIdx = getMdIndex();
      const sampleIdx = getSampleIndex();
      const edges = payload.edges || [];
      const useNormalize = edgeNormalize && edgeNormalize.checked;
      const js2Raw = (payload.js2_md_sample[mdIdx] || [])[sampleIdx] || [];
      const js2 = useNormalize ? normalizeEdgeValues(js2Raw, edges) : js2Raw;
      const edgeLabels = (payload.edges || []).map((edge) => {
        const r = edge[0];
        const s = edge[1];
        const rLabel = payload.residue_labels[r] || r;
        const sLabel = payload.residue_labels[s] || s;
        return `${rLabel}-${sLabel}`;
      });
      Plotly.react(
        "edge-strength",
        [
          {
            type: "scatter",
            mode: "markers",
            x: strength,
            y: js2,
            text: edgeLabels,
            marker: { color: js2, colorscale: "Viridis", size: 6 },
            customdata: js2Raw.map((raw, idx) => {
              const edge = edges[idx] || [];
              const stats = edge.length >= 2 ? edgeStatsFor(edge) : { denom: 0, kprod: null };
              const norm = edgeNormalizedValue(raw, stats);
              return [raw || 0, norm || 0];
            }),
            hovertemplate:
              "Edge %{text}<br>|J|=%{x:.3f}<br>JS2=%{y:.3f}" +
              "<br>JS2(raw): %{customdata[0]:.3f}" +
              "<br>JS2/ln(Kx*Ky): %{customdata[1]:.3f}" +
              "<extra></extra>",
          },
        ],
        {
          margin: { l: 50, r: 20, t: 10, b: 40 },
          xaxis: { title: "Edge strength |J|" },
          yaxis: { title: useNormalize ? "JS2/ln(Kx*Ky)" : "JS2(MD,Sample)" },
        },
        { responsive: true }
      );
    }

    function buildEnergyPlots() {
      if (!payload.energy_bins || payload.energy_bins.length < 2) {
        document.getElementById("energy-hist").innerHTML = "<div class='meta'>Energy histograms unavailable.</div>";
        document.getElementById("energy-cdf").innerHTML = "";
        return;
      }
      const mdIdx = getMdIndex();
      const sampleIdx = getSampleIndex();
      const bins = payload.energy_bins;
      const centers = bins.slice(0, -1).map((b, i) => 0.5 * (b + bins[i + 1]));
      const mdHist = payload.energy_hist_md[mdIdx] || [];
      const sampleHist = payload.energy_hist_sample[sampleIdx] || [];
      const mdCdf = payload.energy_cdf_md[mdIdx] || [];
      const sampleCdf = payload.energy_cdf_sample[sampleIdx] || [];
      Plotly.react(
        "energy-hist",
        [
          { type: "scatter", mode: "lines", x: centers, y: mdHist, name: "MD" },
          { type: "scatter", mode: "lines", x: centers, y: sampleHist, name: "Sample" },
        ],
        { margin: { l: 50, r: 20, t: 10, b: 40 }, xaxis: { title: "Energy" }, yaxis: { title: "Density" } },
        { responsive: true }
      );
      Plotly.react(
        "energy-cdf",
        [
          { type: "scatter", mode: "lines", x: centers, y: mdCdf, name: "MD" },
          { type: "scatter", mode: "lines", x: centers, y: sampleCdf, name: "Sample" },
        ],
        { margin: { l: 50, r: 20, t: 10, b: 40 }, xaxis: { title: "Energy" }, yaxis: { title: "CDF" } },
        { responsive: true }
      );
    }

    function buildNnCdf() {
      if (!payload.nn_bins || !payload.nn_bins.length) {
        document.getElementById("nn-cdf").innerHTML = "<div class='meta'>NN CDFs unavailable.</div>";
        return;
      }
      const mdIdx = getMdIndex();
      const sampleIdx = getSampleIndex();
      const x = payload.nn_bins;
      const sampleToMd = (payload.nn_cdf_sample_to_md[mdIdx] || [])[sampleIdx] || [];
      const mdToSample = (payload.nn_cdf_md_to_sample[mdIdx] || [])[sampleIdx] || [];
      Plotly.react(
        "nn-cdf",
        [
          { type: "scatter", mode: "lines", x: x, y: sampleToMd, name: "Sample → MD" },
          { type: "scatter", mode: "lines", x: x, y: mdToSample, name: "MD → Sample" },
        ],
        { margin: { l: 50, r: 20, t: 10, b: 40 }, xaxis: { title: "Hamming distance" }, yaxis: { title: "CDF" } },
        { responsive: true }
      );
    }

    function buildCrossLikelihood() {
      const fitByOtherRaw = payload.xlik_delta_fit_by_other || [];
      const fitByOther = fitByOtherRaw.map((row) => (row || []).filter((v) => Number.isFinite(v)));
      const otherByOther = payload.xlik_delta_other_by_other || [];
      const otherCounts = payload.xlik_delta_other_counts || [];
      const otherLabels = payload.xlik_other_state_labels || [];
      const aucByOther = payload.xlik_auc_by_other || [];
      const rocFpr = payload.xlik_roc_fpr_by_other || [];
      const rocTpr = payload.xlik_roc_tpr_by_other || [];
      const rocCounts = payload.xlik_roc_counts || [];
      const scoreFitByOtherRaw = payload.xlik_score_fit_by_other || [];
      const scoreFitByOther = scoreFitByOtherRaw.map((row) => (row || []).filter((v) => Number.isFinite(v)));
      const scoreOtherByOther = payload.xlik_score_other_by_other || [];
      const scoreOtherCounts = payload.xlik_score_other_counts || [];
      const aucScoreByOther = payload.xlik_auc_score_by_other || [];
      const rocScoreFpr = payload.xlik_score_roc_fpr_by_other || [];
      const rocScoreTpr = payload.xlik_score_roc_tpr_by_other || [];
      const rocScoreCounts = payload.xlik_score_roc_counts || [];
      const active = (payload.xlik_delta_active || []).filter((v) => Number.isFinite(v));
      const inactive = (payload.xlik_delta_inactive || []).filter((v) => Number.isFinite(v));
      const container = document.getElementById("xlik-hist");
      const rocContainer = document.getElementById("xlik-roc");
      if (!container) {
        return;
      }
      const hasScoreByOther = scoreFitByOther.some((row) => Array.isArray(row) && row.length);
      const hasFitByOther = fitByOther.some((row) => Array.isArray(row) && row.length);
      if (!hasScoreByOther && !hasFitByOther && (!active.length || !inactive.length)) {
        if (xlikMeta) {
          xlikMeta.textContent = "Cross-likelihood classification unavailable.";
        }
        container.innerHTML = "<div class='meta'>No cross-likelihood scores available.</div>";
        if (rocContainer) {
          rocContainer.innerHTML = "";
        }
        return;
      }

      function toRgba(hex, alpha) {
        const cleaned = (hex || "").replace("#", "");
        if (cleaned.length !== 6) return hex;
        const r = parseInt(cleaned.slice(0, 2), 16);
        const g = parseInt(cleaned.slice(2, 4), 16);
        const b = parseInt(cleaned.slice(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
      }

      const palette = ["#38bdf8", "#f59e0b", "#22c55e", "#a855f7", "#f97316", "#10b981", "#e11d48", "#0ea5e9", "#facc15"];
      const fitColor = palette[0];
      const otherPalette = palette.slice(1);
      const traces = [];
      let xaxisTitle = "Delta S = S_fit - S_other";
      if (hasScoreByOther) {
        const activeLabel = (payload.xlik_active_state_labels || []).join(", ") || "Fit MD";
        const fitVals = scoreFitByOther.find((row) => Array.isArray(row) && row.length) || [];
        const fitCount = fitVals.length;
        const metaParts = otherLabels.map((label, idx) => {
          const auc = Number.isFinite(aucScoreByOther[idx]) ? aucScoreByOther[idx].toFixed(3) : "n/a";
          return `${label || `Other ${idx + 1}`}: AUC=${auc}`;
        });
        if (xlikMeta) {
          xlikMeta.textContent = `Scores under fit model · ${activeLabel} (${fitCount}) · ${metaParts.join(" · ")}`;
        }
        xaxisTitle = "S_fit (pseudolikelihood score)";
        if (fitVals.length) {
          traces.push({
            type: "histogram",
            x: fitVals,
            name: `${activeLabel} (S_fit)`,
            opacity: 0.55,
            histnorm: "probability",
            marker: { color: fitColor },
            legendgroup: "score-fit",
          });
        }
        scoreOtherByOther.forEach((row, idx) => {
          const label = otherLabels[idx] || `Other ${idx + 1}`;
          const color = otherPalette[idx % otherPalette.length];
          const otherVals = (row || []).slice(0, scoreOtherCounts[idx] || 0).filter((v) => Number.isFinite(v));
          if (!otherVals.length) return;
          traces.push({
            type: "histogram",
            x: otherVals,
            name: label,
            opacity: 0.65,
            histnorm: "probability",
            marker: { color: color },
            legendgroup: `score-other-${idx}`,
          });
        });
      } else if (hasFitByOther) {
        const activeLabel = (payload.xlik_active_state_labels || []).join(", ") || "Fit MD";
        const fitCount = fitByOther[0] ? fitByOther[0].length : 0;
        const metaParts = otherLabels.map((label, idx) => {
          const auc = Number.isFinite(aucByOther[idx]) ? aucByOther[idx].toFixed(3) : "n/a";
          return `${label || `Other ${idx + 1}`}: AUC=${auc}`;
        });
        if (xlikMeta) {
          xlikMeta.textContent = `Reference: ${activeLabel} (${fitCount}) · ${metaParts.join(" · ")}`;
        }
        fitByOther.forEach((fitVals, idx) => {
          const label = otherLabels[idx] || `Other ${idx + 1}`;
          const color = palette[idx % palette.length];
          const deltaFitColor = toRgba(color, 0.35);
          const otherVals = (otherByOther[idx] || []).slice(0, otherCounts[idx] || 0).filter((v) => Number.isFinite(v));
          traces.push({
            type: "histogram",
            x: otherVals,
            name: label,
            opacity: 0.7,
            histnorm: "probability",
            marker: { color: color },
            legendgroup: `pair-${idx}`,
          });
          traces.push({
            type: "histogram",
            x: fitVals,
            name: `${activeLabel} vs ${label}`,
            opacity: 0.6,
            histnorm: "probability",
            marker: { color: deltaFitColor || color },
            legendgroup: `pair-${idx}`,
          });
        });
      } else {
        const activeLabel = (payload.xlik_active_state_labels || []).join(", ") || "Fit MD";
        const inactiveLabel = (payload.xlik_inactive_state_labels || []).join(", ") || "Other MDs";
        const auc = payload.xlik_auc;
        const aucText = Number.isFinite(auc) ? ` · AUC=${auc.toFixed(3)}` : "";
        if (xlikMeta) {
          xlikMeta.textContent = `Reference: ${activeLabel} (${active.length}) · Other: ${inactiveLabel} (${inactive.length})${aucText}`;
        }
        traces.push({
          type: "histogram",
          x: active,
          name: "Fit",
          opacity: 0.65,
          histnorm: "probability",
          marker: { color: "#38bdf8" },
        });
        traces.push({
          type: "histogram",
          x: inactive,
          name: "Other",
          opacity: 0.65,
          histnorm: "probability",
          marker: { color: "#f59e0b" },
        });
      }

      Plotly.react(
        "xlik-hist",
        traces,
        {
          barmode: "overlay",
          margin: { l: 60, r: 20, t: 10, b: 40 },
          xaxis: { title: xaxisTitle },
          yaxis: { title: "Probability" },
        },
        { responsive: true }
      );

      if (rocContainer) {
        const useScoreRoc = hasScoreByOther && rocScoreFpr.length && rocScoreTpr.length && rocScoreCounts.length;
        if (useScoreRoc || (rocFpr.length && rocTpr.length && rocCounts.length)) {
          const rocTraces = [];
          const rocRows = useScoreRoc ? rocScoreFpr : rocFpr;
          const rocCols = useScoreRoc ? rocScoreTpr : rocTpr;
          const rocN = useScoreRoc ? rocScoreCounts : rocCounts;
          const rocAuc = useScoreRoc ? aucScoreByOther : aucByOther;
          rocRows.forEach((row, idx) => {
            const n = rocN[idx] || 0;
            if (!n) return;
            const label = otherLabels[idx] || `Other ${idx + 1}`;
            const auc = Number.isFinite(rocAuc[idx]) ? rocAuc[idx].toFixed(3) : "n/a";
            const color = palette[idx % palette.length];
            rocTraces.push({
              type: "scatter",
              mode: "lines",
              x: row.slice(0, n),
              y: (rocCols[idx] || []).slice(0, n),
              name: `${label} (AUC=${auc})`,
              line: { color: color, width: 2 },
            });
          });
          if (!rocTraces.length) {
            rocContainer.innerHTML = "";
          } else {
            Plotly.react(
              "xlik-roc",
              rocTraces,
              {
                margin: { l: 60, r: 20, t: 10, b: 40 },
                xaxis: { title: "False positive rate" },
                yaxis: { title: "True positive rate", range: [0, 1] },
              },
              { responsive: true }
            );
          }
        } else {
          rocContainer.innerHTML = "";
        }
      }
    }

    function renderAll() {
      setMeta();
      buildResidueBarcode();
      buildEdgeBarcode();
      buildEdgeHeatmap();
      buildTopTables();
      buildEdgeNetwork();
      buildEdgeStrengthPlot();
      buildEnergyPlots();
      buildNnCdf();
      buildCrossLikelihood();
    }

    mdSelect.addEventListener("change", renderAll);
    sampleSelect.addEventListener("change", renderAll);
    if (edgeNormalize) {
      edgeNormalize.addEventListener("change", renderAll);
    }
    renderAll();
  </script>
</body>
</html>
"""

    html = html.replace("__PAYLOAD__", json.dumps(payload))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def plot_cross_likelihood_report_from_npz(
    *,
    summary_path: str | Path,
    out_path: str | Path,
) -> Path:
    with np.load(summary_path, allow_pickle=False) as data:
        xlik_delta_active = data["xlik_delta_active"] if "xlik_delta_active" in data else np.array([], dtype=float)
        xlik_delta_inactive = data["xlik_delta_inactive"] if "xlik_delta_inactive" in data else np.array([], dtype=float)
        xlik_auc = data["xlik_auc"] if "xlik_auc" in data else np.array([], dtype=float)
        xlik_active_state_ids = data["xlik_active_state_ids"] if "xlik_active_state_ids" in data else np.array([], dtype=str)
        xlik_inactive_state_ids = data["xlik_inactive_state_ids"] if "xlik_inactive_state_ids" in data else np.array([], dtype=str)
        xlik_active_state_labels = data["xlik_active_state_labels"] if "xlik_active_state_labels" in data else np.array([], dtype=str)
        xlik_inactive_state_labels = data["xlik_inactive_state_labels"] if "xlik_inactive_state_labels" in data else np.array([], dtype=str)
        xlik_delta_fit_by_other = data["xlik_delta_fit_by_other"] if "xlik_delta_fit_by_other" in data else np.array([], dtype=float)
        xlik_delta_other_by_other = data["xlik_delta_other_by_other"] if "xlik_delta_other_by_other" in data else np.array([], dtype=float)
        xlik_delta_other_counts = data["xlik_delta_other_counts"] if "xlik_delta_other_counts" in data else np.array([], dtype=int)
        xlik_other_state_ids = data["xlik_other_state_ids"] if "xlik_other_state_ids" in data else np.array([], dtype=str)
        xlik_other_state_labels = data["xlik_other_state_labels"] if "xlik_other_state_labels" in data else np.array([], dtype=str)
        xlik_auc_by_other = data["xlik_auc_by_other"] if "xlik_auc_by_other" in data else np.array([], dtype=float)
        xlik_roc_fpr_by_other = data["xlik_roc_fpr_by_other"] if "xlik_roc_fpr_by_other" in data else np.array([], dtype=float)
        xlik_roc_tpr_by_other = data["xlik_roc_tpr_by_other"] if "xlik_roc_tpr_by_other" in data else np.array([], dtype=float)
        xlik_roc_counts = data["xlik_roc_counts"] if "xlik_roc_counts" in data else np.array([], dtype=int)
        xlik_score_fit_by_other = data["xlik_score_fit_by_other"] if "xlik_score_fit_by_other" in data else np.array([], dtype=float)
        xlik_score_other_by_other = data["xlik_score_other_by_other"] if "xlik_score_other_by_other" in data else np.array([], dtype=float)
        xlik_score_other_counts = data["xlik_score_other_counts"] if "xlik_score_other_counts" in data else np.array([], dtype=int)
        xlik_auc_score_by_other = data["xlik_auc_score_by_other"] if "xlik_auc_score_by_other" in data else np.array([], dtype=float)
        xlik_score_roc_fpr_by_other = data["xlik_score_roc_fpr_by_other"] if "xlik_score_roc_fpr_by_other" in data else np.array([], dtype=float)
        xlik_score_roc_tpr_by_other = data["xlik_score_roc_tpr_by_other"] if "xlik_score_roc_tpr_by_other" in data else np.array([], dtype=float)
        xlik_score_roc_counts = data["xlik_score_roc_counts"] if "xlik_score_roc_counts" in data else np.array([], dtype=int)

    payload = {
        "xlik_delta_active": xlik_delta_active.tolist() if xlik_delta_active.size else [],
        "xlik_delta_inactive": xlik_delta_inactive.tolist() if xlik_delta_inactive.size else [],
        "xlik_auc": float(xlik_auc[0]) if xlik_auc.size else None,
        "xlik_active_state_ids": [str(v) for v in xlik_active_state_ids.tolist()] if xlik_active_state_ids.size else [],
        "xlik_inactive_state_ids": [str(v) for v in xlik_inactive_state_ids.tolist()] if xlik_inactive_state_ids.size else [],
        "xlik_active_state_labels": [str(v) for v in xlik_active_state_labels.tolist()] if xlik_active_state_labels.size else [],
        "xlik_inactive_state_labels": [str(v) for v in xlik_inactive_state_labels.tolist()] if xlik_inactive_state_labels.size else [],
        "xlik_delta_fit_by_other": xlik_delta_fit_by_other.tolist() if xlik_delta_fit_by_other.size else [],
        "xlik_delta_other_by_other": xlik_delta_other_by_other.tolist() if xlik_delta_other_by_other.size else [],
        "xlik_delta_other_counts": xlik_delta_other_counts.tolist() if xlik_delta_other_counts.size else [],
        "xlik_other_state_ids": [str(v) for v in xlik_other_state_ids.tolist()] if xlik_other_state_ids.size else [],
        "xlik_other_state_labels": [str(v) for v in xlik_other_state_labels.tolist()] if xlik_other_state_labels.size else [],
        "xlik_auc_by_other": xlik_auc_by_other.tolist() if xlik_auc_by_other.size else [],
        "xlik_roc_fpr_by_other": xlik_roc_fpr_by_other.tolist() if xlik_roc_fpr_by_other.size else [],
        "xlik_roc_tpr_by_other": xlik_roc_tpr_by_other.tolist() if xlik_roc_tpr_by_other.size else [],
        "xlik_roc_counts": xlik_roc_counts.tolist() if xlik_roc_counts.size else [],
        "xlik_score_fit_by_other": xlik_score_fit_by_other.tolist() if xlik_score_fit_by_other.size else [],
        "xlik_score_other_by_other": xlik_score_other_by_other.tolist() if xlik_score_other_by_other.size else [],
        "xlik_score_other_counts": xlik_score_other_counts.tolist() if xlik_score_other_counts.size else [],
        "xlik_auc_score_by_other": xlik_auc_score_by_other.tolist() if xlik_auc_score_by_other.size else [],
        "xlik_score_roc_fpr_by_other": xlik_score_roc_fpr_by_other.tolist() if xlik_score_roc_fpr_by_other.size else [],
        "xlik_score_roc_tpr_by_other": xlik_score_roc_tpr_by_other.tolist() if xlik_score_roc_tpr_by_other.size else [],
        "xlik_score_roc_counts": xlik_score_roc_counts.tolist() if xlik_score_roc_counts.size else [],
    }

    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Cross-likelihood classification</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { padding: 18px 20px 32px; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    h2 { margin: 12px 0 6px; font-size: 18px; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 12px; }
    .plot { height: 320px; }
    .plot.short { height: 260px; }
    .meta { font-size: 12px; color: #94a3b8; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Cross-likelihood classification</h1>
    <p class="meta">Global cross-likelihood scores comparing the fit model against other MD sources.</p>
    <div class="card">
      <div class="meta" id="xlik-meta"></div>
      <div id="xlik-hist" class="plot"></div>
      <div id="xlik-roc" class="plot short"></div>
    </div>
  </div>

  <script>
    const payload = __PAYLOAD__;
    const xlikMeta = document.getElementById("xlik-meta");

    function buildCrossLikelihood() {
      const fitByOtherRaw = payload.xlik_delta_fit_by_other || [];
      const fitByOther = fitByOtherRaw.map((row) => (row || []).filter((v) => Number.isFinite(v)));
      const otherByOther = payload.xlik_delta_other_by_other || [];
      const otherCounts = payload.xlik_delta_other_counts || [];
      const otherLabels = payload.xlik_other_state_labels || [];
      const aucByOther = payload.xlik_auc_by_other || [];
      const rocFpr = payload.xlik_roc_fpr_by_other || [];
      const rocTpr = payload.xlik_roc_tpr_by_other || [];
      const rocCounts = payload.xlik_roc_counts || [];
      const scoreFitByOtherRaw = payload.xlik_score_fit_by_other || [];
      const scoreFitByOther = scoreFitByOtherRaw.map((row) => (row || []).filter((v) => Number.isFinite(v)));
      const scoreOtherByOther = payload.xlik_score_other_by_other || [];
      const scoreOtherCounts = payload.xlik_score_other_counts || [];
      const aucScoreByOther = payload.xlik_auc_score_by_other || [];
      const rocScoreFpr = payload.xlik_score_roc_fpr_by_other || [];
      const rocScoreTpr = payload.xlik_score_roc_tpr_by_other || [];
      const rocScoreCounts = payload.xlik_score_roc_counts || [];
      const active = (payload.xlik_delta_active || []).filter((v) => Number.isFinite(v));
      const inactive = (payload.xlik_delta_inactive || []).filter((v) => Number.isFinite(v));
      const container = document.getElementById("xlik-hist");
      const rocContainer = document.getElementById("xlik-roc");
      if (!container) {
        return;
      }
      const hasScoreByOther = scoreFitByOther.some((row) => Array.isArray(row) && row.length);
      const hasFitByOther = fitByOther.some((row) => Array.isArray(row) && row.length);
      if (!hasScoreByOther && !hasFitByOther && (!active.length || !inactive.length)) {
        if (xlikMeta) {
          xlikMeta.textContent = "Cross-likelihood classification unavailable.";
        }
        container.innerHTML = "<div class='meta'>No cross-likelihood scores available.</div>";
        if (rocContainer) {
          rocContainer.innerHTML = "";
        }
        return;
      }

      function toRgba(hex, alpha) {
        const cleaned = (hex || "").replace("#", "");
        if (cleaned.length !== 6) return hex;
        const r = parseInt(cleaned.slice(0, 2), 16);
        const g = parseInt(cleaned.slice(2, 4), 16);
        const b = parseInt(cleaned.slice(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
      }

      const palette = ["#38bdf8", "#f59e0b", "#22c55e", "#a855f7", "#f97316", "#10b981", "#e11d48", "#0ea5e9", "#facc15"];
      const fitColor = palette[0];
      const otherPalette = palette.slice(1);
      const traces = [];
      let xaxisTitle = "Delta S = S_fit - S_other";
      if (hasScoreByOther) {
        const activeLabel = (payload.xlik_active_state_labels || []).join(", ") || "Fit MD";
        const fitVals = scoreFitByOther.find((row) => Array.isArray(row) && row.length) || [];
        const fitCount = fitVals.length;
        const metaParts = otherLabels.map((label, idx) => {
          const auc = Number.isFinite(aucScoreByOther[idx]) ? aucScoreByOther[idx].toFixed(3) : "n/a";
          return `${label || `Other ${idx + 1}`}: AUC=${auc}`;
        });
        if (xlikMeta) {
          xlikMeta.textContent = `Scores under fit model · ${activeLabel} (${fitCount}) · ${metaParts.join(" · ")}`;
        }
        xaxisTitle = "S_fit (pseudolikelihood score)";
        if (fitVals.length) {
          traces.push({
            type: "histogram",
            x: fitVals,
            name: `${activeLabel} (S_fit)`,
            opacity: 0.55,
            histnorm: "probability",
            marker: { color: fitColor },
            legendgroup: "score-fit",
          });
        }
        scoreOtherByOther.forEach((row, idx) => {
          const label = otherLabels[idx] || `Other ${idx + 1}`;
          const color = otherPalette[idx % otherPalette.length];
          const otherVals = (row || []).slice(0, scoreOtherCounts[idx] || 0).filter((v) => Number.isFinite(v));
          if (!otherVals.length) return;
          traces.push({
            type: "histogram",
            x: otherVals,
            name: label,
            opacity: 0.65,
            histnorm: "probability",
            marker: { color: color },
            legendgroup: `score-other-${idx}`,
          });
        });
      } else if (hasFitByOther) {
        const activeLabel = (payload.xlik_active_state_labels || []).join(", ") || "Fit MD";
        const fitCount = fitByOther[0] ? fitByOther[0].length : 0;
        const metaParts = otherLabels.map((label, idx) => {
          const auc = Number.isFinite(aucByOther[idx]) ? aucByOther[idx].toFixed(3) : "n/a";
          return `${label || `Other ${idx + 1}`}: AUC=${auc}`;
        });
        if (xlikMeta) {
          xlikMeta.textContent = `Reference: ${activeLabel} (${fitCount}) · ${metaParts.join(" · ")}`;
        }
        fitByOther.forEach((fitVals, idx) => {
          const label = otherLabels[idx] || `Other ${idx + 1}`;
          const color = palette[idx % palette.length];
          const deltaFitColor = toRgba(color, 0.35);
          const otherVals = (otherByOther[idx] || []).slice(0, otherCounts[idx] || 0).filter((v) => Number.isFinite(v));
          traces.push({
            type: "histogram",
            x: otherVals,
            name: label,
            opacity: 0.7,
            histnorm: "probability",
            marker: { color: color },
            legendgroup: `pair-${idx}`,
          });
          traces.push({
            type: "histogram",
            x: fitVals,
            name: `${activeLabel} vs ${label}`,
            opacity: 0.6,
            histnorm: "probability",
            marker: { color: deltaFitColor || color },
            legendgroup: `pair-${idx}`,
          });
        });
      } else {
        const activeLabel = (payload.xlik_active_state_labels || []).join(", ") || "Fit MD";
        const inactiveLabel = (payload.xlik_inactive_state_labels || []).join(", ") || "Other MDs";
        const auc = payload.xlik_auc;
        const aucText = Number.isFinite(auc) ? ` · AUC=${auc.toFixed(3)}` : "";
        if (xlikMeta) {
          xlikMeta.textContent = `Reference: ${activeLabel} (${active.length}) · Other: ${inactiveLabel} (${inactive.length})${aucText}`;
        }
        traces.push({
          type: "histogram",
          x: active,
          name: "Fit",
          opacity: 0.65,
          histnorm: "probability",
          marker: { color: "#38bdf8" },
        });
        traces.push({
          type: "histogram",
          x: inactive,
          name: "Other",
          opacity: 0.65,
          histnorm: "probability",
          marker: { color: "#f59e0b" },
        });
      }

      Plotly.react(
        "xlik-hist",
        traces,
        {
          barmode: "overlay",
          margin: { l: 60, r: 20, t: 10, b: 40 },
          xaxis: { title: xaxisTitle },
          yaxis: { title: "Probability" },
        },
        { responsive: true }
      );

      if (rocContainer) {
        const useScoreRoc = hasScoreByOther && rocScoreFpr.length && rocScoreTpr.length && rocScoreCounts.length;
        if (useScoreRoc || (rocFpr.length && rocTpr.length && rocCounts.length)) {
          const rocTraces = [];
          const rocRows = useScoreRoc ? rocScoreFpr : rocFpr;
          const rocCols = useScoreRoc ? rocScoreTpr : rocTpr;
          const rocN = useScoreRoc ? rocScoreCounts : rocCounts;
          const rocAuc = useScoreRoc ? aucScoreByOther : aucByOther;
          rocRows.forEach((row, idx) => {
            const n = rocN[idx] || 0;
            if (!n) return;
            const label = otherLabels[idx] || `Other ${idx + 1}`;
            const auc = Number.isFinite(rocAuc[idx]) ? rocAuc[idx].toFixed(3) : "n/a";
            const color = palette[idx % palette.length];
            rocTraces.push({
              type: "scatter",
              mode: "lines",
              x: row.slice(0, n),
              y: (rocCols[idx] || []).slice(0, n),
              name: `${label} (AUC=${auc})`,
              line: { color: color, width: 2 },
            });
          });
          if (!rocTraces.length) {
            rocContainer.innerHTML = "";
          } else {
            Plotly.react(
              "xlik-roc",
              rocTraces,
              {
                margin: { l: 60, r: 20, t: 10, b: 40 },
                xaxis: { title: "False positive rate" },
                yaxis: { title: "True positive rate", range: [0, 1] },
              },
              { responsive: true }
            );
          }
        } else {
          rocContainer.innerHTML = "";
        }
      }
    }

    buildCrossLikelihood();
  </script>
</body>
</html>
"""

    html = html.replace("__PAYLOAD__", json.dumps(payload))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


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
