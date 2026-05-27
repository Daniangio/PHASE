import { useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';

export function pickEnergyColor(index) {
  const palette = [
    '#2563eb',
    '#dc2626',
    '#16a34a',
    '#9333ea',
    '#ea580c',
    '#0891b2',
    '#be123c',
    '#4f46e5',
  ];
  return palette[index % palette.length];
}

function finiteValues(values) {
  return (Array.isArray(values) ? values : []).map(Number).filter((v) => Number.isFinite(v));
}

function mean(values) {
  if (!values.length) return NaN;
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

function std(values, mu = mean(values)) {
  if (!values.length || !Number.isFinite(mu)) return NaN;
  const v = values.reduce((acc, x) => acc + (x - mu) ** 2, 0) / Math.max(1, values.length);
  return Math.sqrt(Math.max(0, v));
}

function linspace(start, end, n) {
  if (n <= 1) return [start];
  const step = (end - start) / (n - 1);
  return Array.from({ length: n }, (_, i) => start + i * step);
}

function kdeCurve(values, xs) {
  const arr = finiteValues(values);
  if (arr.length < 2) return [];
  const mu = mean(arr);
  const sd = std(arr, mu);
  const sorted = [...arr].sort((a, b) => a - b);
  const q1 = sorted[Math.floor((sorted.length - 1) * 0.25)];
  const q3 = sorted[Math.floor((sorted.length - 1) * 0.75)];
  const iqr = q3 - q1;
  const sigma = Number.isFinite(iqr) && iqr > 0 ? Math.min(sd || iqr, iqr / 1.34) : sd;
  const bandwidth = Math.max(1e-9, 0.9 * (sigma || 1) * arr.length ** -0.2);
  const norm = 1 / (Math.sqrt(2 * Math.PI) * bandwidth * arr.length);
  return xs.map((x) => {
    let s = 0;
    for (let i = 0; i < arr.length; i += 1) {
      const z = (x - arr[i]) / bandwidth;
      s += Math.exp(-0.5 * z * z);
    }
    return s * norm;
  });
}

function smoothHistogramCurve(bins, density) {
  if (!Array.isArray(bins) || !Array.isArray(density) || bins.length < 2 || density.length < 1) {
    return { x: [], y: [] };
  }
  const centers = [];
  const vals = [];
  for (let i = 0; i < Math.min(density.length, bins.length - 1); i += 1) {
    const y = Number(density[i]);
    const a = Number(bins[i]);
    const b = Number(bins[i + 1]);
    if (!Number.isFinite(y) || !Number.isFinite(a) || !Number.isFinite(b)) continue;
    centers.push((a + b) / 2);
    vals.push(y);
  }
  if (!centers.length) return { x: [], y: [] };
  const y = vals.map((v, i) => {
    const prev = vals[Math.max(0, i - 1)];
    const next = vals[Math.min(vals.length - 1, i + 1)];
    return 0.25 * prev + 0.5 * v + 0.25 * next;
  });
  return { x: centers, y };
}

export function energySeriesId(series, index) {
  return String(series?.id || series?.sample_id || series?.analysis_id || series?.label || series?.name || `series-${index}`);
}

function isMdEnergySeries(series) {
  const type = String(series?.type || series?.kind || series?.sample_type || '').toLowerCase();
  const label = String(series?.label || series?.name || '').toLowerCase();
  return type === 'md_eval' || type === 'md' || label.startsWith('md ');
}

export function useEnergySeriesSelection(series) {
  const allIds = useMemo(() => (Array.isArray(series) ? series.map((s, idx) => energySeriesId(s, idx)) : []), [series]);
  const allIdsKey = allIds.join('\u0001');
  const [selectedIds, setSelectedIds] = useState(null);

  useEffect(() => {
    setSelectedIds((prev) => {
      if (!allIds.length) return [];
      if (prev === null) return allIds;
      const allowed = new Set(allIds);
      const retained = prev.filter((id) => allowed.has(id));
      return retained.length ? retained : allIds;
    });
  }, [allIdsKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectedSet = useMemo(() => new Set(selectedIds || []), [selectedIds]);
  const selectedSeries = useMemo(
    () => (Array.isArray(series) ? series.filter((s, idx) => selectedSet.has(energySeriesId(s, idx))) : []),
    [series, selectedSet]
  );

  return {
    allIds,
    selectedIds: selectedIds || [],
    setSelectedIds,
    selectedSeries,
  };
}

export function EnergySeriesSelectorButton({
  series,
  selectedIds,
  onChange,
  dark = false,
  label = 'Select trajectories',
}) {
  const [open, setOpen] = useState(false);
  const entries = useMemo(
    () =>
      (Array.isArray(series) ? series : []).map((s, idx) => ({
        id: energySeriesId(s, idx),
        label: s?.label || s?.name || `series ${idx + 1}`,
        type: s?.type || s?.kind || s?.sample_type || 'sample',
        isMd: isMdEnergySeries(s),
      })),
    [series]
  );
  const selectedSet = useMemo(() => new Set(selectedIds || []), [selectedIds]);
  const selectedCount = entries.filter((entry) => selectedSet.has(entry.id)).length;

  const setEntry = (id, checked) => {
    const next = new Set(selectedIds || []);
    if (checked) next.add(id);
    else next.delete(id);
    onChange(Array.from(next));
  };

  const buttonClass = dark
    ? 'rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5 text-xs text-gray-100 hover:bg-gray-900'
    : 'rounded border border-gray-300 bg-white px-2 py-1 text-[11px] text-gray-800 hover:bg-gray-50';

  return (
    <>
      <button type="button" className={buttonClass} onClick={() => setOpen(true)} disabled={!entries.length}>
        {label} ({selectedCount}/{entries.length})
      </button>
      {open ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="w-full max-w-xl rounded-lg border border-gray-700 bg-gray-950 shadow-xl">
            <div className="flex items-start justify-between gap-3 border-b border-gray-800 px-4 py-3">
              <div>
                <h3 className="text-sm font-semibold text-gray-100">Energy trajectories</h3>
                <p className="mt-1 text-xs text-gray-400">
                  Hidden trajectories are removed from the graph and from the legend.
                </p>
              </div>
              <button type="button" className="text-gray-400 hover:text-gray-100" onClick={() => setOpen(false)}>
                Close
              </button>
            </div>
            <div className="flex flex-wrap gap-2 border-b border-gray-800 px-4 py-3">
              <button
                type="button"
                className="rounded border border-gray-700 px-2 py-1 text-xs text-gray-200 hover:bg-gray-900"
                onClick={() => onChange(entries.map((entry) => entry.id))}
              >
                Select all
              </button>
              <button
                type="button"
                className="rounded border border-gray-700 px-2 py-1 text-xs text-gray-200 hover:bg-gray-900"
                onClick={() => onChange(entries.filter((entry) => entry.isMd).map((entry) => entry.id))}
              >
                MD only
              </button>
              <button
                type="button"
                className="rounded border border-gray-700 px-2 py-1 text-xs text-gray-200 hover:bg-gray-900"
                onClick={() => onChange([])}
              >
                Clear
              </button>
            </div>
            <div className="max-h-[60vh] overflow-y-auto px-4 py-3">
              {!entries.length ? (
                <p className="text-sm text-gray-400">No energy trajectories are available.</p>
              ) : (
                <div className="space-y-2">
                  {entries.map((entry) => (
                    <label
                      key={entry.id}
                      className="flex items-center justify-between gap-3 rounded-md border border-gray-800 bg-gray-900/60 px-3 py-2 text-sm text-gray-100"
                    >
                      <span className="min-w-0">
                        <span className="block truncate">{entry.label}</span>
                        <span className="text-[11px] uppercase tracking-wide text-gray-500">{entry.type}</span>
                      </span>
                      <input
                        type="checkbox"
                        checked={selectedSet.has(entry.id)}
                        onChange={(event) => setEntry(entry.id, event.target.checked)}
                      />
                    </label>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}

export function buildEnergyDistributionPlot({
  series,
  mode = 'histogram',
  title = '',
  xTitle = 'Energy',
  height = 300,
  background = 'white',
} = {}) {
  const valid = (Array.isArray(series) ? series : []).filter((s) => {
    const values = finiteValues(s.values || s.energies);
    const bins = Array.isArray(s.bins) ? s.bins : [];
    const density = Array.isArray(s.density || s.hist || s.histogram) ? (s.density || s.hist || s.histogram) : [];
    return values.length || (bins.length > 1 && density.length);
  });
  if (!valid.length) return null;

  let globalMin = Infinity;
  let globalMax = -Infinity;
  for (const s of valid) {
    const values = finiteValues(s.values || s.energies);
    if (values.length) {
      globalMin = Math.min(globalMin, ...values);
      globalMax = Math.max(globalMax, ...values);
    }
    const bins = Array.isArray(s.bins) ? s.bins.map(Number).filter(Number.isFinite) : [];
    if (bins.length) {
      globalMin = Math.min(globalMin, bins[0]);
      globalMax = Math.max(globalMax, bins[bins.length - 1]);
    }
  }
  if (!Number.isFinite(globalMin) || !Number.isFinite(globalMax)) return null;
  if (globalMax <= globalMin) {
    globalMax = globalMin + 1;
    globalMin -= 1;
  }
  const range = globalMax - globalMin;
  const pad = range * 0.04;
  const x0 = globalMin - pad;
  const x1 = globalMax + pad;
  const xs = linspace(x0, x1, 220);
  const binSize = Math.max(1e-9, range / 60);

  const traces = [];
  const shapes = [];
  const annotations = [];
  valid.forEach((s, idx) => {
    const color = s.color || pickEnergyColor(idx);
    const label = s.label || s.name || `series ${idx + 1}`;
    const values = finiteValues(s.values || s.energies);
    const isPose = s.kind === 'state_pose' || values.length === 1;
    if (isPose && values.length) {
      const x = Number(values[0]);
      shapes.push({
        type: 'line',
        xref: 'x',
        yref: 'paper',
        x0: x,
        x1: x,
        y0: 0,
        y1: 1,
        line: { color, width: 2, dash: 'dot' },
      });
      annotations.push({
        x,
        y: 1,
        xref: 'x',
        yref: 'paper',
        xanchor: 'left',
        yanchor: 'bottom',
        text: label,
        showarrow: false,
        font: { size: 10, color },
        bgcolor: 'rgba(255,255,255,0.75)',
        bordercolor: color,
        borderwidth: 1,
        borderpad: 2,
      });
      return;
    }

    if (mode === 'histogram') {
      if (values.length > 1) {
        traces.push({
          x: values,
          type: 'histogram',
          histnorm: 'probability density',
          name: `${label} histogram`,
          opacity: 0.18,
          marker: { color: '#6b7280', line: { color: '#374151', width: 0.5 } },
          autobinx: false,
          xbins: { start: globalMin, end: globalMax, size: binSize },
          bingroup: 'energy-distribution',
          showlegend: false,
        });
      } else if (Array.isArray(s.bins)) {
        const density = s.density || s.hist || s.histogram || [];
        const centers = [];
        const widths = [];
        const y = [];
        for (let i = 0; i < Math.min(density.length, s.bins.length - 1); i += 1) {
          centers.push((Number(s.bins[i]) + Number(s.bins[i + 1])) / 2);
          widths.push(Math.abs(Number(s.bins[i + 1]) - Number(s.bins[i])));
          y.push(Number(density[i]) || 0);
        }
        traces.push({
          x: centers,
          y,
          width: widths,
          type: 'bar',
          name: `${label} histogram`,
          opacity: 0.16,
          marker: { color: '#6b7280', line: { color: '#374151', width: 0.4 } },
          showlegend: false,
        });
      }
    }

    let curve = [];
    if (values.length > 1) {
      curve = kdeCurve(values, xs);
      traces.push({
        x: xs,
        y: curve,
        type: 'scatter',
        mode: 'lines',
        name: label,
        line: { color, width: 2.5 },
        fill: mode === 'curves' ? 'tozeroy' : undefined,
        fillcolor: mode === 'curves' ? `${color}22` : undefined,
      });
    } else {
      const smooth = smoothHistogramCurve(s.bins, s.density || s.hist || s.histogram || []);
      if (smooth.x.length) {
        traces.push({
          x: smooth.x,
          y: smooth.y,
          type: 'scatter',
          mode: 'lines',
          name: label,
          line: { color, width: 2.5 },
          fill: mode === 'curves' ? 'tozeroy' : undefined,
          fillcolor: mode === 'curves' ? `${color}22` : undefined,
        });
      }
    }
  });

  const dark = background === 'dark';
  return {
    data: traces,
    layout: {
      title: title ? { text: title, font: { size: 13 } } : undefined,
      height,
      margin: { l: 52, r: 18, t: title ? 34 : 12, b: 48 },
      paper_bgcolor: dark ? 'rgba(0,0,0,0)' : '#ffffff',
      plot_bgcolor: dark ? 'rgba(0,0,0,0)' : '#ffffff',
      font: { color: dark ? '#d1d5db' : '#111827' },
      barmode: 'overlay',
      xaxis: { title: xTitle, color: dark ? '#d1d5db' : '#111827', zeroline: true },
      yaxis: { title: 'Density', color: dark ? '#d1d5db' : '#111827', rangemode: 'tozero' },
      shapes,
      annotations,
      legend: { orientation: 'h', y: -0.22 },
    },
    config: { displayModeBar: false, responsive: true },
  };
}

export default function EnergyDistributionPlot({ plot, height = 300 }) {
  if (!plot) return null;
  return (
    <Plot
      data={plot.data}
      layout={{ ...plot.layout, height }}
      config={plot.config || { displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: '100%', height: `${height}px` }}
    />
  );
}
