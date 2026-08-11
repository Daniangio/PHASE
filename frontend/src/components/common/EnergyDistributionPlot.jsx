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

function hexToTransparent(hex, alpha) {
  const clean = String(hex || '#64748b').replace('#', '');
  const normalized = clean.length === 3 ? clean.split('').map((value) => `${value}${value}`).join('') : clean;
  const parsed = Number.parseInt(normalized, 16);
  if (!Number.isFinite(parsed)) return `rgba(100,116,139,${alpha})`;
  return `rgba(${(parsed >> 16) & 255},${(parsed >> 8) & 255},${parsed & 255},${alpha})`;
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

/** Build a normalized Gaussian KDE, optionally weighting repeated unique rows. */
export function buildKdeCurve(values, { weights = null, points = 220, range = null } = {}) {
  const rows = (Array.isArray(values) ? values : [])
    .map((value, index) => ({
      value: Number(value),
      weight: Math.max(0, Number(Array.isArray(weights) ? weights[index] : 1)),
    }))
    .filter((row) => Number.isFinite(row.value) && row.weight > 0);
  // A one-point distribution is still useful in a curves-only view. Render a
  // narrow normalized Gaussian around it instead of leaving the plot empty.
  if (!rows.length) return { x: [], y: [] };

  const totalWeight = rows.reduce((sum, row) => sum + row.weight, 0);
  const meanValue = rows.reduce((sum, row) => sum + row.weight * row.value, 0) / totalWeight;
  const variance = rows.reduce((sum, row) => sum + row.weight * (row.value - meanValue) ** 2, 0) / totalWeight;
  const deviation = Math.sqrt(Math.max(0, variance));
  const minValue = Math.min(...rows.map((row) => row.value));
  const maxValue = Math.max(...rows.map((row) => row.value));
  const span = Math.max(maxValue - minValue, Math.abs(meanValue) * 0.02, 1e-6);
  const effectiveN = (totalWeight ** 2) / rows.reduce((sum, row) => sum + row.weight ** 2, 0);
  const bandwidth = Math.max(span * 1e-4, 1e-9, 1.06 * Math.max(deviation, span * 0.05) * effectiveN ** -0.2);
  const start = Array.isArray(range) && Number.isFinite(Number(range[0])) ? Number(range[0]) : minValue - 0.04 * span;
  const end = Array.isArray(range) && Number.isFinite(Number(range[1])) ? Number(range[1]) : maxValue + 0.04 * span;
  const x = linspace(start, end > start ? end : start + span, points);
  const normalizer = 1 / (Math.sqrt(2 * Math.PI) * bandwidth * totalWeight);
  const y = x.map((coordinate) => rows.reduce((sum, row) => {
    const z = (coordinate - row.value) / bandwidth;
    return sum + row.weight * Math.exp(-0.5 * z * z);
  }, 0) * normalizer);
  return { x, y };
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
  onColorChange = null,
  showColors = false,
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
        color: s?.color || pickEnergyColor(idx),
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
                    <div
                      key={entry.id}
                      className="flex items-center justify-between gap-3 rounded-md border border-gray-800 bg-gray-900/60 px-3 py-2 text-sm text-gray-100"
                    >
                      <span className="min-w-0">
                        <span className="block truncate">{entry.label}</span>
                        <span className="text-[11px] uppercase tracking-wide text-gray-500">{entry.type}</span>
                      </span>
                      <span className="flex items-center gap-3">
                        {showColors && onColorChange ? (
                          <input
                            type="color"
                            value={entry.color}
                            onChange={(event) => onColorChange(entry.id, event.target.value)}
                            className="h-7 w-9 cursor-pointer rounded border border-gray-700 bg-transparent p-0.5"
                            title={`Color for ${entry.label}`}
                            aria-label={`Color for ${entry.label}`}
                          />
                        ) : null}
                        <input
                          type="checkbox"
                          checked={selectedSet.has(entry.id)}
                          onChange={(event) => setEntry(entry.id, event.target.checked)}
                          aria-label={`Show ${entry.label}`}
                        />
                      </span>
                    </div>
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
  xRange = null,
} = {}) {
  const valid = (Array.isArray(series) ? series : []).filter((s) => {
    const values = finiteValues(s.values || s.energies);
    const bins = Array.isArray(s.bins) ? s.bins : [];
    const density = Array.isArray(s.density || s.hist || s.histogram) ? (s.density || s.hist || s.histogram) : [];
    return values.length || (bins.length > 1 && density.length);
  });
  if (!valid.length) return null;

  let globalMin = Array.isArray(xRange) && Number.isFinite(Number(xRange[0])) ? Number(xRange[0]) : Infinity;
  let globalMax = Array.isArray(xRange) && Number.isFinite(Number(xRange[1])) ? Number(xRange[1]) : -Infinity;
  const hasProvidedRange = Array.isArray(xRange) && Number.isFinite(Number(xRange[0])) && Number.isFinite(Number(xRange[1]));
  for (const s of valid) {
    if (hasProvidedRange) break;
    const values = finiteValues(s.values || s.energies);
    if (values.length) {
      for (let i = 0; i < values.length; i += 1) {
        const value = values[i];
        if (value < globalMin) globalMin = value;
        if (value > globalMax) globalMax = value;
      }
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
    const isPose = values.length === 1;
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
          opacity: 0.16,
          marker: { color, line: { color: hexToTransparent(color, 0.68), width: 0.7 } },
          autobinx: false,
          xbins: { start: globalMin, end: globalMax, size: binSize },
          bingroup: 'energy-distribution',
          showlegend: false,
          hovertemplate: `${label}<br>energy: %{x:.3f}<br>density: %{y:.4f}<extra></extra>`,
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
          opacity: 0.14,
          marker: { color, line: { color: hexToTransparent(color, 0.68), width: 0.6 } },
          showlegend: false,
          hovertemplate: `${label}<br>energy: %{x:.3f}<br>density: %{y:.4f}<extra></extra>`,
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
        hovertemplate: `${label}<br>energy: %{x:.3f}<br>density: %{y:.4f}<extra></extra>`,
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
          hovertemplate: `${label}<br>energy: %{x:.3f}<br>density: %{y:.4f}<extra></extra>`,
        });
      }
    }
  });

  const dark = background === 'dark';
  const textColor = dark ? '#e5e7eb' : '#111827';
  const mutedTextColor = dark ? '#d1d5db' : '#374151';
  const gridColor = dark ? '#374151' : '#d1d5db';
  return {
    data: traces,
    layout: {
      title: title ? { text: title, font: { size: 13 } } : undefined,
      height,
      margin: { l: 52, r: 18, t: title ? 34 : 12, b: 48 },
      paper_bgcolor: dark ? 'rgba(0,0,0,0)' : '#ffffff',
      plot_bgcolor: dark ? 'rgba(2,6,23,0.36)' : '#ffffff',
      font: { color: textColor },
      barmode: 'overlay',
      xaxis: {
        title: { text: xTitle, font: { color: textColor } },
        tickfont: { color: mutedTextColor },
        color: textColor,
        gridcolor: dark ? 'rgba(148,163,184,0.14)' : gridColor,
        griddash: 'dot',
        zerolinecolor: gridColor,
        zeroline: true,
        range: [x0, x1],
      },
      yaxis: {
        title: { text: 'Density', font: { color: textColor } },
        tickfont: { color: mutedTextColor },
        color: textColor,
        gridcolor: dark ? 'rgba(148,163,184,0.14)' : gridColor,
        griddash: 'dot',
        zerolinecolor: gridColor,
        rangemode: 'tozero',
      },
      shapes,
      annotations,
      legend: {
        orientation: 'h',
        y: -0.22,
        x: 0.5,
        xanchor: 'center',
        font: { color: textColor },
        bgcolor: dark ? 'rgba(15,23,42,0.72)' : 'rgba(255,255,255,0.88)',
        bordercolor: gridColor,
        borderwidth: 1,
      },
      hoverlabel: { bgcolor: dark ? '#111827' : '#ffffff', bordercolor: gridColor, font: { color: textColor } },
    },
    config: { displayModeBar: false, responsive: true },
  };
}

export default function EnergyDistributionPlot({ plot, height = 300, foreground = 'auto', frameMarker = null }) {
  if (!plot) return null;
  const transparentOrDark = String(plot.layout?.paper_bgcolor || '').toLowerCase() !== '#ffffff';
  const useDarkForeground = foreground === 'dark' || (foreground === 'auto' && !transparentOrDark);
  const markerValue = Number(frameMarker?.value);
  const markerColor = frameMarker?.color || '#f59e0b';
  const markerVisible = Number.isFinite(markerValue);
  const markerShapes = markerVisible ? [{
    type: 'line',
    xref: 'x',
    yref: 'paper',
    x0: markerValue,
    x1: markerValue,
    y0: 0,
    y1: 1,
    line: { color: markerColor, width: 3, dash: 'dash' },
  }] : [];
  const markerAnnotations = markerVisible ? [{
    x: markerValue,
    y: 1,
    xref: 'x',
    yref: 'paper',
    xanchor: 'left',
    yanchor: 'bottom',
    text: frameMarker?.label || `Current frame: ${markerValue.toFixed(3)}`,
    showarrow: false,
    font: { size: 11, color: markerColor },
    bgcolor: transparentOrDark ? 'rgba(17,24,39,0.9)' : 'rgba(255,255,255,0.9)',
    bordercolor: markerColor,
    borderwidth: 1,
    borderpad: 3,
  }] : [];
  return (
    <div className={`energy-distribution-plot ${useDarkForeground ? 'energy-distribution-plot--dark-foreground' : 'energy-distribution-plot--light-foreground'}`}>
      <Plot
        data={plot.data}
        layout={{
          ...plot.layout,
          height,
          shapes: [...(plot.layout?.shapes || []), ...markerShapes],
          annotations: [...(plot.layout?.annotations || []), ...markerAnnotations],
        }}
        config={plot.config || { displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: '100%', height: `${height}px` }}
      />
    </div>
  );
}
