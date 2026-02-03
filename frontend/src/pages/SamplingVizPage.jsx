import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { Info, Maximize2, Trash2, X } from 'lucide-react';
import Plot from 'react-plotly.js';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { DocOverlay } from '../components/system/SystemDetailOverlays';
import { deleteSamplingSample, fetchSamplingSummary, fetchSystem } from '../api/projects';

const palette = ['#22d3ee', '#f97316', '#a855f7', '#10b981', '#f43f5e', '#fde047', '#60a5fa', '#f59e0b'];

function pickColor(idx) {
  return palette[idx % palette.length];
}

function computeAuc(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || !a.length || !b.length) return null;
  let wins = 0;
  let ties = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < b.length; j += 1) {
      if (a[i] > b[j]) wins += 1;
      else if (a[i] === b[j]) ties += 1;
    }
  }
  const total = a.length * b.length;
  if (!total) return null;
  return (wins + 0.5 * ties) / total;
}

function buildEdgeMatrix(n, edges, values) {
  const matrix = Array.from({ length: n }, () => Array.from({ length: n }, () => null));
  edges.forEach((edge, idx) => {
    const [r, s] = edge;
    const value = values[idx];
    if (r == null || s == null) return;
    matrix[r][s] = value;
    matrix[s][r] = value;
  });
  return matrix;
}

function topK(values, labels, k = 10) {
  const pairs = values.map((v, i) => [v, labels[i] ?? String(i), i]);
  pairs.sort((a, b) => b[0] - a[0]);
  return pairs.slice(0, k);
}

function PlotOverlay({ overlay, onClose }) {
  if (!overlay) return null;
  const layout = { ...(overlay.layout || {}), autosize: true };
  if ('height' in layout) delete layout.height;
  if ('width' in layout) delete layout.width;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="w-[95vw] h-[90vh] bg-gray-900 border border-gray-700 rounded-lg shadow-xl flex flex-col">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-sm font-semibold text-gray-200">{overlay.title}</h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close overlay"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="flex-1 min-h-0 p-3">
          <Plot
            data={overlay.data}
            layout={layout}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>
    </div>
  );
}

function PairwiseMacroPanel({
  pairSourceOptions,
  pairSourceA,
  pairSourceB,
  setPairSourceA,
  setPairSourceB,
  pairA,
  pairB,
  residueLabels,
  jsResidue,
  topResidues,
  edges,
  jsEdges,
  topEdges,
  edgeMatrix,
  edgeStrength,
  edgeMatrixHasValues,
  onOpenOverlay,
}) {
  const invalidPair = pairA && pairB && pairA.kind === pairB.kind;
  const residuePlot = {
    data: [
      {
        x: residueLabels,
        y: jsResidue,
        type: 'bar',
        marker: { color: '#22d3ee' },
      },
    ],
    layout: {
      height: 220,
      margin: { l: 40, r: 10, t: 10, b: 40 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { color: '#111827' },
      xaxis: { tickfont: { size: 9 }, color: '#111827' },
      yaxis: { title: 'JS divergence', color: '#111827' },
    },
  };
  const edgeBarcodePlot = {
    data: [
      {
        x: edges.map((edge) => `${residueLabels[edge[0]]}-${residueLabels[edge[1]]}`),
        y: jsEdges,
        type: 'bar',
        marker: { color: '#f97316' },
      },
    ],
    layout: {
      height: 220,
      margin: { l: 40, r: 10, t: 10, b: 90 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { color: '#111827' },
      xaxis: { tickangle: -45, tickfont: { size: 8 }, color: '#111827' },
      yaxis: { title: 'JS2 divergence', color: '#111827' },
    },
  };
  let edgeHeatmapReady = false;
  let edgeHeatmapData = null;
  let edgeHeatmapLayout = {
    height: 260,
    margin: { l: 60, r: 20, t: 10, b: 60 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#111827' },
    xaxis: { tickfont: { size: 7 }, color: '#111827' },
    yaxis: { tickfont: { size: 7 }, color: '#111827' },
  };
  if (edgeMatrixHasValues && Array.isArray(edgeMatrix) && edgeMatrix.length) {
    const z = edgeMatrix.map((row) =>
      Array.isArray(row) ? row.map((val) => (Number.isFinite(val) ? val : null)) : []
    );
    const flat = z.flat().filter((val) => Number.isFinite(val));
    if (flat.length) {
      let zmin = Math.min(...flat);
      let zmax = Math.max(...flat);
      if (!Number.isFinite(zmin) || !Number.isFinite(zmax)) {
        zmin = null;
        zmax = null;
      } else if (zmin === zmax) {
        const pad = Math.abs(zmin) * 0.1 || 1;
        zmin -= pad;
        zmax += pad;
      }
      edgeHeatmapReady = true;
      edgeHeatmapData = [
        {
          z,
          type: 'heatmap',
          colorscale: 'YlOrRd',
          showscale: true,
          zmin: zmin ?? undefined,
          zmax: zmax ?? undefined,
          zauto: zmin == null || zmax == null,
        },
      ];
    }
  }
  const edgeMismatchPlot = {
    data: [
      {
        x: edgeStrength,
        y: jsEdges,
        mode: 'markers',
        type: 'scatter',
        marker: { color: '#a855f7' },
      },
    ],
    layout: {
      height: 220,
      margin: { l: 40, r: 10, t: 10, b: 40 },
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      font: { color: '#111827' },
      xaxis: { title: '|J|', color: '#111827' },
      yaxis: { title: 'JS2', color: '#111827' },
    },
  };

  return (
    <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">Residue + edge diagnostics</h2>
        </div>
        <div className="grid sm:grid-cols-2 gap-3 w-full md:w-auto">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Source A</label>
            <select
              value={pairSourceA}
              onChange={(event) => setPairSourceA(event.target.value)}
              className="w-full bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-white text-sm"
            >
              {pairSourceOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Source B</label>
            <select
              value={pairSourceB}
              onChange={(event) => setPairSourceB(event.target.value)}
              className="w-full bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-white text-sm"
            >
              {pairSourceOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>
      {invalidPair && (
        <p className="text-xs text-amber-300">
          Pairwise plots currently support MD vs Sample comparisons. Pick one MD and one Sample.
        </p>
      )}
      <div className="grid lg:grid-cols-2 gap-4">
        <div className="bg-gray-950/60 border border-gray-800 rounded-md p-3 space-y-3 min-w-0 overflow-hidden">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-200">Residue barcode</h3>
            <button
              type="button"
              onClick={() => onOpenOverlay('Residue barcode', residuePlot.data, residuePlot.layout)}
              className="text-gray-400 hover:text-gray-200"
              aria-label="Open residue barcode overlay"
            >
              <Maximize2 className="h-4 w-4" />
            </button>
          </div>
          <Plot
            data={residuePlot.data}
            layout={residuePlot.layout}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
        <div className="bg-gray-950/60 border border-gray-800 rounded-md p-3 space-y-3 min-w-0 overflow-hidden">
          <h3 className="text-sm font-semibold text-gray-200">Top residues</h3>
          <ul className="grid gap-1 text-xs text-gray-300">
            {topResidues.map(([val, label], idx) => (
              <li key={`${label}-${idx}`} className="flex justify-between">
                <span>{label}</span>
                <span>{val.toFixed(4)}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-gray-950/60 border border-gray-800 rounded-md p-3 space-y-3 min-w-0 overflow-hidden">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-200">Edge barcode</h3>
            <button
              type="button"
              onClick={() => onOpenOverlay('Edge barcode', edgeBarcodePlot.data, edgeBarcodePlot.layout)}
              className="text-gray-400 hover:text-gray-200"
              aria-label="Open edge barcode overlay"
            >
              <Maximize2 className="h-4 w-4" />
            </button>
          </div>
          <Plot
            data={edgeBarcodePlot.data}
            layout={edgeBarcodePlot.layout}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
        <div className="bg-gray-950/60 border border-gray-800 rounded-md p-3 space-y-3 min-w-0 overflow-hidden">
          <h3 className="text-sm font-semibold text-gray-200">Top edges</h3>
          <ul className="grid gap-1 text-xs text-gray-300">
            {topEdges.map(([val, label], idx) => (
              <li key={`${label}-${idx}`} className="flex justify-between">
                <span>{label}</span>
                <span>{val.toFixed(4)}</span>
              </li>
            ))}
          </ul>
        </div>
        <div className="bg-gray-950/60 border border-gray-800 rounded-md p-3 space-y-3 min-w-0 overflow-hidden">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-200">Edge heatmap</h3>
            {edgeHeatmapReady ? (
              <button
                type="button"
                onClick={() => onOpenOverlay('Edge heatmap', edgeHeatmapData, edgeHeatmapLayout)}
                className="text-gray-400 hover:text-gray-200"
                aria-label="Open edge heatmap overlay"
              >
                <Maximize2 className="h-4 w-4" />
              </button>
            ) : null}
          </div>
          {edgeHeatmapReady ? (
            <Plot
              data={edgeHeatmapData}
              layout={edgeHeatmapLayout}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: '100%', height: '100%' }}
            />
          ) : (
            <p className="text-xs text-gray-400">No edge heatmap data available for this selection.</p>
          )}
        </div>
        <div className="bg-gray-950/60 border border-gray-800 rounded-md p-3 space-y-3 min-w-0 overflow-hidden">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-200">Edge mismatch vs strength</h3>
            <button
              type="button"
              onClick={() => onOpenOverlay('Edge mismatch vs strength', edgeMismatchPlot.data, edgeMismatchPlot.layout)}
              className="text-gray-400 hover:text-gray-200"
              aria-label="Open edge mismatch overlay"
            >
              <Maximize2 className="h-4 w-4" />
            </button>
          </div>
          <Plot
            data={edgeMismatchPlot.data}
            layout={edgeMismatchPlot.layout}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </div>
    </section>
  );
}

export default function SamplingVizPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [summaryCache, setSummaryCache] = useState({});
  const [summaryError, setSummaryError] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [deleteBusyId, setDeleteBusyId] = useState(null);
  const [selectedModelIds, setSelectedModelIds] = useState([]);
  const [pairSourceA, setPairSourceA] = useState('');
  const [pairSourceB, setPairSourceB] = useState('');
  const [overlayPlot, setOverlayPlot] = useState(null);
  const [showPlotDoc, setShowPlotDoc] = useState(false);
  const [infoSampleId, setInfoSampleId] = useState('');

  const clusterOptions = useMemo(
    () => (system?.metastable_clusters || []).filter((run) => run.path && run.status !== 'failed'),
    [system]
  );

  const selectedCluster = useMemo(
    () => clusterOptions.find((c) => c.cluster_id === selectedClusterId) || null,
    [clusterOptions, selectedClusterId]
  );

  const sampleEntries = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);
  const mdSamples = useMemo(() => sampleEntries.filter((s) => s.type === 'md_eval'), [sampleEntries]);
  const pottsModels = useMemo(() => selectedCluster?.potts_models || [], [selectedCluster]);
  const filteredSamples = useMemo(() => {
    if (!selectedModelIds.length) return sampleEntries;
    return sampleEntries.filter((s) => {
      const ids = Array.isArray(s.model_ids)
        ? s.model_ids
        : s.model_id
          ? [s.model_id]
          : [];
      if (!ids.length) return true;
      return ids.some((id) => selectedModelIds.includes(id));
    });
  }, [sampleEntries, selectedModelIds]);
  const gibbsSamples = useMemo(
    () => filteredSamples.filter((s) => s.type === 'potts_sampling' && s.method === 'gibbs'),
    [filteredSamples]
  );
  const saSamples = useMemo(
    () => filteredSamples.filter((s) => s.type === 'potts_sampling' && s.method === 'sa'),
    [filteredSamples]
  );

  useEffect(() => {
    const loadSystem = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
      } catch (err) {
        setSystemError(err.message);
      } finally {
        setLoadingSystem(false);
      }
    };
    loadSystem();
  }, [projectId, systemId]);

  useEffect(() => {
    const params = new URLSearchParams(location.search || '');
    const clusterId = params.get('cluster_id');
    if (clusterId) setSelectedClusterId(clusterId);
  }, [location.search]);

  useEffect(() => {
    if (!clusterOptions.length) return;
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterId]);

  useEffect(() => {
    if (!pottsModels.length) {
      if (selectedModelIds.length) {
        setSelectedModelIds([]);
      }
      return;
    }
    if (!selectedModelIds.length) {
      setSelectedModelIds(pottsModels.map((m) => m.model_id));
      return;
    }
    const allowed = new Set(pottsModels.map((m) => m.model_id));
    const filtered = selectedModelIds.filter((id) => allowed.has(id));
    if (filtered.length !== selectedModelIds.length) {
      setSelectedModelIds(filtered);
    }
  }, [pottsModels, selectedModelIds]);

  const loadSummary = useCallback(
    async (sampleId) => {
      if (!sampleId || summaryCache[sampleId]) return;
      setSummaryLoading(true);
      setSummaryError(null);
      try {
        const data = await fetchSamplingSummary(projectId, systemId, selectedClusterId, sampleId);
        setSummaryCache((prev) => ({ ...prev, [sampleId]: data }));
      } catch (err) {
        setSummaryError(err.message);
      } finally {
        setSummaryLoading(false);
      }
    },
    [projectId, systemId, selectedClusterId, summaryCache]
  );

  useEffect(() => {
    const ids = [...gibbsSamples, ...saSamples].map((s) => s.sample_id);
    ids.forEach((id) => loadSummary(id));
  }, [gibbsSamples, saSamples, loadSummary]);

  const stateNameById = useMemo(() => {
    const map = {};
    Object.values(system?.states || {}).forEach((state) => {
      if (state?.state_id) map[state.state_id] = state.name || state.state_id;
    });
    (system?.metastable_states || []).forEach((state) => {
      if (state?.metastable_id) map[state.metastable_id] = state.name || state.metastable_id;
    });
    return map;
  }, [system]);

  const normalizeMdLabel = useCallback(
    (label, idx, mdSourceIds = []) => {
      if (!label) return label;
      const idFromList = mdSourceIds?.[idx];
      if (idFromList && typeof idFromList === 'string') {
        if (idFromList.startsWith('state:')) {
          const stateId = idFromList.split(':', 2)[1];
          return stateNameById[stateId] || label;
        }
        if (idFromList.startsWith('meta:')) {
          const metaId = idFromList.split(':', 2)[1];
          return stateNameById[metaId] || label;
        }
      }
      const cleaned = label.replace(/^Macro:\s*/i, '').replace(/^MD cluster:\s*/i, '').trim();
      if (stateNameById[cleaned]) return stateNameById[cleaned];
      return cleaned || label;
    },
    [stateNameById]
  );

  const samplingSamples = useMemo(() => [...gibbsSamples, ...saSamples], [gibbsSamples, saSamples]);
  const summaryEntries = useMemo(
    () =>
      samplingSamples
        .map((sample) => ({ sample, summary: summaryCache[sample.sample_id] }))
        .filter((entry) => entry.summary),
    [samplingSamples, summaryCache]
  );
  const summaryBySampleId = useMemo(() => {
    const map = {};
    summaryEntries.forEach(({ sample, summary }) => {
      map[sample.sample_id] = summary;
    });
    return map;
  }, [summaryEntries]);

  const infoSample = useMemo(() => {
    if (!infoSampleId) return null;
    return sampleEntries.find((s) => s.sample_id === infoSampleId) || null;
  }, [infoSampleId, sampleEntries]);

  const infoSummary = useMemo(() => {
    if (!infoSample) return null;
    return infoSample.summary || infoSample.params || null;
  }, [infoSample]);

  const toggleInfo = useCallback((sampleId) => {
    setInfoSampleId((prev) => (prev === sampleId ? '' : sampleId));
  }, []);
  const baseSummary = summaryEntries[0]?.summary || null;

  const handleDeleteSample = useCallback(
    async (sampleId) => {
      if (!selectedClusterId || !sampleId) return;
      const ok = window.confirm('Delete this sampling result? This cannot be undone.');
      if (!ok) return;
      try {
        setDeleteBusyId(sampleId);
        await deleteSamplingSample(projectId, systemId, selectedClusterId, sampleId);
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
        setSummaryCache((prev) => {
          const next = { ...prev };
          delete next[sampleId];
          return next;
        });
      } catch (err) {
        setSummaryError(err.message || 'Failed to delete sampling run.');
      } finally {
        setDeleteBusyId(null);
      }
    },
    [projectId, systemId, selectedClusterId]
  );

  const mdSourceLabels = useMemo(() => {
    const labels = baseSummary?.md_source_labels || [];
    const ids = baseSummary?.md_source_ids || [];
    return labels.map((label, idx) => normalizeMdLabel(label, idx, ids));
  }, [baseSummary, normalizeMdLabel]);

  const pairSourceOptions = useMemo(() => {
    const mdOpts = mdSourceLabels.map((label, idx) => ({
      value: `md:${idx}`,
      label: `MD: ${label}`,
      kind: 'md',
      idx,
    }));
    const sampleOpts = [];
    summaryEntries.forEach(({ sample, summary }) => {
      const sampleLabel = sample.name || summary.sample_name || sample.sample_id;
      (summary.sample_source_labels || []).forEach((label, idx) => {
        sampleOpts.push({
          value: `sample:${sample.sample_id}:${idx}`,
          label: `Sample: ${sampleLabel} · ${label}`,
          kind: 'sample',
          idx,
          sampleId: sample.sample_id,
        });
      });
    });
    return [...mdOpts, ...sampleOpts];
  }, [mdSourceLabels, summaryEntries]);

  useEffect(() => {
    if (!pairSourceOptions.length) {
      setPairSourceA('');
      setPairSourceB('');
      return;
    }
    const values = new Set(pairSourceOptions.map((opt) => opt.value));
    const firstMd = pairSourceOptions.find((opt) => opt.kind === 'md');
    const firstSample = pairSourceOptions.find((opt) => opt.kind === 'sample');
    if (!pairSourceA || !values.has(pairSourceA)) {
      setPairSourceA(firstMd ? firstMd.value : pairSourceOptions[0].value);
    }
    if (!pairSourceB || !values.has(pairSourceB)) {
      setPairSourceB(firstSample ? firstSample.value : pairSourceOptions[0].value);
    }
  }, [pairSourceOptions, pairSourceA, pairSourceB]);

  const pairA = pairSourceOptions.find((opt) => opt.value === pairSourceA) || null;
  const pairB = pairSourceOptions.find((opt) => opt.value === pairSourceB) || null;
  const pairSummary = useMemo(() => {
    if (pairA?.kind === 'sample') {
      return summaryBySampleId[pairA.sampleId] || null;
    }
    if (pairB?.kind === 'sample') {
      return summaryBySampleId[pairB.sampleId] || null;
    }
    return baseSummary;
  }, [pairA, pairB, summaryBySampleId, baseSummary]);

  const jsPair = useMemo(() => {
    if (!pairA || !pairB) return null;
    if (pairA.kind === 'md' && pairB.kind === 'sample') {
      const summary = summaryBySampleId[pairB.sampleId];
      return summary?.js_md_sample?.[pairA.idx]?.[pairB.idx] || null;
    }
    if (pairA.kind === 'sample' && pairB.kind === 'md') {
      const summary = summaryBySampleId[pairA.sampleId];
      return summary?.js_md_sample?.[pairB.idx]?.[pairA.idx] || null;
    }
    return null;
  }, [pairA, pairB, summaryBySampleId]);

  const jsPairEdges = useMemo(() => {
    if (!pairA || !pairB) return null;
    if (pairA.kind === 'md' && pairB.kind === 'sample') {
      const summary = summaryBySampleId[pairB.sampleId];
      return summary?.js2_md_sample?.[pairA.idx]?.[pairB.idx] || null;
    }
    if (pairA.kind === 'sample' && pairB.kind === 'md') {
      const summary = summaryBySampleId[pairA.sampleId];
      return summary?.js2_md_sample?.[pairB.idx]?.[pairA.idx] || null;
    }
    return null;
  }, [pairA, pairB, summaryBySampleId]);

  const residueLabels = pairSummary?.residue_labels || [];
  const edges = pairSummary?.edges || [];
  const jsResidue = jsPair || [];
  const jsEdges = jsPairEdges || [];
  const edgeStrength = pairSummary?.edge_strength || [];
  const edgeMatrix = useMemo(
    () => buildEdgeMatrix(residueLabels.length, edges, jsEdges),
    [residueLabels, edges, jsEdges]
  );
  const edgeMatrixHasValues = useMemo(
    () => edgeMatrix?.some((row) => row?.some((val) => Number.isFinite(val))),
    [edgeMatrix]
  );

  const crossLikelihoodTraces = useMemo(() => {
    const traces = [];
    let colorIdx = 0;
    const baseActive = baseSummary?.xlik_delta_active || [];
    if (Array.isArray(baseActive) && baseActive.length) {
      traces.push({
        x: baseActive,
        type: 'histogram',
        name: 'MD (fit)',
        opacity: 0.6,
        marker: { color: pickColor(colorIdx) },
      });
      colorIdx += 1;
    }
    summaryEntries.forEach(({ sample, summary }) => {
      const active = summary.xlik_delta_active || [];
      const labelPrefix = sample.name || summary.sample_name || sample.sample_id;
      if (active.length) {
        traces.push({
          x: active,
          type: 'histogram',
          name: labelPrefix,
          opacity: 0.6,
          marker: { color: pickColor(colorIdx) },
        });
        colorIdx += 1;
      }
    });
    return traces;
  }, [summaryEntries, baseSummary]);

  const crossLikelihoodMatrix = useMemo(() => {
    const entries = [];
    const baseActive = baseSummary?.xlik_delta_active || [];
    if (Array.isArray(baseActive) && baseActive.length) {
      entries.push({ label: 'MD (fit)', values: baseActive });
    }
    summaryEntries.forEach(({ sample, summary }) => {
      const active = summary.xlik_delta_active || [];
      if (Array.isArray(active) && active.length) {
        entries.push({
          label: sample.name || summary.sample_name || sample.sample_id,
          values: active,
        });
      }
    });
    if (!entries.length) {
      return { labels: [], auc: [], delta: [] };
    }
    const labels = entries.map((e) => e.label);
    const auc = labels.map(() => labels.map(() => null));
    const delta = labels.map(() => labels.map(() => null));
    for (let i = 0; i < entries.length; i += 1) {
      for (let j = 0; j < entries.length; j += 1) {
        const a = entries[i].values;
        const b = entries[j].values;
        const aucVal = computeAuc(a, b);
        if (aucVal != null) auc[i][j] = aucVal;
        const meanA = a.length ? a.reduce((sum, v) => sum + v, 0) / a.length : null;
        const meanB = b.length ? b.reduce((sum, v) => sum + v, 0) / b.length : null;
        if (meanA != null && meanB != null) delta[i][j] = meanA - meanB;
      }
    }
    return { labels, auc, delta };
  }, [summaryEntries, baseSummary]);

  const topResidues = useMemo(() => topK(jsResidue, residueLabels, 12), [jsResidue, residueLabels]);
  const topEdges = useMemo(() => {
    const labels = edges.map((edge) => {
      const r = residueLabels[edge[0]] || `res_${edge[0]}`;
      const s = residueLabels[edge[1]] || `res_${edge[1]}`;
      return `${r}-${s}`;
    });
    return topK(jsEdges, labels, 12);
  }, [edges, jsEdges, residueLabels]);

  const openOverlay = useCallback((title, data, layout) => {
    if (!data || !layout) return;
    setOverlayPlot({ title, data, layout });
  }, []);

  const energyTraces = useMemo(() => {
    const traces = [];
    let colorIdx = 0;
    const mdSeen = new Set();
    summaryEntries.forEach(({ sample, summary }) => {
      const bins = summary.energy_bins || [];
      if (!bins.length) return;
      const xVals = bins.slice(0, -1);
      const sampleLabel = sample.name || summary.sample_name || sample.sample_id;
      const modelKey =
        summary.model_id ||
        (Array.isArray(sample.model_ids) ? sample.model_ids.join('+') : sample.model_id) ||
        sample.sample_id;

      if (!mdSeen.has(modelKey)) {
        const mdLabels = summary.md_source_labels || [];
        const mdIds = summary.md_source_ids || [];
        const mdHists = summary.energy_hist_md || [];
        mdHists.forEach((hist, idx) => {
          if (!Array.isArray(hist) || !hist.length) return;
          const mdLabel = normalizeMdLabel(mdLabels[idx] || `MD ${idx + 1}`, idx, mdIds);
          traces.push({
            x: xVals,
            y: hist,
            type: 'scatter',
            mode: 'lines',
            name: `MD · ${mdLabel}`,
            line: { color: pickColor(colorIdx) },
          });
          colorIdx += 1;
        });
        mdSeen.add(modelKey);
      }

      const sampleLabels = summary.sample_source_labels || [];
      const sampleHists = summary.energy_hist_sample || [];
      sampleHists.forEach((hist, idx) => {
        if (!Array.isArray(hist) || !hist.length) return;
        const label = sampleLabels[idx] || `Sample ${idx + 1}`;
        traces.push({
          x: xVals,
          y: hist,
          type: 'scatter',
          mode: 'lines',
          name: `${sampleLabel} · ${label}`,
          line: { color: pickColor(colorIdx) },
        });
        colorIdx += 1;
      });
    });
    return traces;
  }, [summaryEntries, normalizeMdLabel]);

  const betaEffScanTraces = useMemo(() => {
    const traces = [];
    summaryEntries.forEach(({ sample, summary }, idx) => {
      const grid = summary.beta_eff_grid || [];
      const scheduleLabels = summary.sa_schedule_labels || [];
      const sampleLabel = sample.name || summary.sample_name || sample.sample_id;
      let rows = [];
      const bySchedule = summary.beta_eff_distances_by_schedule;
      if (Array.isArray(bySchedule) && bySchedule.length) {
        rows = Array.isArray(bySchedule[0]) ? bySchedule : [bySchedule];
      }
      if (!rows.length) {
        const flat = summary.beta_eff_distances;
        if (Array.isArray(flat) && flat.length) {
          rows = [flat];
        }
      }
      if (!grid.length || !rows.length) return;
      rows.forEach((row, rowIdx) => {
        if (!Array.isArray(row) || !row.length) return;
        traces.push({
          x: grid,
          y: row,
          type: 'scatter',
          mode: 'lines+markers',
          name: `${sampleLabel} · ${scheduleLabels[rowIdx] || `Schedule ${rowIdx + 1}`}`,
          line: { color: pickColor(idx) },
        });
      });
    });
    return traces;
  }, [summaryEntries]);

  const betaEffMarkers = useMemo(() => {
    const traces = [];
    summaryEntries.forEach(({ sample, summary }, summaryIdx) => {
      const grid = summary.beta_eff_grid || [];
      const scheduleLabels = summary.sa_schedule_labels || [];
      const sampleLabel = sample.name || summary.sample_name || sample.sample_id;
      let rows = [];
      const bySchedule = summary.beta_eff_distances_by_schedule;
      if (Array.isArray(bySchedule) && bySchedule.length) {
        rows = Array.isArray(bySchedule[0]) ? bySchedule : [bySchedule];
      }
      if (!rows.length) {
        const flat = summary.beta_eff_distances;
        if (Array.isArray(flat) && flat.length) {
          rows = [flat];
        }
      }
      if (grid.length && rows.length) {
        rows.forEach((row, rowIdx) => {
          if (!Array.isArray(row) || !row.length) return;
          let minIdx = 0;
          let minVal = row[0];
          for (let i = 1; i < row.length; i += 1) {
            if (row[i] < minVal) {
              minVal = row[i];
              minIdx = i;
            }
          }
          const betaEff = grid[minIdx];
          if (betaEff === undefined || betaEff === null || Number.isNaN(Number(betaEff))) return;
          traces.push({
            x: [Number(betaEff)],
            y: [Number(minVal)],
            type: 'scatter',
            mode: 'markers',
            name: `${sampleLabel} · ${scheduleLabels[rowIdx] || `Schedule ${rowIdx + 1}`} (β_eff)`,
            marker: { color: pickColor(summaryIdx), size: 10, symbol: 'circle' },
          });
        });
        return;
      }

      const betaEffBySchedule = summary.beta_eff_by_schedule || [];
      if (Array.isArray(betaEffBySchedule) && betaEffBySchedule.length) {
        betaEffBySchedule.forEach((betaEff, rowIdx) => {
          if (betaEff === undefined || betaEff === null || Number.isNaN(Number(betaEff))) return;
          traces.push({
            x: [Number(betaEff)],
            y: [0],
            type: 'scatter',
            mode: 'markers',
            name: `${sampleLabel} · ${scheduleLabels[rowIdx] || `Schedule ${rowIdx + 1}`} (β_eff)`,
            marker: { color: pickColor(summaryIdx), size: 10, symbol: 'circle' },
          });
        });
        return;
      }

      const betaEff = summary.beta_eff;
      const betaValue = Array.isArray(betaEff) ? betaEff[0] : betaEff;
      if (betaValue !== undefined && betaValue !== null && !Number.isNaN(Number(betaValue))) {
        traces.push({
          x: [Number(betaValue)],
          y: [0],
          type: 'scatter',
          mode: 'markers',
          name: `${sampleLabel} (β_eff)`,
          marker: { color: pickColor(summaryIdx), size: 10, symbol: 'circle' },
        });
      }
    });
    return traces;
  }, [summaryEntries]);

  if (loadingSystem) {
    return <Loader label="Loading sampling explorer..." />;
  }
  if (systemError) {
    return <ErrorMessage message={systemError} />;
  }

  const energyPlotLayout = {
    height: 280,
    margin: { l: 40, r: 10, t: 10, b: 40 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#111827' },
    xaxis: { title: 'Energy', color: '#111827' },
    yaxis: { title: 'Density', color: '#111827' },
  };
  const crossLikelihoodLayout = {
    height: 260,
    margin: { l: 40, r: 10, t: 10, b: 40 },
    barmode: 'overlay',
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#111827' },
    xaxis: { title: 'Δ log-likelihood', color: '#111827' },
  };
  const crossLikelihoodMatrixLayout = {
    height: 200,
    margin: { l: 70, r: 20, t: 20, b: 70 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#111827', size: 10 },
    xaxis: { tickfont: { size: 9 }, color: '#111827' },
    yaxis: { tickfont: { size: 9 }, color: '#111827' },
  };
  const betaEffLayout = {
    height: 260,
    margin: { l: 40, r: 10, t: 10, b: 40 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: { color: '#111827' },
    xaxis: { title: 'beta_eff', color: '#111827' },
    yaxis: { title: 'distance', color: '#111827' },
  };

  return (
    <div className="p-6 space-y-6 text-white">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Sampling Explorer</h1>
          <p className="text-sm text-gray-400 mt-1">Interactive sampling diagnostics for Gibbs and SA runs.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowPlotDoc(true)}
            className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/40"
            aria-label="Open plotting documentation"
          >
            <Info className="h-4 w-4" />
            Plot guide
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/40"
          >
            Back to system
          </button>
        </div>
      </div>

      <div className="grid md:grid-cols-[260px_1fr] gap-6">
        <aside className="space-y-4">
          <div>
            <label className="block text-sm text-gray-300 mb-1">Cluster</label>
            <select
              value={selectedClusterId}
              onChange={(event) => setSelectedClusterId(event.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
            >
              {clusterOptions.map((run) => (
                <option key={run.cluster_id} value={run.cluster_id}>
                  {run.name || run.cluster_id}
                </option>
              ))}
            </select>
          </div>

          <div>
            <p className="text-xs font-semibold text-gray-300">Potts models</p>
            <div className="space-y-1 mt-2 text-xs text-gray-300">
              {pottsModels.length === 0 && <p className="text-gray-500">None</p>}
              {pottsModels.map((model) => {
                const checked = selectedModelIds.includes(model.model_id);
                const label = model.name || model.path?.split('/').pop() || model.model_id;
                return (
                  <label key={model.model_id} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() =>
                        setSelectedModelIds((prev) =>
                          prev.includes(model.model_id)
                            ? prev.filter((id) => id !== model.model_id)
                            : [...prev, model.model_id]
                        )
                      }
                    />
                    <span className="truncate">{label}</span>
                  </label>
                );
              })}
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-gray-300">MD samples</p>
            <div className="space-y-1 mt-2 text-xs text-gray-300">
              {mdSamples.length === 0 && <p className="text-gray-500">None</p>}
              {mdSamples.map((s) => (
                <div key={s.sample_id} className="flex items-center justify-between gap-2">
                  <span className="truncate">{s.name || s.sample_id}</span>
                  <button
                    type="button"
                    onClick={() => toggleInfo(s.sample_id)}
                    className="text-gray-400 hover:text-gray-200"
                    aria-label={`Show info for ${s.name || s.sample_id}`}
                  >
                    ℹ
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-gray-300">Gibbs samples</p>
            <div className="space-y-1 mt-2 text-xs text-gray-300">
              {gibbsSamples.length === 0 && <p className="text-gray-500">None</p>}
              {gibbsSamples.map((s) => (
                <div key={s.sample_id} className="flex items-center justify-between gap-2">
                  <span className="truncate">{s.name || s.sample_id}</span>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => toggleInfo(s.sample_id)}
                      className="text-gray-400 hover:text-gray-200"
                      aria-label={`Show info for ${s.name || s.sample_id}`}
                    >
                      ℹ
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDeleteSample(s.sample_id)}
                      className="text-gray-400 hover:text-red-300"
                      aria-label={`Delete ${s.name || s.sample_id}`}
                      disabled={deleteBusyId === s.sample_id}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <p className="text-xs font-semibold text-gray-300">SA samples</p>
            <div className="space-y-1 mt-2 text-xs text-gray-300">
              {saSamples.length === 0 && <p className="text-gray-500">None</p>}
              {saSamples.map((s) => (
                <div key={s.sample_id} className="flex items-center justify-between gap-2">
                  <span className="truncate">{s.name || s.sample_id}</span>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => toggleInfo(s.sample_id)}
                      className="text-gray-400 hover:text-gray-200"
                      aria-label={`Show info for ${s.name || s.sample_id}`}
                    >
                      ℹ
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDeleteSample(s.sample_id)}
                      className="text-gray-400 hover:text-red-300"
                      aria-label={`Delete ${s.name || s.sample_id}`}
                      disabled={deleteBusyId === s.sample_id}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {infoSample && (
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-3 text-xs text-gray-300 space-y-2">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="text-sm font-semibold text-white">{infoSample.name || infoSample.sample_id}</p>
                  <p className="text-[11px] text-gray-400">Sample info</p>
                </div>
                <button
                  type="button"
                  onClick={() => setInfoSampleId('')}
                  className="text-gray-400 hover:text-gray-200"
                  aria-label="Close sample info"
                >
                  ✕
                </button>
              </div>
              <div className="space-y-1">
                <div><span className="text-gray-400">id:</span> {infoSample.sample_id}</div>
                {infoSample.created_at && <div><span className="text-gray-400">created:</span> {infoSample.created_at}</div>}
                {infoSample.method && <div><span className="text-gray-400">method:</span> {infoSample.method}</div>}
                {infoSample.model_names && infoSample.model_names.length > 0 && (
                  <div><span className="text-gray-400">models:</span> {infoSample.model_names.join(', ')}</div>
                )}
                {infoSample.model_id && !infoSample.model_names && (
                  <div><span className="text-gray-400">model id:</span> {infoSample.model_id}</div>
                )}
                {infoSample.path && <div><span className="text-gray-400">path:</span> {infoSample.path}</div>}
              </div>
              {infoSummary && (
                <details className="text-xs text-gray-300">
                  <summary className="cursor-pointer text-gray-200">Run details</summary>
                  <pre className="mt-2 max-h-64 overflow-auto rounded bg-gray-950 p-2 text-[11px] text-gray-300">
                    {JSON.stringify(infoSummary, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          )}
        </aside>

        <main className="space-y-6">

          {summaryError && <ErrorMessage message={summaryError} />}
          {summaryLoading && <Loader label="Loading summary..." />}

          {summaryEntries.length > 0 && (
            <>
              <PairwiseMacroPanel
                pairSourceOptions={pairSourceOptions}
                pairSourceA={pairSourceA}
                pairSourceB={pairSourceB}
                setPairSourceA={setPairSourceA}
                setPairSourceB={setPairSourceB}
                pairA={pairA}
                pairB={pairB}
                residueLabels={residueLabels}
                jsResidue={jsResidue}
                topResidues={topResidues}
                edges={edges}
                jsEdges={jsEdges}
                topEdges={topEdges}
                edgeMatrix={edgeMatrix}
                edgeStrength={edgeStrength}
                edgeMatrixHasValues={edgeMatrixHasValues}
                onOpenOverlay={openOverlay}
              />

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold">Energy histogram</h2>
                  <button
                    type="button"
                    onClick={() => openOverlay('Energy histogram', energyTraces, energyPlotLayout)}
                    className="text-gray-400 hover:text-gray-200"
                    aria-label="Open energy histogram overlay"
                  >
                    <Maximize2 className="h-4 w-4" />
                  </button>
                </div>
                <Plot
                  data={energyTraces}
                  layout={energyPlotLayout}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '100%' }}
                />
              </section>

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold">Cross-likelihood classification</h2>
                  <button
                    type="button"
                    onClick={() =>
                      openOverlay('Cross-likelihood classification', crossLikelihoodTraces, crossLikelihoodLayout)
                    }
                    className="text-gray-400 hover:text-gray-200"
                    aria-label="Open cross-likelihood overlay"
                  >
                    <Maximize2 className="h-4 w-4" />
                  </button>
                </div>
                <div className="grid lg:grid-cols-[1.2fr_1fr] gap-4">
                  <Plot
                    data={crossLikelihoodTraces}
                    layout={crossLikelihoodLayout}
                    config={{ displayModeBar: false, responsive: true }}
                    useResizeHandler
                    style={{ width: '100%', height: '100%' }}
                  />
                  <div className="grid gap-3">
                    <div className="bg-white rounded-md border border-gray-200 p-2">
                      <h4 className="text-xs font-semibold text-gray-700 mb-2">AUC separability</h4>
                      <Plot
                        data={[
                          {
                            z: crossLikelihoodMatrix.auc,
                            x: crossLikelihoodMatrix.labels,
                            y: crossLikelihoodMatrix.labels,
                            type: 'heatmap',
                            colorscale: 'Viridis',
                            zmin: 0,
                            zmax: 1,
                            showscale: true,
                          },
                        ]}
                        layout={crossLikelihoodMatrixLayout}
                        config={{ displayModeBar: false, responsive: true }}
                        useResizeHandler
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                    <div className="bg-white rounded-md border border-gray-200 p-2">
                      <h4 className="text-xs font-semibold text-gray-700 mb-2">Δ mean separability</h4>
                      <Plot
                        data={[
                          {
                            z: crossLikelihoodMatrix.delta,
                            x: crossLikelihoodMatrix.labels,
                            y: crossLikelihoodMatrix.labels,
                            type: 'heatmap',
                            colorscale: 'RdBu',
                            zmid: 0,
                            showscale: true,
                          },
                        ]}
                        layout={crossLikelihoodMatrixLayout}
                        config={{ displayModeBar: false, responsive: true }}
                        useResizeHandler
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                  </div>
                </div>
              </section>

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold">Beta-eff scan</h2>
                  <button
                    type="button"
                    onClick={() =>
                      openOverlay('Beta-eff scan', [...betaEffScanTraces, ...betaEffMarkers], betaEffLayout)
                    }
                    className="text-gray-400 hover:text-gray-200"
                    aria-label="Open beta-eff overlay"
                  >
                    <Maximize2 className="h-4 w-4" />
                  </button>
                </div>
                <Plot
                  data={[...betaEffScanTraces, ...betaEffMarkers]}
                  layout={betaEffLayout}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '100%' }}
                />
              </section>
            </>
          )}
        </main>
      </div>
      <PlotOverlay overlay={overlayPlot} onClose={() => setOverlayPlot(null)} />
      {showPlotDoc && (
        <DocOverlay docId="plotting" onClose={() => setShowPlotDoc(false)} onNavigate={() => {}} />
      )}
    </div>
  );
}
