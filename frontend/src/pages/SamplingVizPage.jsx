import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { Trash2 } from 'lucide-react';
import Plot from 'react-plotly.js';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { deleteSamplingSample, fetchSamplingSummary, fetchSystem } from '../api/projects';

const palette = ['#22d3ee', '#f97316', '#a855f7', '#10b981', '#f43f5e', '#fde047', '#60a5fa', '#f59e0b'];

function pickColor(idx) {
  return palette[idx % palette.length];
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

export default function SamplingVizPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [selectedSampleId, setSelectedSampleId] = useState('');
  const [summaryCache, setSummaryCache] = useState({});
  const [summaryError, setSummaryError] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [deleteBusyId, setDeleteBusyId] = useState(null);
  const [selectedModelIds, setSelectedModelIds] = useState([]);
  const [pairSourceA, setPairSourceA] = useState('');
  const [pairSourceB, setPairSourceB] = useState('');

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
    return sampleEntries.filter((s) => !s.model_id || selectedModelIds.includes(s.model_id));
  }, [sampleEntries, selectedModelIds]);
  const gibbsSamples = useMemo(
    () => filteredSamples.filter((s) => s.type === 'potts_sampling' && s.method === 'gibbs'),
    [filteredSamples]
  );
  const saSamples = useMemo(
    () => filteredSamples.filter((s) => s.type === 'potts_sampling' && s.method === 'sa'),
    [filteredSamples]
  );

  const activeSummary = useMemo(() => summaryCache[selectedSampleId] || null, [summaryCache, selectedSampleId]);

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
    const sampleId = params.get('sample_id');
    if (clusterId) setSelectedClusterId(clusterId);
    if (sampleId) setSelectedSampleId(sampleId);
  }, [location.search]);

  useEffect(() => {
    if (!clusterOptions.length) return;
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterId]);

  useEffect(() => {
    if (!pottsModels.length) {
      setSelectedModelIds([]);
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

  useEffect(() => {
    const candidates = [...gibbsSamples, ...saSamples];
    if (!candidates.length) {
      setSelectedSampleId('');
      return;
    }
    if (!selectedSampleId || !candidates.some((s) => s.sample_id === selectedSampleId)) {
      setSelectedSampleId(candidates[0].sample_id);
    }
  }, [gibbsSamples, saSamples, selectedSampleId]);

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
    if (selectedSampleId) {
      loadSummary(selectedSampleId);
    }
  }, [selectedSampleId, loadSummary]);

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
    (label, idx) => {
      if (!label) return label;
      const idFromList = activeSummary?.md_source_ids?.[idx];
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
    [activeSummary, stateNameById]
  );

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
        if (selectedSampleId === sampleId) {
          setSelectedSampleId('');
        }
      } catch (err) {
        setSummaryError(err.message || 'Failed to delete sampling run.');
      } finally {
        setDeleteBusyId(null);
      }
    },
    [projectId, systemId, selectedClusterId, selectedSampleId]
  );

  const residueLabels = activeSummary?.residue_labels || [];
  const mdSourceLabels = useMemo(
    () => (activeSummary?.md_source_labels || []).map((label, idx) => normalizeMdLabel(label, idx)),
    [activeSummary, normalizeMdLabel]
  );
  const sampleSourceLabels = useMemo(
    () => activeSummary?.sample_source_labels || [],
    [activeSummary]
  );

  const pairSourceOptions = useMemo(() => {
    const mdOpts = mdSourceLabels.map((label, idx) => ({
      value: `md:${idx}`,
      label: `MD: ${label}`,
      kind: 'md',
      idx,
    }));
    const sampleOpts = sampleSourceLabels.map((label, idx) => ({
      value: `sample:${idx}`,
      label: `Sample: ${label}`,
      kind: 'sample',
      idx,
    }));
    return [...mdOpts, ...sampleOpts];
  }, [mdSourceLabels, sampleSourceLabels]);

  useEffect(() => {
    if (!pairSourceOptions.length) {
      setPairSourceA('');
      setPairSourceB('');
      return;
    }
    if (!pairSourceA) {
      const firstMd = pairSourceOptions.find((opt) => opt.kind === 'md');
      setPairSourceA(firstMd ? firstMd.value : pairSourceOptions[0].value);
    }
    if (!pairSourceB) {
      const firstSample = pairSourceOptions.find((opt) => opt.kind === 'sample');
      setPairSourceB(firstSample ? firstSample.value : pairSourceOptions[0].value);
    }
  }, [pairSourceOptions, pairSourceA, pairSourceB]);

  const pairA = pairSourceOptions.find((opt) => opt.value === pairSourceA) || null;
  const pairB = pairSourceOptions.find((opt) => opt.value === pairSourceB) || null;

  const jsPair = useMemo(() => {
    if (!activeSummary || !pairA || !pairB) return null;
    if (pairA.kind === 'md' && pairB.kind === 'sample') {
      return activeSummary.js_md_sample?.[pairA.idx]?.[pairB.idx] || null;
    }
    if (pairA.kind === 'sample' && pairB.kind === 'md') {
      return activeSummary.js_md_sample?.[pairB.idx]?.[pairA.idx] || null;
    }
    return null;
  }, [activeSummary, pairA, pairB]);

  const jsPairEdges = useMemo(() => {
    if (!activeSummary || !pairA || !pairB) return null;
    if (pairA.kind === 'md' && pairB.kind === 'sample') {
      return activeSummary.js2_md_sample?.[pairA.idx]?.[pairB.idx] || null;
    }
    if (pairA.kind === 'sample' && pairB.kind === 'md') {
      return activeSummary.js2_md_sample?.[pairB.idx]?.[pairA.idx] || null;
    }
    return null;
  }, [activeSummary, pairA, pairB]);
  const edges = activeSummary?.edges || [];
  const jsResidue = jsPair || [];
  const jsEdges = jsPairEdges || [];
  const edgeStrength = activeSummary?.edge_strength || [];

  const crossLikelihoodTraces = useMemo(() => {
    const traces = [];
    const samples = [...gibbsSamples, ...saSamples];
    samples.forEach((sample, idx) => {
      const summary = summaryCache[sample.sample_id];
      if (!summary) return;
      const active = summary.xlik_delta_active || [];
      const inactive = summary.xlik_delta_inactive || [];
      if (active.length) {
        traces.push({
          x: active,
          type: 'histogram',
          name: `${summary.sample_name || sample.sample_id} (fit)`,
          opacity: 0.55,
          marker: { color: pickColor(idx) },
        });
      }
      if (inactive.length) {
        traces.push({
          x: inactive,
          type: 'histogram',
          name: `${summary.sample_name || sample.sample_id} (other)`,
          opacity: 0.35,
          marker: { color: pickColor(idx) },
        });
      }
    });
    return traces;
  }, [gibbsSamples, saSamples, summaryCache]);

  const topResidues = useMemo(() => topK(jsResidue, residueLabels, 12), [jsResidue, residueLabels]);
  const topEdges = useMemo(() => {
    const labels = edges.map((edge) => {
      const r = residueLabels[edge[0]] || `res_${edge[0]}`;
      const s = residueLabels[edge[1]] || `res_${edge[1]}`;
      return `${r}-${s}`;
    });
    return topK(jsEdges, labels, 12);
  }, [edges, jsEdges, residueLabels]);

  const energyTraces = useMemo(() => {
    const traces = [];
    const samples = [...gibbsSamples, ...saSamples];
    samples.forEach((sample, idx) => {
      const summary = summaryCache[sample.sample_id];
      if (!summary) return;
      const hist = summary.energy_hist_sample?.[0];
      if (!hist || !summary.energy_bins?.length) return;
      traces.push({
        x: summary.energy_bins.slice(0, -1),
        y: hist,
        type: 'scatter',
        mode: 'lines',
        name: summary.sample_name || sample.sample_id,
        line: { color: pickColor(idx) },
      });
    });
    return traces;
  }, [gibbsSamples, saSamples, summaryCache]);

  const betaEffTraces = useMemo(() => {
    const traces = [];
    const samples = [...gibbsSamples, ...saSamples];
    samples.forEach((sample, idx) => {
      const summary = summaryCache[sample.sample_id];
      if (!summary) return;
      const betaEff = summary.beta_eff;
      let value = null;
      if (Array.isArray(betaEff)) {
        value = betaEff.length ? betaEff[0] : null;
      } else if (betaEff !== undefined && betaEff !== null) {
        value = betaEff;
      }
      if (value === null || Number.isNaN(Number(value))) return;
      traces.push({
        x: [Number(value)],
        y: [0],
        type: 'scatter',
        mode: 'markers',
        name: summary.sample_name || sample.sample_id,
        marker: { color: pickColor(idx), size: 10 },
      });
    });
    return traces;
  }, [gibbsSamples, saSamples, summaryCache]);

  if (loadingSystem) {
    return <Loader label="Loading sampling explorer..." />;
  }
  if (systemError) {
    return <ErrorMessage message={systemError} />;
  }

  return (
    <div className="p-6 space-y-6 text-white">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Sampling Explorer</h1>
          <p className="text-sm text-gray-400 mt-1">Interactive sampling diagnostics for Gibbs and SA runs.</p>
        </div>
        <button
          type="button"
          onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)}
          className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/40"
        >
          Back to system
        </button>
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
                <div key={s.sample_id} className="truncate">{s.name || s.sample_id}</div>
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
              ))}
            </div>
          </div>
        </aside>

        <main className="space-y-6">

          {summaryError && <ErrorMessage message={summaryError} />}
          {summaryLoading && <Loader label="Loading summary..." />}

          {activeSummary && (
            <>
              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-3">
                <h2 className="text-sm font-semibold text-gray-200">Pairwise comparison</h2>
                <div className="grid md:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Summary sample</label>
                    <select
                      value={selectedSampleId}
                      onChange={(event) => setSelectedSampleId(event.target.value)}
                      className="w-full bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-white text-sm"
                    >
                      {[...gibbsSamples, ...saSamples].map((s) => (
                        <option key={s.sample_id} value={s.sample_id}>
                          {s.name || s.sample_id}
                        </option>
                      ))}
                    </select>
                  </div>
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
                {pairA && pairB && pairA.kind === pairB.kind && (
                  <p className="text-xs text-amber-300">
                    Pairwise plots currently support MD vs Sample comparisons. Pick one MD and one Sample.
                  </p>
                )}
              </section>

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <h2 className="text-lg font-semibold">Residue barcode + Top residues</h2>
                <Plot
                  data={[
                    {
                      x: residueLabels,
                      y: jsResidue,
                      type: 'bar',
                      marker: { color: '#22d3ee' },
                    },
                  ]}
                  layout={{
                    height: 260,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { tickfont: { size: 9 }, color: '#111827' },
                    yaxis: { title: 'JS divergence', color: '#111827' },
                  }}
                  config={{ displayModeBar: false }}
                />
                <div className="text-xs text-gray-300">
                  <p className="font-semibold mb-1">Top residues</p>
                  <ul className="grid md:grid-cols-2 gap-1">
                    {topResidues.map(([val, label], idx) => (
                      <li key={`${label}-${idx}`} className="flex justify-between">
                        <span>{label}</span>
                        <span>{val.toFixed(4)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </section>

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <h2 className="text-lg font-semibold">Edge barcode + Top edges</h2>
                <Plot
                  data={[
                    {
                      x: edges.map((edge) => `${residueLabels[edge[0]]}-${residueLabels[edge[1]]}`),
                      y: jsEdges,
                      type: 'bar',
                      marker: { color: '#f97316' },
                    },
                  ]}
                  layout={{
                    height: 260,
                    margin: { l: 40, r: 10, t: 10, b: 120 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { tickangle: -45, tickfont: { size: 8 }, color: '#111827' },
                    yaxis: { title: 'JS2 divergence', color: '#111827' },
                  }}
                  config={{ displayModeBar: false }}
                />
                <div className="text-xs text-gray-300">
                  <p className="font-semibold mb-1">Top edges</p>
                  <ul className="grid md:grid-cols-2 gap-1">
                    {topEdges.map(([val, label], idx) => (
                      <li key={`${label}-${idx}`} className="flex justify-between">
                        <span>{label}</span>
                        <span>{val.toFixed(4)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </section>

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <h2 className="text-lg font-semibold">Edge heatmap</h2>
                <Plot
                  data={[
                    {
                      z: buildEdgeMatrix(residueLabels.length, edges, jsEdges),
                      type: 'heatmap',
                      colorscale: 'YlOrRd',
                      showscale: true,
                    },
                  ]}
                  layout={{
                    height: 360,
                    margin: { l: 60, r: 20, t: 10, b: 60 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { tickfont: { size: 7 }, color: '#111827' },
                    yaxis: { tickfont: { size: 7 }, color: '#111827' },
                  }}
                  config={{ displayModeBar: false }}
                />
              </section>

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <h2 className="text-lg font-semibold">Edge mismatch vs strength</h2>
                <Plot
                  data={[
                    {
                      x: edgeStrength,
                      y: jsEdges,
                      mode: 'markers',
                      type: 'scatter',
                      marker: { color: '#a855f7' },
                    },
                  ]}
                  layout={{
                    height: 260,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { title: '|J|', color: '#111827' },
                    yaxis: { title: 'JS2', color: '#111827' },
                  }}
                  config={{ displayModeBar: false }}
                />
              </section>

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <h2 className="text-lg font-semibold">Energy histogram</h2>
                <Plot
                  data={energyTraces}
                  layout={{
                    height: 280,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { title: 'Energy', color: '#111827' },
                    yaxis: { title: 'Density', color: '#111827' },
                  }}
                  config={{ displayModeBar: false }}
                />
              </section>

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <h2 className="text-lg font-semibold">Cross-likelihood classification</h2>
                <Plot
                  data={crossLikelihoodTraces}
                  layout={{
                    height: 260,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    barmode: 'overlay',
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { title: 'Δ log-likelihood', color: '#111827' },
                  }}
                  config={{ displayModeBar: false }}
                />
              </section>

              <section className="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-4">
                <h2 className="text-lg font-semibold">Beta-eff scan</h2>
                <Plot
                  data={[
                    ...(activeSummary.beta_eff_distances_by_schedule || []).map((row, idx) => ({
                      x: activeSummary.beta_eff_grid || [],
                      y: row,
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: activeSummary.sa_schedule_labels?.[idx] || `Schedule ${idx + 1}`,
                    })),
                    ...betaEffTraces,
                  ]}
                  layout={{
                    height: 260,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { title: 'beta_eff', color: '#111827' },
                    yaxis: { title: 'distance', color: '#111827' },
                  }}
                  config={{ displayModeBar: false }}
                />
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
