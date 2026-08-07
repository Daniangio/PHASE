import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { CircleHelp, RefreshCw } from 'lucide-react';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitHamiltonianSpectralJob } from '../api/jobs';

function safeArray(x) {
  return Array.isArray(x) ? x : [];
}

function scalarString(x) {
  if (Array.isArray(x)) return String(x[0] ?? '');
  return String(x ?? '');
}

function fmt(x, digits = 4) {
  const v = Number(x);
  return Number.isFinite(v) ? v.toFixed(digits) : 'n/a';
}

function analysisTitle(meta) {
  if (!meta) return '';
  if (meta.mode === 'pair') return `${meta.state_a_name || meta.state_a_id} → ${meta.state_b_name || meta.state_b_id}`;
  return `${meta.state_name || meta.state_id}`;
}

function buildTopEdges(matrix, labels, limit = 30) {
  const rows = safeArray(matrix);
  const out = [];
  for (let i = 0; i < rows.length; i += 1) {
    const row = safeArray(rows[i]);
    for (let j = i + 1; j < row.length; j += 1) {
      const value = Number(row[j]);
      if (!Number.isFinite(value) || value === 0) continue;
      out.push({ i, j, a: labels[i] || `res_${i + 1}`, b: labels[j] || `res_${j + 1}`, value, abs: Math.abs(value) });
    }
  }
  out.sort((a, b) => b.abs - a.abs);
  return out.slice(0, limit);
}

function orderMatrix(matrix, order) {
  const rows = safeArray(matrix);
  const ord = safeArray(order).map(Number).filter((x) => Number.isInteger(x) && x >= 0 && x < rows.length);
  if (!ord.length) return rows;
  return ord.map((i) => ord.map((j) => Number(rows[i]?.[j] ?? 0)));
}

function orderLabels(labels, order) {
  const ord = safeArray(order).map(Number).filter((x) => Number.isInteger(x) && x >= 0 && x < labels.length);
  if (!ord.length) return labels;
  return ord.map((i) => labels[i] || `res_${i + 1}`);
}

export default function HamiltonianSpectralPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);
  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [mode, setMode] = useState('single');
  const [spectralView, setSpectralView] = useState('laplacian');
  const [analyses, setAnalyses] = useState([]);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [loadingData, setLoadingData] = useState(false);
  const [dataError, setDataError] = useState(null);
  const [runPanelOpen, setRunPanelOpen] = useState(false);
  const [selectedStateIds, setSelectedStateIds] = useState([]);
  const [topK, setTopK] = useState(20);
  const [overwrite, setOverwrite] = useState(false);
  const [componentIndex, setComponentIndex] = useState(0);
  const [edgeLimit, setEdgeLimit] = useState(30);
  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        const payload = await fetchSystem(projectId, systemId);
        if (!cancelled) setSystem(payload);
      } catch (err) {
        if (!cancelled) setSystemError(err.message || 'Failed to load system.');
      } finally {
        if (!cancelled) setLoadingSystem(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [projectId, systemId]);

  const clusters = useMemo(() => safeArray(system?.metastable_clusters).filter((c) => c.cluster_id), [system]);
  const selectedCluster = useMemo(() => clusters.find((c) => c.cluster_id === selectedClusterId) || null, [clusters, selectedClusterId]);
  const stateOptions = useMemo(() => {
    const rawStates = system?.states;
    let rows = [];
    if (Array.isArray(rawStates)) {
      rows = rawStates.map((s) => ({ state_id: s.state_id || s.id, name: s.name || s.state_id || s.id }));
    } else if (rawStates && typeof rawStates === 'object') {
      rows = Object.entries(rawStates).map(([id, s]) => ({ state_id: id, name: s?.name || id }));
    }
    if (!rows.length && selectedCluster) {
      rows = safeArray(selectedCluster.state_ids || selectedCluster.metastable_ids).map((id) => ({ state_id: String(id), name: String(id) }));
    }
    const seen = new Set();
    return rows.filter((s) => s.state_id && !seen.has(String(s.state_id)) && seen.add(String(s.state_id)));
  }, [system, selectedCluster]);

  useEffect(() => {
    if (!clusters.length) return;
    const qs = new URLSearchParams(location.search || '');
    const requested = String(qs.get('cluster_id') || '').trim();
    if (requested && clusters.some((c) => c.cluster_id === requested)) {
      setSelectedClusterId(requested);
      return;
    }
    if (!selectedClusterId || !clusters.some((c) => c.cluster_id === selectedClusterId)) setSelectedClusterId(clusters[0].cluster_id);
  }, [clusters, selectedClusterId, location.search]);

  useEffect(() => {
    if (!selectedStateIds.length && stateOptions.length) {
      setSelectedStateIds([String(stateOptions[0].state_id)]);
    }
  }, [stateOptions, selectedStateIds.length]);

  const activeAnalysisType = mode === 'pair' ? 'hamiltonian_spectral_pair' : 'hamiltonian_spectral_single';

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setDataError(null);
    try {
      const payload = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: activeAnalysisType });
      const arr = safeArray(payload?.analyses);
      setAnalyses(arr);
      setSelectedAnalysisId((prev) => (prev && arr.some((a) => String(a.analysis_id) === String(prev)) ? prev : String(arr[0]?.analysis_id || '')));
    } catch (err) {
      setDataError(err.message || 'Failed to load analyses.');
    }
  }, [projectId, systemId, selectedClusterId, activeAnalysisType]);

  useEffect(() => { loadAnalyses(); }, [loadAnalyses]);

  useEffect(() => {
    if (!selectedAnalysisId || !selectedClusterId) {
      setAnalysisData(null);
      return;
    }
    let cancelled = false;
    const load = async () => {
      setLoadingData(true);
      setDataError(null);
      try {
        const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, activeAnalysisType, selectedAnalysisId);
        if (!cancelled) {
          setAnalysisData(payload);
          setComponentIndex(0);
        }
      } catch (err) {
        if (!cancelled) setDataError(err.message || 'Failed to load analysis data.');
      } finally {
        if (!cancelled) setLoadingData(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [projectId, systemId, selectedClusterId, activeAnalysisType, selectedAnalysisId]);

  useEffect(() => {
    if (!job?.job_id) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await fetchJobStatus(job.job_id);
        if (cancelled) return;
        setJobStatus(status);
        const done = ['finished', 'failed', 'canceled', 'cancelled'].includes(String(status?.status || status?.meta?.status || '').toLowerCase());
        if (done) loadAnalyses();
      } catch (err) {
        if (!cancelled) setJobError(err.message || 'Failed to poll job.');
      }
    };
    poll();
    const timer = setInterval(poll, 2000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [job, loadAnalyses]);

  const submitRun = useCallback(async () => {
    setJobError(null);
    try {
      const res = await submitHamiltonianSpectralJob({
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        state_ids: selectedStateIds,
        top_k: Math.max(1, Number(topK) || 20),
        overwrite,
      });
      setJob(res);
      setJobStatus(null);
      setRunPanelOpen(false);
    } catch (err) {
      setJobError(err.message || 'Failed to submit Hamiltonian spectral analysis.');
    }
  }, [projectId, systemId, selectedClusterId, selectedStateIds, topK, overwrite]);

  const meta = analysisData?.metadata || null;
  const data = analysisData?.data || {};
  const labels = useMemo(() => safeArray(data.residue_keys).map(String), [data.residue_keys]);
  const pairMode = mode === 'pair';
  const hasLaplacian = Array.isArray(data.laplacian_matrix) && Array.isArray(data.laplacian_top_eigenvectors);
  const viewMode = spectralView === 'laplacian' && hasLaplacian ? 'laplacian' : 'absolute';
  const communityIds = useMemo(() => safeArray(data.community_ids).map(Number), [data.community_ids]);
  const communitySizes = useMemo(() => safeArray(data.community_sizes).map((row) => safeArray(row).map(Number)), [data.community_sizes]);
  const communityOrder = useMemo(() => safeArray(data.community_matrix_order).map(Number), [data.community_matrix_order]);
  const communityInteraction = useMemo(() => safeArray(data.community_interaction_matrix), [data.community_interaction_matrix]);
  const rawMatrix = useMemo(() => {
    if (viewMode === 'laplacian') return safeArray(data.laplacian_source_matrix?.length ? data.laplacian_source_matrix : data.laplacian_matrix);
    return safeArray(data.matrix);
  }, [data.laplacian_source_matrix, data.laplacian_matrix, data.matrix, viewMode]);
  const matrix = useMemo(() => (viewMode === 'laplacian' ? orderMatrix(rawMatrix, communityOrder) : rawMatrix), [rawMatrix, communityOrder, viewMode]);
  const matrixLabels = useMemo(() => (viewMode === 'laplacian' ? orderLabels(labels, communityOrder) : labels), [labels, communityOrder, viewMode]);
  const eigenvalues = useMemo(() => safeArray(viewMode === 'laplacian' ? data.laplacian_eigenvalues : data.eigenvalues).map(Number), [data.laplacian_eigenvalues, data.eigenvalues, viewMode]);
  const topValues = useMemo(() => safeArray(viewMode === 'laplacian' ? data.laplacian_top_eigenvalues : data.top_eigenvalues).map(Number), [data.laplacian_top_eigenvalues, data.top_eigenvalues, viewMode]);
  const topVectors = useMemo(() => safeArray(viewMode === 'laplacian' ? data.laplacian_top_eigenvectors : data.top_eigenvectors), [data.laplacian_top_eigenvectors, data.top_eigenvectors, viewMode]);
  const selectedVector = useMemo(() => safeArray(topVectors[Math.min(componentIndex, Math.max(0, topVectors.length - 1))]).map(Number), [topVectors, componentIndex]);
  const selectedEigenvalue = topValues[Math.min(componentIndex, Math.max(0, topValues.length - 1))];
  const topEdges = useMemo(() => buildTopEdges(matrix, matrixLabels, edgeLimit), [matrix, matrixLabels, edgeLimit]);
  const viewTitle = viewMode === 'laplacian'
    ? (pairMode ? 'Differential Laplacian allostery' : 'Single-state Laplacian communities')
    : (pairMode ? 'ΔF spectral rewiring' : 'Absolute entropy / Frobenius');

  const eigenPlot = useMemo(() => ({
    data: [{ x: eigenvalues.map((_, i) => i + 1), y: eigenvalues, type: 'bar', marker: { color: viewMode === 'laplacian' ? '#a855f7' : (pairMode ? '#f59e0b' : '#22d3ee') } }],
    layout: { title: viewMode === 'laplacian' ? 'Laplacian spectrum (ascending)' : 'Eigenvalue spectrum', paper_bgcolor: '#111827', plot_bgcolor: '#111827', font: { color: '#e5e7eb' }, height: 260, margin: { l: 55, r: 15, t: 35, b: 45 }, xaxis: { title: 'component' }, yaxis: { title: viewMode === 'laplacian' ? 'L eigenvalue' : (pairMode ? 'ΔF eigenvalue' : 'F eigenvalue') } },
    config: { responsive: true, displayModeBar: false },
  }), [eigenvalues, pairMode, viewMode]);

  const vectorPlot = useMemo(() => ({
    data: [{
      x: labels,
      y: selectedVector.map((v) => (pairMode ? v : Math.abs(v))),
      type: 'bar',
      marker: { color: selectedVector.map((v) => (pairMode ? (v >= 0 ? '#ef4444' : '#3b82f6') : '#22c55e')) },
      hovertemplate: '%{x}<br>loading=%{y:.5f}<extra></extra>',
    }],
    layout: { title: `${pairMode ? 'Signed' : 'Absolute'} residue loadings · ${viewTitle} · component ${componentIndex + 1} · λ=${fmt(selectedEigenvalue)}`, paper_bgcolor: '#111827', plot_bgcolor: '#111827', font: { color: '#e5e7eb' }, height: 330, margin: { l: 55, r: 15, t: 45, b: 95 }, xaxis: { tickangle: -70, automargin: true }, yaxis: { title: pairMode ? 'v_i' : '|v_i|' } },
    config: { responsive: true, displayModeBar: false },
  }), [labels, selectedVector, pairMode, componentIndex, selectedEigenvalue, viewTitle]);

  const heatmapPlot = useMemo(() => ({
    data: [{ z: matrix, x: matrixLabels, y: matrixLabels, type: 'heatmap', colorscale: pairMode && viewMode !== 'laplacian' ? 'RdBu' : 'Viridis', reversescale: pairMode && viewMode !== 'laplacian', zmid: pairMode && viewMode !== 'laplacian' ? 0 : undefined, colorbar: { title: viewMode === 'laplacian' ? (pairMode ? '|ΔF|' : 'F') : (pairMode ? 'ΔF' : 'F') } }],
    layout: { title: viewMode === 'laplacian' ? 'Community-ordered Laplacian source matrix' : (pairMode ? 'Differential Frobenius matrix ΔF = F_B - F_A' : 'Frobenius coupling matrix F'), paper_bgcolor: '#111827', plot_bgcolor: '#111827', font: { color: '#e5e7eb' }, height: 520, margin: { l: 90, r: 20, t: 45, b: 90 }, xaxis: { tickangle: -70, automargin: true }, yaxis: { automargin: true } },
    config: { responsive: true, displayModeBar: true },
  }), [matrix, matrixLabels, pairMode, viewMode]);

  const communityPlot = useMemo(() => ({
    data: [{
      z: communityInteraction,
      x: communitySizes.map((row) => `c${row[0]}`),
      y: communitySizes.map((row) => `c${row[0]}`),
      type: 'heatmap',
      colorscale: 'YlOrRd',
      colorbar: { title: pairMode ? '|ΔF| sum' : 'F sum' },
    }],
    layout: { title: 'Sector-level network matrix', paper_bgcolor: '#111827', plot_bgcolor: '#111827', font: { color: '#e5e7eb' }, height: 320, margin: { l: 65, r: 20, t: 45, b: 60 } },
    config: { responsive: true, displayModeBar: false },
  }), [communityInteraction, communitySizes, pairMode]);

  if (loadingSystem) return <Loader message="Loading Hamiltonian spectral analyses..." />;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-6 overflow-y-auto">
      <HelpDrawer open={helpOpen} onClose={() => setHelpOpen(false)} title="Hamiltonian spectral analysis help" docPath="/docs/hamiltonian_spectral.md" />
      {runPanelOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-lg border border-gray-700 bg-gray-900 p-4 shadow-xl">
            <div className="flex items-start justify-between gap-3 border-b border-gray-800 pb-3">
              <div>
                <h2 className="text-lg font-semibold text-white">Run Hamiltonian spectral analysis</h2>
                <p className="text-xs text-gray-400 mt-1">Select states. Singles are computed first; missing pairs involving selected states are then added.</p>
              </div>
              <button type="button" onClick={() => setRunPanelOpen(false)} className="text-sm text-gray-400 hover:text-gray-100">Close</button>
            </div>
            <div className="space-y-3 pt-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Cluster</label>
                <select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100">
                  {clusters.map((cluster) => <option key={cluster.cluster_id} value={cluster.cluster_id}>{cluster.name || cluster.cluster_id}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">States</label>
                <select multiple value={selectedStateIds} onChange={(e) => setSelectedStateIds(Array.from(e.target.selectedOptions).map((o) => String(o.value)))} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 h-48">
                  {stateOptions.map((state) => <option key={state.state_id} value={state.state_id}>{state.name || state.state_id} ({state.state_id})</option>)}
                </select>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Eigenvectors to store</label>
                  <input type="number" min={1} step={1} value={topK} onChange={(e) => setTopK(Number(e.target.value))} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100" />
                </div>
                <label className="flex items-center gap-2 text-sm text-gray-200 pt-6">
                  <input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} className="h-4 w-4 rounded border-gray-700 bg-gray-950 text-cyan-500" />
                  Overwrite existing analyses
                </label>
              </div>
              <button type="button" onClick={submitRun} disabled={!selectedClusterId || !selectedStateIds.length} className="w-full rounded-md bg-cyan-500 px-3 py-2 text-sm font-semibold text-black hover:bg-cyan-400 disabled:opacity-50">Run analysis</button>
            </div>
          </div>
        </div>
      ) : null}

      <div className="max-w-7xl mx-auto space-y-4 pb-16">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-cyan-400">PHASE sectors</div>
            <h1 className="text-2xl font-semibold text-white mt-2">Hamiltonian spectral analysis</h1>
            <p className="text-sm text-gray-400 mt-1 max-w-3xl">
              Single mode diagonalizes the zero-sum-gauged Frobenius coupling matrix F. Pair mode diagonalizes ΔF = F_B - F_A to expose rewiring sectors.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => setHelpOpen(true)} className="inline-flex items-center gap-2 rounded-md border border-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-800"><CircleHelp className="h-4 w-4" /> Help</button>
            <button type="button" onClick={loadAnalyses} className="inline-flex items-center gap-2 rounded-md border border-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-800"><RefreshCw className="h-4 w-4" /> Refresh</button>
            <button type="button" onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/hamiltonian_spectral_3d${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)} className="rounded-md border border-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-800">3D view</button>
            <button type="button" onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/spectral_intersection${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)} className="rounded-md border border-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-800">Pistons</button>
            <button type="button" onClick={() => setRunPanelOpen(true)} className="rounded-md bg-cyan-500 px-3 py-2 text-sm font-semibold text-black hover:bg-cyan-400">Run analysis</button>
          </div>
        </div>
        {systemError ? <ErrorMessage message={systemError} /> : null}
        {dataError ? <ErrorMessage message={dataError} /> : null}
        {jobError ? <ErrorMessage message={jobError} /> : null}
        {job ? <div className="rounded-lg border border-cyan-800 bg-cyan-950/30 p-3 text-sm text-cyan-400">Job {job.job_id}: {jobStatus?.meta?.status || jobStatus?.status || 'queued'} · progress {Math.round(Number(jobStatus?.meta?.progress || jobStatus?.progress || 0))}%</div> : null}

        <div className="grid grid-cols-1 xl:grid-cols-[330px_minmax(0,1fr)] gap-4">
          <aside className="space-y-3">
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-3 space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Cluster</label>
                <select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100">
                  {clusters.map((cluster) => <option key={cluster.cluster_id} value={cluster.cluster_id}>{cluster.name || cluster.cluster_id}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Mode</label>
                <select value={mode} onChange={(e) => setMode(e.target.value)} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100">
                  <option value="single">Single-state sectors</option>
                  <option value="pair">Pair rewiring sectors</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Spectral view</label>
                <select
                  value={spectralView}
                  onChange={(e) => { setSpectralView(e.target.value); setComponentIndex(0); }}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  <option value="laplacian">{pairMode ? 'Differential Laplacian allostery' : 'Single-state Laplacian communities'}</option>
                  <option value="absolute">{pairMode ? 'ΔF spectral rewiring' : 'Absolute entropy / Frobenius'}</option>
                </select>
                {spectralView === 'laplacian' && !hasLaplacian ? <p className="mt-1 text-[11px] text-amber-300">This analysis was computed before v3 Laplacian/community support. Rerun spectral analysis to upgrade it.</p> : null}
              </div>
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-[0.15em] text-gray-500">Available analyses</div>
                {analyses.map((analysis) => (
                  <button key={analysis.analysis_id} type="button" onClick={() => setSelectedAnalysisId(String(analysis.analysis_id))} className={`w-full rounded-md border px-3 py-2 text-left ${String(selectedAnalysisId) === String(analysis.analysis_id) ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-700 bg-gray-950/50 hover:bg-gray-800'}`}>
                    <div className="text-sm text-gray-100">{analysisTitle(analysis)}</div>
                    <div className="text-xs text-gray-400 mt-1">{String(analysis.updated_at || analysis.created_at || '').slice(0, 19)} · top k {analysis?.summary?.top_k ?? 'n/a'}</div>
                  </button>
                ))}
                {!analyses.length ? <p className="text-xs text-gray-500">No {mode} analyses yet.</p> : null}
              </div>
            </div>
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-3 space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Component</label>
                <select value={componentIndex} onChange={(e) => setComponentIndex(Number(e.target.value))} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100">
                  {topValues.map((value, idx) => <option key={idx} value={idx}>{viewMode === 'laplacian' ? 'Fiedler ' : 'v'}{idx + 1} · λ={fmt(value)}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Top edges shown</label>
                <input type="number" min={5} step={5} value={edgeLimit} onChange={(e) => setEdgeLimit(Number(e.target.value))} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100" />
              </div>
            </div>
          </aside>

          <section className="space-y-4 min-w-0">
            {loadingData ? <Loader message="Loading spectral analysis..." /> : null}
            {!loadingData && !analysisData ? <div className="rounded-lg border border-gray-800 bg-gray-900 p-6 text-sm text-gray-300">Select an existing analysis or run a new one.</div> : null}
            {!loadingData && analysisData ? (
              <>
                <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                  <div className="text-xs uppercase tracking-[0.15em] text-gray-500">Selected analysis</div>
                  <h2 className="mt-2 text-lg font-semibold text-white">{analysisTitle(meta)}</h2>
                  <p className="mt-1 text-sm text-gray-400">
                    {pairMode
                      ? (viewMode === 'laplacian' ? `Differential Laplacian uses A=|ΔF| and DADApy communities to highlight normalized allosteric pathways between ${meta?.state_a_name || meta?.state_a_id} and ${meta?.state_b_name || meta?.state_b_id}.` : `Pair analysis uses signed ΔF. Red positive loadings increase from ${meta?.state_a_name || meta?.state_a_id} to ${meta?.state_b_name || meta?.state_b_id}; blue negative loadings decrease.`)
                      : (viewMode === 'laplacian' ? `Single-state Laplacian uses A=F and DADApy communities to identify rigid coupled structural modules in ${meta?.state_name || scalarString(data.state_name)}.` : `Single analysis uses |v_i| from the Frobenius coupling matrix of ${meta?.state_name || scalarString(data.state_name)}.`)}
                  </p>
                </div>
                <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4">
                  <div className="rounded-lg border border-gray-800 bg-gray-900 p-3 overflow-hidden space-y-2">
                    <div>
                      <h3 className="text-sm font-semibold text-gray-100">Eigenvalue spectrum</h3>
                      <p className="text-xs text-gray-400">Absolute mode uses large eigenvalues. Laplacian allostery mode focuses on the smallest non-zero eigenvalues, i.e. Fiedler-like rewiring communities.</p>
                    </div>
                    <Plot data={eigenPlot.data} layout={eigenPlot.layout} config={eigenPlot.config} style={{ width: '100%' }} />
                  </div>
                  <div className="rounded-lg border border-gray-800 bg-gray-900 p-3 overflow-hidden space-y-2">
                    <div>
                      <h3 className="text-sm font-semibold text-gray-100">Residue loadings</h3>
                      <p className="text-xs text-gray-400">Absolute modes show Hamiltonian sector loadings. Laplacian modes show Fiedler-like loadings used to build the community embedding.</p>
                    </div>
                    <Plot data={vectorPlot.data} layout={vectorPlot.layout} config={vectorPlot.config} style={{ width: '100%' }} />
                  </div>
                </div>
                <div className="rounded-lg border border-gray-800 bg-gray-900 p-3 overflow-hidden space-y-2">
                  <div>
                    <h3 className="text-sm font-semibold text-gray-100">Coupling matrix heatmap</h3>
                    <p className="text-xs text-gray-400">Absolute modes show F or signed ΔF. Laplacian modes show the source adjacency reordered by DADApy community: F for single-state modules and |ΔF| for pair allostery pathways.</p>
                  </div>
                  <Plot data={heatmapPlot.data} layout={heatmapPlot.layout} config={heatmapPlot.config} style={{ width: '100%' }} />
                </div>
                {viewMode === 'laplacian' && communityIds.length ? (
                  <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4">
                    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 overflow-x-auto">
                      <h3 className="text-sm font-semibold text-gray-100 mb-2">DADApy communities</h3>
                      <p className="text-xs text-gray-400 mb-3">Community labels are categorical sectors from cosine-distance density peak clustering in Laplacian spectral space.</p>
                      <table className="min-w-full text-sm">
                        <thead className="text-xs uppercase tracking-[0.12em] text-gray-500"><tr><th className="px-2 py-2 text-left">Community</th><th className="px-2 py-2 text-right">Residues</th></tr></thead>
                        <tbody>
                          {communitySizes.map((row) => <tr key={row[0]} className="border-t border-gray-800"><td className="px-2 py-2 text-gray-200">c{row[0]}</td><td className="px-2 py-2 text-right text-gray-300">{row[1]}</td></tr>)}
                        </tbody>
                      </table>
                    </div>
                    <div className="rounded-lg border border-gray-800 bg-gray-900 p-3 overflow-hidden">
                      <Plot data={communityPlot.data} layout={communityPlot.layout} config={communityPlot.config} style={{ width: '100%' }} />
                    </div>
                  </div>
                ) : null}
                <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 overflow-x-auto">
                  <h3 className="text-sm font-semibold text-gray-100 mb-2">Strongest matrix edges</h3>
                  <table className="min-w-full text-sm">
                    <thead className="text-xs uppercase tracking-[0.12em] text-gray-500"><tr><th className="px-2 py-2 text-left">Residue A</th><th className="px-2 py-2 text-left">Residue B</th><th className="px-2 py-2 text-right">Value</th><th className="px-2 py-2 text-right">|Value|</th></tr></thead>
                    <tbody>
                      {topEdges.map((edge) => <tr key={`${edge.i}:${edge.j}`} className="border-t border-gray-800"><td className="px-2 py-2 text-gray-200">{edge.a}</td><td className="px-2 py-2 text-gray-200">{edge.b}</td><td className="px-2 py-2 text-right text-gray-300">{fmt(edge.value)}</td><td className="px-2 py-2 text-right text-gray-300">{fmt(edge.abs)}</td></tr>)}
                    </tbody>
                  </table>
                </div>
              </>
            ) : null}
          </section>
        </div>
      </div>
    </div>
  );
}
