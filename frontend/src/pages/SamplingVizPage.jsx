import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { Info, Play, RefreshCw, Trash2, X } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import {
  deleteSamplingSample,
  fetchClusterAnalyses,
  fetchClusterAnalysisData,
  fetchPottsClusterInfo,
  fetchSampleStats,
  fetchSystem,
} from '../api/projects';
import { fetchJobStatus, submitMdSamplesRefreshJob, submitPottsAnalysisJob } from '../api/jobs';

const palette = ['#22d3ee', '#f97316', '#10b981', '#f43f5e', '#60a5fa', '#f59e0b', '#a855f7', '#84cc16'];

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

function SampleInfoPanel({ sample, stats, onClose }) {
  if (!sample) return null;
  return (
    <div className="rounded-md border border-gray-800 bg-gray-950/60 p-2 text-[11px] text-gray-300 space-y-2">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-xs font-semibold text-white">{sample.name || sample.sample_id}</p>
          <p className="text-[10px] text-gray-500">Sample info</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-gray-400 hover:text-gray-200"
          aria-label="Close sample info"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="space-y-1">
        <div>
          <span className="text-gray-400">id:</span> {sample.sample_id}
        </div>
        {sample.created_at && (
          <div>
            <span className="text-gray-400">created:</span> {sample.created_at}
          </div>
        )}
        {sample.type && (
          <div>
            <span className="text-gray-400">type:</span> {sample.type}
          </div>
        )}
        {sample.method && (
          <div>
            <span className="text-gray-400">method:</span> {sample.method}
          </div>
        )}
        {sample.source && (
          <div>
            <span className="text-gray-400">source:</span> {sample.source}
          </div>
        )}
        {sample.model_names && sample.model_names.length > 0 && (
          <div>
            <span className="text-gray-400">models:</span> {sample.model_names.join(', ')}
          </div>
        )}
        {sample.path && (
          <div className="break-all">
            <span className="text-gray-400">path:</span> {sample.path}
          </div>
        )}
      </div>

      {stats && (
        <div className="space-y-1">
          <p className="text-[10px] text-gray-500">NPZ stats</p>
          <div>
            <span className="text-gray-400">frames:</span> {stats.n_frames}
          </div>
          <div>
            <span className="text-gray-400">residues:</span> {stats.n_residues}
          </div>
          {typeof stats.invalid_count === 'number' && typeof stats.invalid_fraction === 'number' && (
            <div>
              <span className="text-gray-400">invalid:</span> {stats.invalid_count} (
              {(stats.invalid_fraction * 100).toFixed(2)}%)
            </div>
          )}
          <div>
            <span className="text-gray-400">has halo labels:</span> {stats.has_halo ? 'yes' : 'no'}
          </div>
        </div>
      )}

      {sample.params && (
        <details className="text-[11px] text-gray-300">
          <summary className="cursor-pointer text-gray-200">Params</summary>
          <pre className="mt-2 max-h-56 overflow-auto rounded bg-gray-900 p-2 text-[10px] text-gray-300">
            {JSON.stringify(sample.params, null, 2)}
          </pre>
        </details>
      )}
    </div>
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

  const [clusterInfo, setClusterInfo] = useState(null);
  const [clusterInfoLoading, setClusterInfoLoading] = useState(false);
  const [clusterInfoError, setClusterInfoError] = useState(null);

  const [analyses, setAnalyses] = useState([]);
  const [analysesLoading, setAnalysesLoading] = useState(false);
  const [analysesError, setAnalysesError] = useState(null);
  const [analysisDataCache, setAnalysisDataCache] = useState({});

  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [keepInvalid, setKeepInvalid] = useState(false);

  const [selectedModelId, setSelectedModelId] = useState('');

  const [selectedMdSampleId, setSelectedMdSampleId] = useState('');
  const [selectedSampleId, setSelectedSampleId] = useState('');

  const [analysisJob, setAnalysisJob] = useState(null);
  const [analysisJobStatus, setAnalysisJobStatus] = useState(null);

  const [mdRefreshJob, setMdRefreshJob] = useState(null);
  const [mdRefreshJobStatus, setMdRefreshJobStatus] = useState(null);
  const [mdRefreshError, setMdRefreshError] = useState(null);

  const [infoSampleId, setInfoSampleId] = useState('');
  const [sampleStatsCache, setSampleStatsCache] = useState({});
  const [sampleStatsError, setSampleStatsError] = useState(null);

  const [overlayPlot, setOverlayPlot] = useState(null);

  const clusterOptions = useMemo(
    () => (system?.metastable_clusters || []).filter((run) => run.path && run.status !== 'failed'),
    [system]
  );

  const selectedCluster = useMemo(
    () => clusterOptions.find((c) => c.cluster_id === selectedClusterId) || null,
    [clusterOptions, selectedClusterId]
  );

  const sampleEntries = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);
  const pottsModels = useMemo(() => selectedCluster?.potts_models || [], [selectedCluster]);

  const mdSamples = useMemo(() => sampleEntries.filter((s) => s.type === 'md_eval'), [sampleEntries]);
  const filteredSamples = useMemo(() => {
    if (!selectedModelId) return sampleEntries;
    return sampleEntries.filter((s) => {
      if (s.type === 'md_eval') return true;
      const ids = Array.isArray(s.model_ids) ? s.model_ids : s.model_id ? [s.model_id] : [];
      if (!ids.length) return true;
      return ids.includes(selectedModelId);
    });
  }, [sampleEntries, selectedModelId]);
  const gibbsSamples = useMemo(
    () => filteredSamples.filter((s) => s.type === 'potts_sampling' && s.method === 'gibbs'),
    [filteredSamples]
  );
  const saSamples = useMemo(
    () => filteredSamples.filter((s) => s.type === 'potts_sampling' && s.method === 'sa'),
    [filteredSamples]
  );
  const pottsSamples = useMemo(() => [...gibbsSamples, ...saSamples], [gibbsSamples, saSamples]);

  const infoSample = useMemo(() => sampleEntries.find((s) => s.sample_id === infoSampleId) || null, [sampleEntries, infoSampleId]);
  const infoSampleStats = useMemo(() => (infoSampleId ? sampleStatsCache[infoSampleId] : null), [sampleStatsCache, infoSampleId]);

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
      setSelectedModelId('');
      return;
    }
    if (!selectedModelId) {
      setSelectedModelId(pottsModels[0]?.model_id || '');
    } else if (!pottsModels.some((m) => m.model_id === selectedModelId)) {
      setSelectedModelId(pottsModels[0]?.model_id || '');
    }
  }, [pottsModels, selectedModelId]);

  const loadClusterInfo = useCallback(async (modelIdOverride) => {
    if (!selectedClusterId) return;
    setClusterInfoLoading(true);
    setClusterInfoError(null);
    try {
      const modelId = typeof modelIdOverride === 'string' && modelIdOverride ? modelIdOverride : '';
      const data = await fetchPottsClusterInfo(projectId, systemId, selectedClusterId, { modelId: modelId || undefined });
      setClusterInfo(data);
    } catch (err) {
      setClusterInfoError(err.message || 'Failed to load cluster info.');
      setClusterInfo(null);
    } finally {
      setClusterInfoLoading(false);
    }
  }, [projectId, systemId, selectedClusterId]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesLoading(true);
    setAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId);
      setAnalyses(Array.isArray(data?.analyses) ? data.analyses : []);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
    } finally {
      setAnalysesLoading(false);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    loadClusterInfo(selectedModelId);
    loadAnalyses();
    setAnalysisDataCache({});
    setSelectedMdSampleId('');
    setInfoSampleId('');
  }, [selectedClusterId, loadClusterInfo, loadAnalyses]);

  useEffect(() => {
    if (!selectedClusterId) return;
    // Update edge count/info when switching the active Potts model.
    loadClusterInfo(selectedModelId);
  }, [selectedModelId, selectedClusterId, loadClusterInfo]);

  useEffect(() => {
    if (!mdSamples.length) {
      setSelectedMdSampleId('');
      return;
    }
    if (!selectedMdSampleId || !mdSamples.some((s) => s.sample_id === selectedMdSampleId)) {
      setSelectedMdSampleId(mdSamples[0].sample_id);
    }
  }, [mdSamples, selectedMdSampleId]);

  useEffect(() => {
    if (!pottsSamples.length) {
      setSelectedSampleId('');
      return;
    }
    if (!selectedSampleId || !pottsSamples.some((s) => s.sample_id === selectedSampleId)) {
      setSelectedSampleId(pottsSamples[0].sample_id);
    }
  }, [pottsSamples, selectedSampleId]);

  const mdVsSampleAnalyses = useMemo(
    () => analyses.filter((a) => a.analysis_type === 'md_vs_sample'),
    [analyses]
  );
  const modelEnergyAnalyses = useMemo(
    () => analyses.filter((a) => a.analysis_type === 'model_energy'),
    [analyses]
  );

  const dropInvalid = !keepInvalid;

  const selectedMdVsMeta = useMemo(() => {
    if (!selectedMdSampleId || !selectedSampleId) return null;
    const candidates = mdVsSampleAnalyses.filter((a) => {
      const mode = (a.md_label_mode || 'assigned').toLowerCase();
      return (
        a.md_sample_id === selectedMdSampleId &&
        a.sample_id === selectedSampleId &&
        mode === mdLabelMode &&
        Boolean(a.drop_invalid) === Boolean(dropInvalid)
      );
    });
    if (!candidates.length) return null;
    if (selectedModelId) {
      const withModel = candidates.find((a) => a.model_id === selectedModelId);
      if (withModel) return withModel;
    }
    return candidates[0];
  }, [mdVsSampleAnalyses, selectedMdSampleId, selectedSampleId, mdLabelMode, dropInvalid, selectedModelId]);

  const loadAnalysisData = useCallback(
    async (analysisType, analysisId) => {
      if (!analysisType || !analysisId) return null;
      const cacheKey = `${analysisType}:${analysisId}`;
      if (analysisDataCache[cacheKey]) return analysisDataCache[cacheKey];
      const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, analysisType, analysisId);
      setAnalysisDataCache((prev) => ({ ...prev, [cacheKey]: payload }));
      return payload;
    },
    [analysisDataCache, projectId, systemId, selectedClusterId]
  );

  const [comparisonData, setComparisonData] = useState(null);
  const [comparisonError, setComparisonError] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);

  useEffect(() => {
    const run = async () => {
      setComparisonError(null);
      setComparisonData(null);
      if (!selectedMdVsMeta) return;
      setComparisonLoading(true);
      try {
        const payload = await loadAnalysisData('md_vs_sample', selectedMdVsMeta.analysis_id);
        setComparisonData(payload);
      } catch (err) {
        setComparisonError(err.message || 'Failed to load analysis.');
      } finally {
        setComparisonLoading(false);
      }
    };
    run();
  }, [selectedMdVsMeta, loadAnalysisData]);

  const residueLabels = useMemo(() => {
    const keys = clusterInfo?.residue_keys || [];
    if (Array.isArray(keys) && keys.length) return keys;
    const n = clusterInfo?.n_residues || 0;
    return Array.from({ length: n }, (_, i) => String(i));
  }, [clusterInfo]);

  const edges = useMemo(() => {
    const fromAnalysis = comparisonData?.data?.edges;
    if (Array.isArray(fromAnalysis) && fromAnalysis.length) return fromAnalysis;
    return Array.isArray(clusterInfo?.edges) ? clusterInfo.edges : [];
  }, [clusterInfo, comparisonData]);

  const nodeJs = useMemo(() => {
    const arr = comparisonData?.data?.node_js || [];
    return Array.isArray(arr) ? arr : [];
  }, [comparisonData]);
  const edgeJs = useMemo(() => {
    const arr = comparisonData?.data?.edge_js || [];
    return Array.isArray(arr) ? arr : [];
  }, [comparisonData]);

  const topResidues = useMemo(() => topK(nodeJs, residueLabels, 10), [nodeJs, residueLabels]);
  const topEdges = useMemo(() => {
    if (!edges.length || !edgeJs.length) return [];
    const labels = edges.map((e) => `${residueLabels[e[0]] ?? e[0]} — ${residueLabels[e[1]] ?? e[1]}`);
    return topK(edgeJs, labels, 10);
  }, [edges, edgeJs, residueLabels]);

  const edgeMatrix = useMemo(
    () => buildEdgeMatrix(residueLabels.length, edges, edgeJs),
    [residueLabels, edges, edgeJs]
  );
  const edgeMatrixHasValues = useMemo(
    () => edgeMatrix?.some((row) => row?.some((val) => Number.isFinite(val))),
    [edgeMatrix]
  );

  const handleRunAnalysis = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesError(null);
    setAnalysisJob(null);
    setAnalysisJobStatus(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        md_label_mode: mdLabelMode,
        keep_invalid: keepInvalid,
      };
      if (selectedModelId) payload.model_id = selectedModelId;
      const res = await submitPottsAnalysisJob(payload);
      setAnalysisJob(res);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to submit analysis job.');
    }
  }, [projectId, systemId, selectedClusterId, mdLabelMode, keepInvalid, selectedModelId]);

  const handleRefreshMdSamples = useCallback(async () => {
    if (!selectedClusterId) return;
    setMdRefreshError(null);
    setMdRefreshJob(null);
    setMdRefreshJobStatus(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        overwrite: true,
        cleanup: true,
      };
      const res = await submitMdSamplesRefreshJob(payload);
      setMdRefreshJob(res);
    } catch (err) {
      setMdRefreshError(err.message || 'Failed to submit MD refresh job.');
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!analysisJob?.job_id) return;
    let cancelled = false;
    const terminal = new Set(['finished', 'failed', 'canceled']);
    const poll = async () => {
      try {
        const status = await fetchJobStatus(analysisJob.job_id);
        if (cancelled) return;
        setAnalysisJobStatus(status);
        if (terminal.has(status?.status)) {
          // Stop polling once the job is done.
          clearInterval(timer);
          if (status?.status === 'finished') await loadAnalyses();
        }
      } catch (err) {
        if (!cancelled) setAnalysesError(err.message || 'Failed to poll analysis job.');
      }
    };
    const timer = setInterval(poll, 2000);
    poll();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [analysisJob, loadAnalyses]);

  useEffect(() => {
    if (!mdRefreshJob?.job_id) return;
    let cancelled = false;
    const terminal = new Set(['finished', 'failed', 'canceled']);
    const poll = async () => {
      try {
        const status = await fetchJobStatus(mdRefreshJob.job_id);
        if (cancelled) return;
        setMdRefreshJobStatus(status);
        if (terminal.has(status?.status)) {
          clearInterval(timer);
          if (status?.status === 'finished') {
            const data = await fetchSystem(projectId, systemId);
            if (!cancelled) setSystem(data);
          }
        }
      } catch (err) {
        if (!cancelled) setMdRefreshError(err.message || 'Failed to poll MD refresh job.');
      }
    };
    const timer = setInterval(poll, 2000);
    poll();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [mdRefreshJob, projectId, systemId]);

  const handleDeleteSample = useCallback(
    async (sampleId) => {
      if (!selectedClusterId || !sampleId) return;
      const ok = window.confirm('Delete this sample? This cannot be undone.');
      if (!ok) return;
      try {
        await deleteSamplingSample(projectId, systemId, selectedClusterId, sampleId);
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
        setInfoSampleId('');
      } catch (err) {
        setSystemError(err.message || 'Failed to delete sample.');
      }
    },
    [projectId, systemId, selectedClusterId]
  );

  const toggleInfo = useCallback((sampleId) => {
    setInfoSampleId((prev) => (prev === sampleId ? '' : sampleId));
  }, []);

  useEffect(() => {
    const load = async () => {
      if (!infoSampleId) return;
      if (sampleStatsCache[infoSampleId]) return;
      setSampleStatsError(null);
      try {
        const stats = await fetchSampleStats(projectId, systemId, selectedClusterId, infoSampleId);
        setSampleStatsCache((prev) => ({ ...prev, [infoSampleId]: stats }));
      } catch (err) {
        setSampleStatsError(err.message || 'Failed to load sample stats.');
      }
    };
    load();
  }, [infoSampleId, sampleStatsCache, projectId, systemId, selectedClusterId]);

  const energyAnalysesForModel = useMemo(() => {
    if (!selectedModelId) return [];
    // Dedupe: repeated runs create multiple analysis entries per sample_id.
    // Analyses are already sorted newest-first by the backend, so keep the first per sample_id.
    const out = [];
    const seen = new Set();
    for (const a of modelEnergyAnalyses) {
      if (a.model_id !== selectedModelId) continue;
      if ((a.md_label_mode || 'assigned') !== mdLabelMode) continue;
      if (Boolean(a.drop_invalid) !== Boolean(dropInvalid)) continue;
      const sid = a.sample_id || '';
      if (!sid) continue;
      if (seen.has(sid)) continue;
      seen.add(sid);
      out.push(a);
    }
    return out;
  }, [modelEnergyAnalyses, selectedModelId, mdLabelMode, dropInvalid]);

  const [energySeries, setEnergySeries] = useState([]);
  const [energyError, setEnergyError] = useState(null);
  const [energyLoading, setEnergyLoading] = useState(false);

  useEffect(() => {
    const run = async () => {
      setEnergyError(null);
      setEnergySeries([]);
      if (!selectedModelId) return;
      const metas = energyAnalysesForModel;
      if (!metas.length) return;
      setEnergyLoading(true);
      try {
        const series = [];
        // Load energies for all samples (MD + Potts) that have an analysis for this model.
        for (let idx = 0; idx < metas.length; idx += 1) {
          const meta = metas[idx];
          const payload = await loadAnalysisData('model_energy', meta.analysis_id);
          const energies = payload?.data?.energies || [];
          if (!Array.isArray(energies) || !energies.length) continue;
          const sample = sampleEntries.find((s) => s.sample_id === meta.sample_id);
          series.push({
            sample_id: meta.sample_id,
            label: sample?.name || meta.sample_id,
            energies,
          });
        }
        setEnergySeries(series);
      } catch (err) {
        setEnergyError(err.message || 'Failed to load energies.');
      } finally {
        setEnergyLoading(false);
      }
    };
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModelId, energyAnalysesForModel, loadAnalysisData]);

  const energyPlot = useMemo(() => {
    if (!energySeries.length) return null;

    // Use a shared xbins across traces so histograms align.
    let globalMin = Infinity;
    let globalMax = -Infinity;
    let minBinSize = Infinity;
    for (const s of energySeries) {
      const arr = Array.isArray(s.energies) ? s.energies : [];
      if (!arr.length) continue;
      let localMin = Infinity;
      let localMax = -Infinity;
      for (let i = 0; i < arr.length; i += 1) {
        const v = arr[i];
        if (!Number.isFinite(v)) continue;
        if (v < localMin) localMin = v;
        if (v > localMax) localMax = v;
      }
      if (!Number.isFinite(localMin) || !Number.isFinite(localMax)) continue;
      if (Number.isFinite(localMin)) globalMin = Math.min(globalMin, localMin);
      if (Number.isFinite(localMax)) globalMax = Math.max(globalMax, localMax);
      const range = localMax - localMin;
      if (Number.isFinite(range) && range > 0) {
        // "Thinner binning among all distributions": pick the smallest implied bin size.
        minBinSize = Math.min(minBinSize, range / 40);
      }
    }
    if (!Number.isFinite(globalMin) || !Number.isFinite(globalMax)) return null;
    const globalRange = globalMax - globalMin;
    if (!Number.isFinite(minBinSize) || minBinSize <= 0) {
      minBinSize = globalRange > 0 ? globalRange / 40 : 1.0;
    }
    if (globalRange > 0) {
      const maxBins = 200;
      const impliedBins = globalRange / minBinSize;
      if (Number.isFinite(impliedBins) && impliedBins > maxBins) {
        minBinSize = globalRange / maxBins;
      }
    }

    return {
      data: energySeries.map((s, idx) => ({
        x: s.energies,
        type: 'histogram',
        name: s.label,
        opacity: 0.55,
        marker: { color: pickColor(idx) },
        autobinx: false,
        xbins: { start: globalMin, end: globalMax, size: minBinSize },
        bingroup: 'energies',
      })),
      layout: {
        height: 260,
        margin: { l: 40, r: 10, t: 10, b: 40 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#111827' },
        barmode: 'overlay',
        xaxis: { title: 'Energy', color: '#111827' },
        yaxis: { title: 'Count', color: '#111827' },
      },
    };
  }, [energySeries]);

  if (loadingSystem) return <Loader message="Loading sampling explorer..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  const canCompare = Boolean(selectedMdSampleId && selectedSampleId);
  const comparisonMissing = canCompare && !selectedMdVsMeta;

  return (
    <div className="space-y-4">
      <PlotOverlay overlay={overlayPlot} onClose={() => setOverlayPlot(null)} />

      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Sampling Explorer</h1>
          <p className="text-sm text-gray-400">
            Sampling runs save only <code>sample.npz</code>. Use the analysis job to generate derived metrics under{' '}
            <code>clusters/&lt;cluster_id&gt;/analyses/</code>.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_eval`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Delta eval
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Back to system
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-4">
        <aside className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Cluster</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              >
                {clusterOptions.map((run) => {
                  const name = run.name || run.path?.split('/').pop() || run.cluster_id;
                  return (
                    <option key={run.cluster_id} value={run.cluster_id}>
                      {name}
                    </option>
                  );
                })}
              </select>
              {clusterInfoLoading && <p className="text-[11px] text-gray-500">Loading cluster info…</p>}
              {clusterInfoError && <p className="text-[11px] text-red-300">{clusterInfoError}</p>}
              {clusterInfo && (
                <p className="text-[11px] text-gray-500">
                  Residues: {clusterInfo.n_residues} · Edges: {clusterInfo.n_edges}
                  {clusterInfo.edges_source ? ` (${clusterInfo.edges_source})` : ''}
                </p>
              )}
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400">MD label mode</label>
                <select
                  value={mdLabelMode}
                  onChange={(e) => setMdLabelMode(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                >
                  <option value="assigned">Assigned</option>
                  <option value="halo">Halo</option>
                </select>
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 text-xs text-gray-300">
                  <input
                    type="checkbox"
                    checked={keepInvalid}
                    onChange={(e) => setKeepInvalid(e.target.checked)}
                    className="h-4 w-4 rounded border-gray-500 bg-gray-900 text-cyan-500 focus:ring-cyan-500"
                  />
                  Keep invalid SA
                </label>
              </div>
            </div>

            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Potts model (edges + energies + sample filter)</label>
              <select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                disabled={!pottsModels.length}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white disabled:opacity-60"
              >
                {!pottsModels.length && <option value="">No models</option>}
                {pottsModels.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.name || m.model_id}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleRunAnalysis}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white text-sm"
              >
                <Play className="h-4 w-4" />
                Run analysis
              </button>
              <button
                type="button"
                onClick={async () => {
                  await loadClusterInfo(selectedModelId);
                  await loadAnalyses();
                }}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 text-gray-200 text-sm hover:border-gray-500"
              >
                <RefreshCw className="h-4 w-4" />
                Refresh
              </button>
            </div>

            {analysisJob?.job_id && (
              <div className="text-[11px] text-gray-300">
                Job: <span className="text-gray-200">{analysisJob.job_id}</span>{' '}
                {analysisJobStatus?.meta?.status ? `· ${analysisJobStatus.meta.status}` : ''}
                {typeof analysisJobStatus?.meta?.progress === 'number' ? ` · ${analysisJobStatus.meta.progress}%` : ''}
              </div>
            )}
            {analysesError && <ErrorMessage message={analysesError} />}
          </div>

          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
            <div>
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-semibold text-gray-300">MD samples</p>
                <button
                  type="button"
                  onClick={handleRefreshMdSamples}
                  className="inline-flex items-center gap-1 text-[11px] text-gray-300 hover:text-white disabled:opacity-50"
                  disabled={!selectedClusterId || mdRefreshJobStatus?.status === 'started' || mdRefreshJobStatus?.status === 'queued'}
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  Recompute
                </button>
              </div>
              {mdRefreshJob?.job_id && (
                <div className="text-[11px] text-gray-400 mt-1">
                  Refresh job: <span className="text-gray-200">{mdRefreshJob.job_id}</span>{' '}
                  {mdRefreshJobStatus?.meta?.status ? `· ${mdRefreshJobStatus.meta.status}` : ''}
                  {typeof mdRefreshJobStatus?.meta?.progress === 'number' ? ` · ${mdRefreshJobStatus.meta.progress}%` : ''}
                </div>
              )}
              {mdRefreshError && <ErrorMessage message={mdRefreshError} />}
              {mdSamples.length === 0 && <p className="text-[11px] text-gray-500 mt-1">No MD samples yet.</p>}
              {mdSamples.length > 0 && (
                <div className="space-y-1 mt-2">
                  {mdSamples.map((sample) => (
                    <div
                      key={sample.sample_id || sample.path}
                      className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/40 px-2 py-1 text-[11px] text-gray-300"
                    >
                      <button
                        type="button"
                        onClick={() => setSelectedMdSampleId(sample.sample_id)}
                        className={`truncate text-left ${selectedMdSampleId === sample.sample_id ? 'text-cyan-200' : ''}`}
                      >
                        {sample.name || 'MD sample'} • {sample.created_at || ''}
                      </button>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => toggleInfo(sample.sample_id)}
                          className="text-gray-400 hover:text-gray-200"
                          aria-label={`Show info for ${sample.name || 'MD sample'}`}
                        >
                          <Info className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <p className="text-xs font-semibold text-gray-300">Potts samples</p>
              {pottsSamples.length === 0 && <p className="text-[11px] text-gray-500 mt-1">No Potts samples yet.</p>}
              {pottsSamples.length > 0 && (
                <div className="space-y-1 mt-2">
                  {pottsSamples.map((sample) => (
                    <div
                      key={sample.sample_id || sample.path}
                      className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/40 px-2 py-1 text-[11px] text-gray-300"
                    >
                      <button
                        type="button"
                        onClick={() => setSelectedSampleId(sample.sample_id)}
                        className={`truncate text-left ${selectedSampleId === sample.sample_id ? 'text-cyan-200' : ''}`}
                      >
                        {sample.name || 'Potts sample'} • {sample.created_at || ''}
                      </button>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => toggleInfo(sample.sample_id)}
                          className="text-gray-400 hover:text-gray-200"
                          aria-label={`Show info for ${sample.name || 'Potts sample'}`}
                        >
                          <Info className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeleteSample(sample.sample_id)}
                          className="text-gray-400 hover:text-red-300"
                          aria-label={`Delete ${sample.name || 'Potts sample'}`}
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {sampleStatsError && <ErrorMessage message={sampleStatsError} />}
            <SampleInfoPanel
              sample={infoSample}
              stats={infoSampleStats}
              onClose={() => setInfoSampleId('')}
            />
          </div>
        </aside>

        <main className="space-y-4">
          <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-gray-200">MD vs sample</h2>
                <p className="text-[11px] text-gray-500">
                  Shows JS divergence on nodes/edges for the selected MD sample and Potts sample.
                </p>
              </div>
              {analysesLoading && <p className="text-[11px] text-gray-500">Loading analyses…</p>}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">MD sample</label>
                <select
                  value={selectedMdSampleId}
                  onChange={(e) => setSelectedMdSampleId(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  {mdSamples.map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {s.name || s.sample_id}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Potts sample</label>
                <select
                  value={selectedSampleId}
                  onChange={(e) => setSelectedSampleId(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  {pottsSamples.map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {s.name || s.sample_id}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {comparisonMissing && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                No analysis found for this pair/settings. Click <span className="font-semibold">Run analysis</span> to compute it.
              </div>
            )}
            {comparisonError && <ErrorMessage message={comparisonError} />}
            {comparisonLoading && <p className="text-sm text-gray-400">Loading…</p>}

            {comparisonData && (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div className="rounded-md border border-gray-800 bg-white p-3">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <p className="text-xs font-semibold text-gray-800">Node JS</p>
                    <button
                      type="button"
                      className="text-[11px] text-gray-600 hover:text-gray-800"
                      onClick={() =>
                        setOverlayPlot({
                          title: 'Node JS',
                          data: [
                            {
                              x: residueLabels,
                              y: nodeJs,
                              type: 'bar',
                              marker: { color: '#22d3ee' },
                            },
                          ],
                          layout: {
                            margin: { l: 40, r: 10, t: 20, b: 80 },
                            paper_bgcolor: '#ffffff',
                            plot_bgcolor: '#ffffff',
                            font: { color: '#111827' },
                            xaxis: { tickfont: { size: 9 }, color: '#111827' },
                            yaxis: { title: 'JS divergence', color: '#111827' },
                          },
                        })
                      }
                    >
                      Maximize
                    </button>
                  </div>
                  <Plot
                    data={[
                      {
                        x: residueLabels,
                        y: nodeJs,
                        type: 'bar',
                        marker: { color: '#22d3ee' },
                      },
                    ]}
                    layout={{
                      margin: { l: 40, r: 10, t: 10, b: 60 },
                      paper_bgcolor: '#ffffff',
                      plot_bgcolor: '#ffffff',
                      font: { color: '#111827' },
                      xaxis: { tickfont: { size: 9 }, color: '#111827' },
                      yaxis: { title: 'JS', color: '#111827' },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    useResizeHandler
                    style={{ width: '100%', height: '220px' }}
                  />
                  <div className="mt-2 text-[11px] text-gray-700">
                    <p className="font-semibold">Top residues</p>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-1">
                      {topResidues.map(([v, label]) => (
                        <div key={label} className="flex items-center justify-between gap-2">
                          <span className="truncate">{label}</span>
                          <span className="font-mono">{Number(v).toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="rounded-md border border-gray-800 bg-white p-3">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <p className="text-xs font-semibold text-gray-800">Edge JS</p>
                    {edgeMatrixHasValues && (
                      <button
                        type="button"
                        className="text-[11px] text-gray-600 hover:text-gray-800"
                        onClick={() =>
                          setOverlayPlot({
                            title: 'Edge JS heatmap',
                          data: [
                              {
                                z: edgeMatrix,
                                x: residueLabels,
                                y: residueLabels,
                                type: 'heatmap',
                                colorscale: 'Viridis',
                                zmin: 0,
                                hovertemplate: 'x: %{x}<br>y: %{y}<br>JS: %{z:.4f}<extra></extra>',
                              },
                            ],
                            layout: {
                              margin: { l: 60, r: 10, t: 20, b: 60 },
                              paper_bgcolor: '#ffffff',
                              plot_bgcolor: '#ffffff',
                              font: { color: '#111827' },
                              xaxis: { title: 'Residue', color: '#111827' },
                              yaxis: { title: 'Residue', color: '#111827' },
                            },
                          })
                        }
                      >
                        Maximize
                      </button>
                    )}
                  </div>

                  {!edgeMatrixHasValues && (
                    <p className="text-[11px] text-gray-600">
                      {!selectedModelId
                        ? 'Select a Potts model to compute edge metrics.'
                        : !edges.length
                          ? 'Selected Potts model has no edges.'
                          : 'No edge JS available yet. Run analysis for this model.'}
                    </p>
                  )}
                  {edgeMatrixHasValues && (
                    <Plot
                      data={[
                        {
                          z: edgeMatrix,
                          x: residueLabels,
                          y: residueLabels,
                          type: 'heatmap',
                          colorscale: 'Viridis',
                          zmin: 0,
                          hovertemplate: 'x: %{x}<br>y: %{y}<br>JS: %{z:.4f}<extra></extra>',
                        },
                      ]}
                      layout={{
                        margin: { l: 50, r: 10, t: 10, b: 50 },
                        paper_bgcolor: '#ffffff',
                        plot_bgcolor: '#ffffff',
                        font: { color: '#111827' },
                      }}
                      config={{ displayModeBar: false, responsive: true }}
                      useResizeHandler
                      style={{ width: '100%', height: '220px' }}
                    />
                  )}
                  {!!topEdges.length && (
                    <div className="mt-2 text-[11px] text-gray-700">
                      <p className="font-semibold">Top edges</p>
                      <div className="space-y-1 mt-1">
                        {topEdges.map(([v, label]) => (
                          <div key={label} className="flex items-center justify-between gap-2">
                            <span className="truncate">{label}</span>
                            <span className="font-mono">{Number(v).toFixed(4)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </section>

          <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-2">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-gray-200">Energies</h2>
                <p className="text-[11px] text-gray-500">
                  Energy distributions are computed on-demand for all samples under the selected model.
                </p>
              </div>
              <div className="text-[11px] text-gray-500">
                {energyAnalysesForModel.length ? `${energyAnalysesForModel.length} analyses` : 'no analyses'}
              </div>
            </div>

            {!selectedModelId && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                Select a Potts model to compute energies.
              </div>
            )}
            {!!selectedModelId && !energyAnalysesForModel.length && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                No energy analyses found for this model/settings. Click <span className="font-semibold">Run analysis</span>{' '}
                to compute them.
              </div>
            )}
            {energyError && <ErrorMessage message={energyError} />}
            {energyLoading && <p className="text-sm text-gray-400">Loading…</p>}

            {energyPlot && (
              <div className="rounded-md border border-gray-800 bg-white p-3">
                <div className="flex items-center justify-between gap-2 mb-2">
                  <p className="text-xs font-semibold text-gray-800">Energy histograms</p>
                  <button
                    type="button"
                    className="text-[11px] text-gray-600 hover:text-gray-800"
                    onClick={() => setOverlayPlot({ ...energyPlot, title: 'Energy histograms (overlay)' })}
                  >
                    Maximize
                  </button>
                </div>
                <Plot
                  data={energyPlot.data}
                  layout={energyPlot.layout}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '260px' }}
                />
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
