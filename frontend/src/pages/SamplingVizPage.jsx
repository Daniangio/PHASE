import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, Info, Play, RefreshCw, Trash2, X } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import EnergyDistributionPlot, {
  buildEnergyDistributionPlot,
} from '../components/common/EnergyDistributionPlot';
import {
  deleteClusterAnalysis,
  deleteSamplingSample,
  fetchClusterAnalyses,
  fetchClusterAnalysisData,
  fetchPottsClusterInfo,
  fetchSampleStats,
  fetchSystem,
} from '../api/projects';
import { fetchJobStatus, submitMdSamplesRefreshJob, submitPottsAnalysisJob } from '../api/jobs';

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
        {sample.series_kind && (
          <div>
            <span className="text-gray-400">series:</span> {sample.series_kind}
          </div>
        )}
        {sample.series_id && (
          <div className="break-all">
            <span className="text-gray-400">series_id:</span> {sample.series_id}
          </div>
        )}
        {typeof sample.lambda === 'number' && Number.isFinite(sample.lambda) && (
          <div>
            <span className="text-gray-400">lambda:</span> {sample.lambda.toFixed(3)}
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

function AnalysisInfoPanel({ group, analyses, onClose }) {
  if (!group) return null;
  const rows = (analyses || []).filter((analysis) => analysis.model_id === group.modelId);
  return (
    <div className="rounded-md border border-gray-800 bg-gray-950/70 p-3 text-[11px] text-gray-300 space-y-2">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-xs font-semibold text-white break-words">{group.modelName || group.modelId}</p>
          <p className="text-[10px] text-gray-500 break-all">{group.modelId}</p>
        </div>
        <button type="button" onClick={onClose} className="text-gray-400 hover:text-gray-100" aria-label="Close analysis info">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div><span className="text-gray-500">MD/sample:</span> {group.mdVsCount}</div>
        <div><span className="text-gray-500">Energies:</span> {group.energyCount}</div>
        <div className="col-span-2"><span className="text-gray-500">latest:</span> {group.latestCreatedAt || 'unknown'}</div>
      </div>
      <details>
        <summary className="cursor-pointer text-gray-200">Stored analysis metadata</summary>
        <div className="mt-2 max-h-72 overflow-auto space-y-2">
          {rows.map((analysis) => (
            <pre key={`${analysis.analysis_type}:${analysis.analysis_id}`} className="rounded bg-gray-900 p-2 text-[10px] text-gray-300">
              {JSON.stringify(analysis, null, 2)}
            </pre>
          ))}
        </div>
      </details>
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
  const [, setAnalysisDataCache] = useState({});
  // Avoid effect dependency loops and duplicate network calls by using refs for cache + in-flight tracking.
  const analysisDataCacheRef = useRef({});
  const analysisDataInFlightRef = useRef({});

  const [selectedAnalysisModelId, setSelectedAnalysisModelId] = useState('');
  const [runAnalysisModelId, setRunAnalysisModelId] = useState('');
  const [selectedPoseStateId, setSelectedPoseStateId] = useState('');
  const [energyLoadLimit, setEnergyLoadLimit] = useState(1500);
  const [energyGraphMode, setEnergyGraphMode] = useState('histogram');
  const [analysisEdgeMode, setAnalysisEdgeMode] = useState('model');
  const [analysisContactCutoff, setAnalysisContactCutoff] = useState('10');
  const [analysisContactAtomMode, setAnalysisContactAtomMode] = useState('CA');
  const [edgeDisplayMode, setEdgeDisplayMode] = useState('all');

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
  const [helpOpen, setHelpOpen] = useState(false);
  const [runPanelOpen, setRunPanelOpen] = useState(false);
  const [activeExplorerTab, setActiveExplorerTab] = useState('js');
  const [activeJsSubtab, setActiveJsSubtab] = useState('nodes');
  const [selectedAnalysisModelIds, setSelectedAnalysisModelIds] = useState([]);
  const [analysisInfoModelId, setAnalysisInfoModelId] = useState('');
  const [pairSelections, setPairSelections] = useState({});
  const [comparisonDataMap, setComparisonDataMap] = useState({});

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
  const stateEntries = useMemo(() => Object.values(system?.states || {}), [system]);
  const mdSamples = useMemo(() => sampleEntries.filter((s) => s.type === 'md_eval'), [sampleEntries]);
  const poseEligibleStates = useMemo(
    () => stateEntries.filter((state) => state?.pdb_file && state?.descriptor_file),
    [stateEntries]
  );

  const analysisLinkedSampleIds = useMemo(() => {
    if (!selectedAnalysisModelId) return new Set();
    const linked = new Set();
    analyses.forEach((analysis) => {
      if (analysis?.model_id !== selectedAnalysisModelId) return;
      const sampleId = String(analysis?.sample_id || '').trim();
      if (sampleId) linked.add(sampleId);
    });
    return linked;
  }, [analyses, selectedAnalysisModelId]);

  const filteredSamples = useMemo(() => {
    if (!selectedAnalysisModelId) return sampleEntries;
    return sampleEntries.filter((s) => {
      if (s.type === 'md_eval') return true;
      if (s.type === 'potts_lambda_sweep') return true;
      if (analysisLinkedSampleIds.has(String(s.sample_id || ''))) return true;
      const ids = Array.isArray(s.model_ids) ? s.model_ids : s.model_id ? [s.model_id] : [];
      if (!ids.length) return true;
      return ids.includes(selectedAnalysisModelId);
    });
  }, [sampleEntries, selectedAnalysisModelId, analysisLinkedSampleIds]);
  const gibbsSamples = useMemo(
    () => filteredSamples.filter((s) => s.type === 'potts_sampling' && s.method === 'gibbs'),
    [filteredSamples]
  );
  const saSamples = useMemo(
    () => filteredSamples.filter((s) => s.type === 'potts_sampling' && s.method === 'sa'),
    [filteredSamples]
  );
  const pottsSamples = useMemo(() => [...gibbsSamples, ...saSamples], [gibbsSamples, saSamples]);
  const lambdaSweepSamples = useMemo(
    () => filteredSamples.filter((s) => s.type === 'potts_lambda_sweep'),
    [filteredSamples]
  );
  const selectableSamples = useMemo(() => [...pottsSamples, ...lambdaSweepSamples], [pottsSamples, lambdaSweepSamples]);

  const lambdaSweepSeries = useMemo(() => {
    const map = new Map();
    lambdaSweepSamples.forEach((s) => {
      const sid = s.series_id || 'unknown';
      if (!map.has(sid)) {
        map.set(sid, {
          series_id: sid,
          label: s.series_label || `Lambda sweep (${sid.slice(0, 8)})`,
          samples: [],
        });
      }
      map.get(sid).samples.push(s);
    });
    const out = Array.from(map.values());
    out.forEach((g) => {
      g.samples.sort((a, b) => {
        const la = typeof a.lambda === 'number' ? a.lambda : Number.POSITIVE_INFINITY;
        const lb = typeof b.lambda === 'number' ? b.lambda : Number.POSITIVE_INFINITY;
        return la - lb;
      });
    });
    out.sort((a, b) => String(a.label).localeCompare(String(b.label)));
    return out;
  }, [lambdaSweepSamples]);

  const infoSample = useMemo(() => sampleEntries.find((s) => s.sample_id === infoSampleId) || null, [sampleEntries, infoSampleId]);
  const infoSampleStats = useMemo(() => (infoSampleId ? sampleStatsCache[infoSampleId] : null), [sampleStatsCache, infoSampleId]);
  const analysisSummary = analysisJobStatus?.result?.results?.summary || analysisJobStatus?.meta?.summary || null;
  const analysisSkippedSamples = useMemo(
    () => (Array.isArray(analysisSummary?.skipped_samples) ? analysisSummary.skipped_samples : []),
    [analysisSummary]
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
      setSelectedAnalysisModelId('');
      setRunAnalysisModelId('');
      return;
    }
    if (!runAnalysisModelId) {
      setRunAnalysisModelId(pottsModels[0]?.model_id || '');
    } else if (!pottsModels.some((m) => m.model_id === runAnalysisModelId)) {
      setRunAnalysisModelId(pottsModels[0]?.model_id || '');
    }
  }, [pottsModels, runAnalysisModelId]);

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
      const [mdVsData, energyData] = await Promise.all([
        fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'md_vs_sample' }),
        fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'model_energy' }),
      ]);
      const merged = [
        ...(Array.isArray(mdVsData?.analyses) ? mdVsData.analyses : []),
        ...(Array.isArray(energyData?.analyses) ? energyData.analyses : []),
      ];
      merged.sort((a, b) => String(b?.created_at || '').localeCompare(String(a?.created_at || '')));
      setAnalyses(merged);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
    } finally {
      setAnalysesLoading(false);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    loadClusterInfo('');
    loadAnalyses();
    setAnalysisDataCache({});
    analysisDataCacheRef.current = {};
    analysisDataInFlightRef.current = {};
    setSelectedMdSampleId('');
    setInfoSampleId('');
  }, [selectedClusterId, loadClusterInfo, loadAnalyses]);

  useEffect(() => {
    if (!selectedClusterId) return;
    // Update edge count/info when switching the active Potts model.
    loadClusterInfo(selectedAnalysisModelId);
  }, [selectedAnalysisModelId, selectedClusterId, loadClusterInfo]);

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
    if (!selectableSamples.length) {
      setSelectedSampleId('');
      return;
    }
    if (!selectedSampleId || !selectableSamples.some((s) => s.sample_id === selectedSampleId)) {
      setSelectedSampleId(selectableSamples[0].sample_id);
    }
  }, [selectableSamples, selectedSampleId]);

  const mdVsSampleAnalyses = useMemo(
    () => analyses.filter((a) => a.analysis_type === 'md_vs_sample'),
    [analyses]
  );
  const modelEnergyAnalyses = useMemo(
    () => analyses.filter((a) => a.analysis_type === 'model_energy'),
    [analyses]
  );

  const analysisGroups = useMemo(() => {
    const byModel = new Map();
    const modelNameById = new Map((pottsModels || []).map((m) => [m.model_id, m.name || m.model_id]));
    [...mdVsSampleAnalyses, ...modelEnergyAnalyses].forEach((analysis) => {
      const modelId = String(analysis.model_id || '').trim();
      if (!modelId) return;
      const key = modelId;
      if (!byModel.has(key)) {
        byModel.set(key, {
          modelId,
          modelName: analysis.model_name || modelNameById.get(modelId) || modelId,
          latestCreatedAt: String(analysis.created_at || ''),
          mdVsCount: 0,
          energyCount: 0,
        });
      }
      const group = byModel.get(key);
      if (analysis.analysis_type === 'md_vs_sample') group.mdVsCount += 1;
      if (analysis.analysis_type === 'model_energy') group.energyCount += 1;
      const createdAt = String(analysis.created_at || '');
      if (createdAt > String(group.latestCreatedAt || '')) {
        group.latestCreatedAt = createdAt;
      }
      if (!group.modelName && (analysis.model_name || modelNameById.get(modelId))) {
        group.modelName = analysis.model_name || modelNameById.get(modelId) || modelId;
      }
    });
    return Array.from(byModel.values()).sort((a, b) => String(b.latestCreatedAt || '').localeCompare(String(a.latestCreatedAt || '')));
  }, [mdVsSampleAnalyses, modelEnergyAnalyses, pottsModels]);

  const pendingAnalysisEntry = useMemo(() => {
    const modelId = String(analysisJob?.model_id || '').trim();
    if (!modelId || !selectedClusterId) return null;
    const modelName =
      pottsModels.find((m) => m.model_id === modelId)?.name ||
      analysisGroups.find((g) => g.modelId === modelId)?.modelName ||
      modelId;
    const progress = typeof analysisJobStatus?.meta?.progress === 'number' ? analysisJobStatus.meta.progress : 0;
    const status = String(analysisJobStatus?.status || analysisJobStatus?.meta?.status || 'queued');
    return {
      modelId,
      modelName,
      progress,
      status,
      jobId: analysisJob?.job_id || '',
    };
  }, [analysisGroups, analysisJob, analysisJobStatus, pottsModels, selectedClusterId]);

  const analysisGroupMap = useMemo(() => {
    const map = new Map();
    analysisGroups.forEach((group) => map.set(group.modelId, group));
    return map;
  }, [analysisGroups]);

  const selectedAnalysisGroups = useMemo(
    () => selectedAnalysisModelIds.map((id) => analysisGroupMap.get(id)).filter(Boolean),
    [analysisGroupMap, selectedAnalysisModelIds]
  );

  useEffect(() => {
    if (!analysisGroups.length) {
      setSelectedAnalysisModelId('');
      setSelectedAnalysisModelIds([]);
      return;
    }
    setSelectedAnalysisModelIds((prev) => {
      const allowed = new Set(analysisGroups.map((g) => g.modelId));
      const retained = prev.filter((id) => allowed.has(id));
      return retained.length ? retained : [analysisGroups[0].modelId];
    });
    if (!selectedAnalysisModelId || !analysisGroups.some((g) => g.modelId === selectedAnalysisModelId)) {
      setSelectedAnalysisModelId(analysisGroups[0].modelId);
    }
  }, [analysisGroups, selectedAnalysisModelId]);

  useEffect(() => {
    const first = selectedAnalysisModelIds[0] || '';
    if (first && first !== selectedAnalysisModelId) setSelectedAnalysisModelId(first);
  }, [selectedAnalysisModelIds, selectedAnalysisModelId]);

  useEffect(() => {
    if (!poseEligibleStates.length) {
      setSelectedPoseStateId('');
      return;
    }
    if (!selectedPoseStateId || !poseEligibleStates.some((s) => s.state_id === selectedPoseStateId)) {
      setSelectedPoseStateId(poseEligibleStates[0].state_id);
    }
  }, [poseEligibleStates, selectedPoseStateId]);

  const mdLabelMode = 'assigned';
  const dropInvalid = true;

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
    if (selectedAnalysisModelId) {
      const withModel = candidates.find((a) => a.model_id === selectedAnalysisModelId);
      if (withModel) return withModel;
    }
    return candidates[0];
  }, [mdVsSampleAnalyses, selectedMdSampleId, selectedSampleId, selectedAnalysisModelId, mdLabelMode, dropInvalid]);

  const findMdVsMeta = useCallback(
    (modelId, mdSampleId, sampleId) => {
      if (!modelId || !mdSampleId || !sampleId) return null;
      return mdVsSampleAnalyses.find((a) => {
        const mode = (a.md_label_mode || 'assigned').toLowerCase();
        return (
          a.model_id === modelId &&
          a.md_sample_id === mdSampleId &&
          a.sample_id === sampleId &&
          mode === mdLabelMode &&
          Boolean(a.drop_invalid) === Boolean(dropInvalid)
        );
      }) || null;
    },
    [mdVsSampleAnalyses, mdLabelMode, dropInvalid]
  );

  const makeDefaultPair = useCallback(
    (modelId) => {
      const modelAnalyses = mdVsSampleAnalyses.filter((a) => a.model_id === modelId);
      const first = modelAnalyses[0];
      return {
        id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        mdSampleId: first?.md_sample_id || mdSamples[0]?.sample_id || '',
        sampleId: first?.sample_id || selectableSamples[0]?.sample_id || '',
      };
    },
    [mdSamples, mdVsSampleAnalyses, selectableSamples]
  );

  useEffect(() => {
    setPairSelections((prev) => {
      const selected = new Set(selectedAnalysisModelIds);
      const next = {};
      selectedAnalysisModelIds.forEach((modelId) => {
        const retained = Array.isArray(prev[modelId]) ? prev[modelId].filter((pair) => {
          const mdOk = mdSamples.some((s) => s.sample_id === pair.mdSampleId);
          const sampleOk = selectableSamples.some((s) => s.sample_id === pair.sampleId);
          return mdOk && sampleOk;
        }) : [];
        next[modelId] = retained.length ? retained : [makeDefaultPair(modelId)];
      });
      Object.keys(prev).forEach((modelId) => {
        if (!selected.has(modelId)) return;
        if (!next[modelId]?.length) next[modelId] = [makeDefaultPair(modelId)];
      });
      return next;
    });
  }, [selectedAnalysisModelIds, mdSamples, selectableSamples, makeDefaultPair]);

  const jsComparisonItems = useMemo(() => {
    const items = [];
    selectedAnalysisModelIds.forEach((modelId) => {
      const group = analysisGroupMap.get(modelId);
      const pairs = Array.isArray(pairSelections[modelId]) ? pairSelections[modelId] : [];
      pairs.forEach((pair, pairIndex) => {
        const meta = findMdVsMeta(modelId, pair.mdSampleId, pair.sampleId);
        const md = mdSamples.find((s) => s.sample_id === pair.mdSampleId);
        const sample = sampleEntries.find((s) => s.sample_id === pair.sampleId);
        const label = `${group?.modelName || modelId}: ${md?.name || pair.mdSampleId} vs ${sample?.name || pair.sampleId}`;
        items.push({
          key: `${modelId}:${pair.id || pairIndex}`,
          modelId,
          modelName: group?.modelName || modelId,
          pair,
          pairIndex,
          meta,
          mdName: md?.name || pair.mdSampleId,
          sampleName: sample?.name || pair.sampleId,
          label,
        });
      });
    });
    return items;
  }, [analysisGroupMap, findMdVsMeta, mdSamples, pairSelections, sampleEntries, selectedAnalysisModelIds]);

  const loadAnalysisData = useCallback(
    async (analysisType, analysisId, options = {}) => {
      if (!analysisType || !analysisId) return null;
      const maxRowsKey = options?.maxRows != null ? `:maxRows=${Number(options.maxRows)}` : '';
      const summaryOnlyKey = options?.summaryOnly ? ':summaryOnly=1' : '';
      const sampleSeedKey = options?.sampleSeed != null ? `:seed=${Number(options.sampleSeed)}` : '';
      const cacheKey = `${analysisType}:${analysisId}${maxRowsKey}${summaryOnlyKey}${sampleSeedKey}`;
      const cached = analysisDataCacheRef.current;
      if (Object.prototype.hasOwnProperty.call(cached, cacheKey)) return cached[cacheKey];

      const inflight = analysisDataInFlightRef.current;
      if (inflight[cacheKey]) return inflight[cacheKey];

      const p = fetchClusterAnalysisData(projectId, systemId, selectedClusterId, analysisType, analysisId, options)
        .then((payload) => {
          analysisDataCacheRef.current = { ...analysisDataCacheRef.current, [cacheKey]: payload };
          setAnalysisDataCache((prev) => ({ ...prev, [cacheKey]: payload }));
          delete analysisDataInFlightRef.current[cacheKey];
          return payload;
        })
        .catch((err) => {
          delete analysisDataInFlightRef.current[cacheKey];
          throw err;
        });
      inflight[cacheKey] = p;
      return p;
    },
    [projectId, systemId, selectedClusterId]
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

  useEffect(() => {
    const run = async () => {
      const loadable = jsComparisonItems.filter((item) => item.meta?.analysis_id);
      if (!loadable.length) {
        setComparisonDataMap({});
        return;
      }
      setComparisonLoading(true);
      setComparisonError(null);
      try {
        const entries = await Promise.all(
          loadable.map(async (item) => {
            const payload = await loadAnalysisData('md_vs_sample', item.meta.analysis_id);
            return [item.key, payload];
          })
        );
        setComparisonDataMap(Object.fromEntries(entries));
      } catch (err) {
        setComparisonError(err.message || 'Failed to load selected JS analyses.');
      } finally {
        setComparisonLoading(false);
      }
    };
    run();
  }, [jsComparisonItems, loadAnalysisData]);

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

  const modelEdgeSet = useMemo(() => {
    const out = new Set();
    (clusterInfo?.edges || []).forEach((edge) => {
      if (!Array.isArray(edge) || edge.length < 2) return;
      const a = Number(edge[0]);
      const b = Number(edge[1]);
      if (!Number.isFinite(a) || !Number.isFinite(b)) return;
      const r = Math.min(a, b);
      const s = Math.max(a, b);
      out.add(`${r}-${s}`);
    });
    return out;
  }, [clusterInfo]);

  const jsLoadedItems = useMemo(() => {
    return jsComparisonItems
      .map((item, idx) => {
        const payload = comparisonDataMap[item.key];
        const data = payload?.data || {};
        const rawEdges = Array.isArray(data.edges) && data.edges.length ? data.edges : edges;
        const itemNodeJs = Array.isArray(data.node_js) ? data.node_js : [];
        const rawEdgeJs = Array.isArray(data.edge_js) ? data.edge_js : [];
        const itemEdges = [];
        const itemEdgeJs = [];
        rawEdges.forEach((edge, edgeIdx) => {
          if (!Array.isArray(edge) || edge.length < 2) return;
          const a = Number(edge[0]);
          const b = Number(edge[1]);
          const key = `${Math.min(a, b)}-${Math.max(a, b)}`;
          const isInModel = modelEdgeSet.has(key);
          if (
            edgeDisplayMode === 'all' ||
            (edgeDisplayMode === 'within_model' && isInModel) ||
            (edgeDisplayMode === 'over_model' && !isInModel)
          ) {
            itemEdges.push(edge);
            itemEdgeJs.push(rawEdgeJs[edgeIdx]);
          }
        });
        return {
          ...item,
          color: ['#0891b2', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#4f46e5', '#be123c'][idx % 7],
          data,
          edges: itemEdges,
          nodeJs: itemNodeJs,
          edgeJs: itemEdgeJs,
          edgeMatrix: buildEdgeMatrix(residueLabels.length, itemEdges, itemEdgeJs),
        };
      })
      .filter((item) => item.meta?.analysis_id && (item.nodeJs.length || item.edgeJs.length));
  }, [comparisonDataMap, edgeDisplayMode, edges, jsComparisonItems, modelEdgeSet, residueLabels.length]);

  const globalNodeJsMax = useMemo(() => {
    const vals = jsLoadedItems.flatMap((item) => item.nodeJs).map(Number).filter(Number.isFinite);
    return vals.length ? Math.max(...vals, 1e-9) : 1;
  }, [jsLoadedItems]);

  const globalEdgeJsMax = useMemo(() => {
    const vals = jsLoadedItems.flatMap((item) => item.edgeJs).map(Number).filter(Number.isFinite);
    return vals.length ? Math.max(...vals, 1e-9) : 1;
  }, [jsLoadedItems]);

  const multiNodePlot = useMemo(() => {
    if (!jsLoadedItems.length) return null;
    return {
      data: jsLoadedItems.map((item) => ({
        x: residueLabels,
        y: item.nodeJs,
        type: 'bar',
        name: item.label,
        marker: { color: item.color },
        hovertemplate: `${item.label}<br>%{x}<br>JS: %{y:.4f}<extra></extra>`,
      })),
      layout: {
        barmode: 'group',
        margin: { l: 55, r: 20, t: 20, b: 100 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#111827' },
        xaxis: { tickangle: -45, automargin: true, tickfont: { size: 9 } },
        yaxis: { title: 'Node JS', range: [0, globalNodeJsMax * 1.05], color: '#111827' },
        legend: { orientation: 'h', y: -0.28 },
      },
      config: { displayModeBar: false, responsive: true },
    };
  }, [globalNodeJsMax, jsLoadedItems, residueLabels]);

  const multiTopResidues = useMemo(() => {
    const rows = [];
    jsLoadedItems.forEach((item) => {
      item.nodeJs.forEach((value, idx) => {
        if (Number.isFinite(Number(value))) {
          rows.push({ label: residueLabels[idx] || String(idx), value: Number(value), item: item.label });
        }
      });
    });
    rows.sort((a, b) => b.value - a.value);
    return rows.slice(0, 20);
  }, [jsLoadedItems, residueLabels]);

  const multiTopEdges = useMemo(() => {
    const rows = [];
    jsLoadedItems.forEach((item) => {
      item.edges.forEach((edge, idx) => {
        const value = Number(item.edgeJs[idx]);
        if (!Number.isFinite(value)) return;
        rows.push({
          label: `${residueLabels[edge[0]] ?? edge[0]} — ${residueLabels[edge[1]] ?? edge[1]}`,
          value,
          item: item.label,
        });
      });
    });
    rows.sort((a, b) => b.value - a.value);
    return rows.slice(0, 20);
  }, [jsLoadedItems, residueLabels]);

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
        md_label_mode: 'assigned',
        keep_invalid: false,
        analysis_edge_mode: analysisEdgeMode || 'model',
      };
      if ((analysisEdgeMode || 'model') === 'contact') {
        const parsedCutoff = Number(analysisContactCutoff);
        if (Number.isFinite(parsedCutoff) && parsedCutoff > 0) payload.analysis_contact_cutoff = parsedCutoff;
        payload.analysis_contact_atom_mode = analysisContactAtomMode || 'CA';
      }
      if (runAnalysisModelId) payload.model_id = runAnalysisModelId;
      const res = await submitPottsAnalysisJob(payload);
      setAnalysisJob({ ...res, model_id: runAnalysisModelId });
      if (runAnalysisModelId) setSelectedAnalysisModelId(runAnalysisModelId);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to submit analysis job.');
    }
  }, [
    projectId,
    systemId,
    selectedClusterId,
    runAnalysisModelId,
    analysisEdgeMode,
    analysisContactCutoff,
    analysisContactAtomMode,
  ]);

  const handleAppendStatePoseEnergy = useCallback(async () => {
    if (!selectedClusterId || !selectedAnalysisModelId || !selectedPoseStateId) return;
    setAnalysesError(null);
    setAnalysisJob(null);
    setAnalysisJobStatus(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        model_id: selectedAnalysisModelId,
        pose_only: true,
        state_pose_ids: [selectedPoseStateId],
      };
      const res = await submitPottsAnalysisJob(payload);
      setAnalysisJob({ ...res, model_id: selectedAnalysisModelId });
    } catch (err) {
      setAnalysesError(err.message || 'Failed to append state pose energy.');
    }
  }, [projectId, systemId, selectedClusterId, selectedAnalysisModelId, selectedPoseStateId]);

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
          if (status?.status === 'finished') {
            await loadAnalyses();
            const data = await fetchSystem(projectId, systemId);
            if (!cancelled) setSystem(data);
          }
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
  }, [analysisJob, loadAnalyses, projectId, systemId]);

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
        await loadAnalyses();
        setInfoSampleId('');
      } catch (err) {
        setSystemError(err.message || 'Failed to delete sample.');
      }
    },
    [projectId, systemId, selectedClusterId, loadAnalyses]
  );

  const toggleInfo = useCallback((sampleId) => {
    setInfoSampleId((prev) => (prev === sampleId ? '' : sampleId));
  }, []);

  const handleDeleteAnalysisGroup = useCallback(
    async (modelId) => {
      if (!selectedClusterId || !modelId) return;
      const targets = analyses.filter(
        (a) => a.model_id === modelId && (a.analysis_type === 'md_vs_sample' || a.analysis_type === 'model_energy')
      );
      if (!targets.length) return;
      const ok = window.confirm(`Delete ${targets.length} Sampling Explorer analyses for this Potts model?`);
      if (!ok) return;
      setAnalysesError(null);
      try {
        for (const analysis of targets) {
          try {
            await deleteClusterAnalysis(projectId, systemId, selectedClusterId, analysis.analysis_type, analysis.analysis_id);
          } catch (err) {
            const message = String(err?.message || '');
            if (!/not found/i.test(message) && !/404/.test(message)) {
              throw err;
            }
          }
        }
        if (selectedAnalysisModelId === modelId) setSelectedAnalysisModelId('');
        await loadAnalyses();
      } catch (err) {
        setAnalysesError(err.message || 'Failed to delete analysis group.');
      }
    },
    [analyses, loadAnalyses, projectId, selectedAnalysisModelId, selectedClusterId, systemId]
  );

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

  useEffect(() => {
    const load = async () => {
      if (!selectedSampleId) return;
      if (sampleStatsCache[selectedSampleId]) return;
      try {
        const stats = await fetchSampleStats(projectId, systemId, selectedClusterId, selectedSampleId);
        setSampleStatsCache((prev) => ({ ...prev, [selectedSampleId]: stats }));
      } catch (err) {
        setSampleStatsError(err.message || 'Failed to load sample stats.');
      }
    };
    load();
  }, [selectedSampleId, sampleStatsCache, projectId, systemId, selectedClusterId]);

  const baseEnergyAnalysesForModel = useMemo(() => {
    if (!selectedAnalysisModelId) return [];
    return modelEnergyAnalyses.filter(
      (a) =>
        a.model_id === selectedAnalysisModelId &&
        String(a.sample_type || '').toLowerCase() !== 'state_pose' &&
        String(a.sample_type || '').toLowerCase() !== 'state_eval'
    );
  }, [modelEnergyAnalyses, selectedAnalysisModelId]);

  const [energyGraphs, setEnergyGraphs] = useState([]);
  const [energyError, setEnergyError] = useState(null);
  const [energyLoading, setEnergyLoading] = useState(false);

  useEffect(() => {
    const run = async () => {
      setEnergyError(null);
      setEnergyGraphs([]);
      if (!selectedAnalysisModelIds.length) return;
      setEnergyLoading(true);
      try {
        const graphs = [];
        for (const group of selectedAnalysisGroups) {
          const metas = [];
          const seen = new Set();
          for (const a of modelEnergyAnalyses) {
            if (a.model_id !== group.modelId) continue;
            const sampleType = String(a.sample_type || '').toLowerCase();
            const isStateDerived = sampleType === 'state_pose' || sampleType === 'state_eval';
            if (!isStateDerived) {
              if ((a.md_label_mode || 'assigned') !== mdLabelMode) continue;
              if (Boolean(a.drop_invalid) !== Boolean(dropInvalid)) continue;
            }
            const sid = a.sample_id || '';
            if (!sid || seen.has(sid)) continue;
            seen.add(sid);
            metas.push(a);
          }
          const series = [];
          for (const meta of metas) {
            const payload = await loadAnalysisData('model_energy', meta.analysis_id, {
              maxRows: energyLoadLimit > 0 ? energyLoadLimit : undefined,
              sampleSeed: 0,
            });
            const energies = payload?.data?.energies || [];
            if (!Array.isArray(energies) || !energies.length) continue;
            const sample = sampleEntries.find((s) => s.sample_id === meta.sample_id);
            const sampleType = String(meta.sample_type || sample?.type || '').toLowerCase();
            series.push({
              id: `${group.modelId}:${meta.sample_id}`,
              sample_id: meta.sample_id,
              label: sample?.name || meta.sample_name || meta.sample_id,
              kind: sampleType === 'state_pose' ? 'state_pose' : sampleType || 'sample',
              type: sampleType || 'sample',
              values: energies,
            });
          }
          graphs.push({ modelId: group.modelId, modelName: group.modelName, series });
        }
        setEnergyGraphs(graphs);
      } catch (err) {
        setEnergyError(err.message || 'Failed to load selected model energies.');
      } finally {
        setEnergyLoading(false);
      }
    };
    run();
  }, [
    selectedAnalysisModelIds,
    selectedAnalysisGroups,
    modelEnergyAnalyses,
    mdLabelMode,
    dropInvalid,
    loadAnalysisData,
    energyLoadLimit,
    sampleEntries,
  ]);

  const globalEnergyRange = useMemo(() => {
    const vals = energyGraphs.flatMap((graph) => graph.series.flatMap((series) => series.values || [])).map(Number).filter(Number.isFinite);
    if (!vals.length) return null;
    return [Math.min(...vals), Math.max(...vals)];
  }, [energyGraphs]);

  if (loadingSystem) return <Loader message="Loading sampling explorer..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <PlotOverlay overlay={overlayPlot} onClose={() => setOverlayPlot(null)} />
      <HelpDrawer
        open={helpOpen}
        title="Sampling Explorer: How To Read The Plots"
        docPath="/docs/sampling_viz_help.md"
        onClose={() => setHelpOpen(false)}
      />
      {runPanelOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-lg border border-gray-700 bg-gray-950 p-4 shadow-xl">
            <div className="flex items-start justify-between gap-3 border-b border-gray-800 pb-3">
              <div>
                <h2 className="text-lg font-semibold text-white">Run Sampling Explorer analysis</h2>
                <p className="mt-1 text-xs text-gray-400">Computes MD-vs-sample JS metrics and model-energy distributions for the selected Potts model.</p>
              </div>
              <button type="button" onClick={() => setRunPanelOpen(false)} className="text-sm text-gray-400 hover:text-gray-100">Close</button>
            </div>
            <div className="space-y-3 pt-4">
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">Potts model</label>
                <select
                  value={runAnalysisModelId}
                  onChange={(e) => setRunAnalysisModelId(e.target.value)}
                  disabled={!pottsModels.length}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white disabled:opacity-60"
                >
                  {!pottsModels.length && <option value="">No models</option>}
                  {pottsModels.map((m) => (
                    <option key={m.model_id} value={m.model_id}>{m.name || m.model_id}</option>
                  ))}
                </select>
              </div>
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">Metric edge set</label>
                <select
                  value={analysisEdgeMode}
                  onChange={(e) => setAnalysisEdgeMode(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  <option value="model">Potts model edges</option>
                  <option value="cluster">Cluster edges</option>
                  <option value="contact">Contact edges (custom cutoff)</option>
                  <option value="all_vs_all">All residue pairs</option>
                </select>
              </div>
              {analysisEdgeMode === 'contact' && (
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="block text-xs text-gray-400">Contact cutoff (A)</label>
                    <input value={analysisContactCutoff} onChange={(e) => setAnalysisContactCutoff(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white" />
                  </div>
                  <div className="space-y-1">
                    <label className="block text-xs text-gray-400">Atom mode</label>
                    <select value={analysisContactAtomMode} onChange={(e) => setAnalysisContactAtomMode(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white">
                      <option value="CA">CA</option>
                      <option value="CM">CM</option>
                    </select>
                  </div>
                </div>
              )}
              <div className="rounded-md border border-gray-800 bg-gray-900/60 p-3 text-[11px] text-gray-400">
                MD label mode is fixed to assigned labels and invalid SA frames are dropped.
              </div>
              <button
                type="button"
                onClick={async () => {
                  await handleRunAnalysis();
                  setRunPanelOpen(false);
                }}
                disabled={!runAnalysisModelId}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white text-sm disabled:opacity-60 disabled:cursor-not-allowed"
              >
                <Play className="h-4 w-4" />
                Run analysis
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="flex items-start justify-between gap-3">
        <div>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)}
            className="mb-3 inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Back to system
          </button>
          <h1 className="text-2xl font-semibold text-white">Sampling Explorer</h1>
          <p className="text-sm text-gray-400">
            Sampling runs save only <code>sample.npz</code>. Use the analysis job to generate derived metrics under{' '}
            <code>clusters/&lt;cluster_id&gt;/analyses/</code>.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
          >
            <CircleHelp className="h-4 w-4" />
            Help
          </button>
          <button
            type="button"
            onClick={() => setRunPanelOpen(true)}
            className="text-xs px-3 py-2 rounded-md bg-cyan-600 text-white hover:bg-cyan-500 inline-flex items-center gap-2"
          >
            <Play className="h-4 w-4" />
            New analysis
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_eval`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Delta eval
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/lambda_sweep`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Lambda sweep
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/gibbs_relaxation`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Gibbs relaxation
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/potts_nn_mapping`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Nearest neighbours
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[340px_320px_minmax(0,1fr)] gap-4">
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

            <p className="text-[11px] text-gray-500">Analysis mode: assigned labels only. Invalid SA frames are always dropped.</p>

            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs font-semibold text-gray-300">Analyses</p>
                {analysesLoading && <span className="text-[11px] text-gray-500">Loading...</span>}
                <button
                  type="button"
                  onClick={async () => {
                    await loadClusterInfo(selectedAnalysisModelId);
                    await loadAnalyses();
                  }}
                  className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-gray-700 text-gray-200 text-[11px] hover:border-gray-500"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  Refresh
                </button>
              </div>
              <div className="space-y-2 rounded-md border border-gray-800 bg-gray-950/40 p-2">
                {!analysisGroups.length && !pendingAnalysisEntry && (
                  <p className="text-[11px] text-gray-500">No Potts analyses yet.</p>
                )}
                {analysisGroups.map((group) => {
                  const isSelected = selectedAnalysisModelIds.includes(group.modelId);
                  const isPending = pendingAnalysisEntry?.modelId === group.modelId;
                  const progress = isPending ? pendingAnalysisEntry.progress : null;
                  const status = isPending ? pendingAnalysisEntry.status : null;
                  return (
                    <div
                      key={group.modelId}
                      className={`w-full rounded-md border px-3 py-2 ${
                        isSelected
                          ? 'border-cyan-500 bg-cyan-950/30'
                          : 'border-gray-800 bg-gray-900/40 hover:border-gray-600'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedAnalysisModelIds((prev) => (
                              prev.includes(group.modelId)
                                ? prev.filter((id) => id !== group.modelId)
                                : [...prev, group.modelId]
                            ));
                          }}
                          className="min-w-0 flex-1 text-left"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-sm text-white truncate">
                              <input
                                type="checkbox"
                                readOnly
                                checked={selectedAnalysisModelIds.includes(group.modelId)}
                                className="mr-2 align-middle"
                              />
                              {group.modelName}
                            </span>
                            <span className="text-[10px] text-gray-500 whitespace-nowrap">
                              {group.mdVsCount} compare · {group.energyCount} energy
                            </span>
                          </div>
                          <div className="text-[10px] text-gray-500 mt-0.5 font-mono">
                            {String(group.modelId || '').slice(0, 8)}
                          </div>
                        </button>
                        <button
                          type="button"
                          onClick={() => setAnalysisInfoModelId((prev) => (prev === group.modelId ? '' : group.modelId))}
                          className="text-gray-400 hover:text-cyan-200"
                          title="Show analysis metadata"
                          aria-label={`Show metadata for ${group.modelName}`}
                        >
                          <Info className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeleteAnalysisGroup(group.modelId)}
                          className="text-gray-400 hover:text-red-300"
                          title="Delete this model analysis group"
                          aria-label={`Delete analyses for ${group.modelName}`}
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                      {isPending && (
                        <div className="mt-2 space-y-1">
                          <div className="h-1.5 rounded bg-gray-800 overflow-hidden">
                            <div
                              className="h-full bg-cyan-500"
                              style={{ width: `${Math.max(0, Math.min(100, Number(progress) || 0))}%` }}
                            />
                          </div>
                          <p className="text-[10px] text-cyan-200">
                            {status || 'running'} · {Math.max(0, Math.min(100, Number(progress) || 0))}%
                          </p>
                        </div>
                      )}
                      {analysisInfoModelId === group.modelId && (
                        <div className="mt-2">
                          <AnalysisInfoPanel
                            group={group}
                            analyses={analyses}
                            onClose={() => setAnalysisInfoModelId('')}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
                {!!pendingAnalysisEntry && !analysisGroups.some((g) => g.modelId === pendingAnalysisEntry.modelId) && (
                  <div className="rounded-md border border-cyan-800 bg-cyan-950/20 px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm text-white truncate">{pendingAnalysisEntry.modelName}</span>
                      <span className="text-[10px] text-cyan-200">running</span>
                    </div>
                    <div className="mt-2 h-1.5 rounded bg-gray-800 overflow-hidden">
                      <div
                        className="h-full bg-cyan-500"
                        style={{ width: `${Math.max(0, Math.min(100, Number(pendingAnalysisEntry.progress) || 0))}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-2 rounded-md border border-gray-800 bg-gray-950/40 p-3">
              <p className="text-xs font-semibold text-gray-300">Load limits</p>
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">Max energy points per sample</label>
                <select
                  value={String(energyLoadLimit)}
                  onChange={(e) => {
                    const next = Number(e.target.value);
                    setEnergyLoadLimit(Number.isFinite(next) ? next : 1500);
                  }}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  <option value="500">500</option>
                  <option value="1000">1000</option>
                  <option value="1500">1500</option>
                  <option value="3000">3000</option>
                  <option value="5000">5000</option>
                  <option value="0">All</option>
                </select>
              </div>
              <p className="text-[11px] text-gray-500">
                Energy histograms can require large payloads because each analysis stores one energy per frame. By default, the page loads a
                random server-side subset of 1500 energies per sample.
              </p>
            </div>

            <div className="space-y-2 rounded-md border border-gray-800 bg-gray-950/40 p-3">
              <p className="text-xs font-semibold text-gray-300">Edge Display Filter</p>
              <select
                value={edgeDisplayMode}
                onChange={(e) => setEdgeDisplayMode(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                <option value="all">All analysis edges</option>
                <option value="within_model">Only within model cutoff</option>
                <option value="over_model">Only over model cutoff</option>
              </select>
              <p className="text-[11px] text-gray-500">
                Uses current Potts model edges as the cutoff mask: edges in model = within cutoff, others = over cutoff.
              </p>
            </div>

            <div className="space-y-2 rounded-md border border-gray-800 bg-gray-950/40 p-3">
              <p className="text-xs font-semibold text-gray-300">State energy</p>
              <p className="text-[11px] text-gray-500">
                Use a descriptor-ready state and place its Potts energy on the current histogram. If the state is not yet materialized as an assigned cluster sample, PHASE will evaluate it once and reuse that assignment.
              </p>
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">State</label>
                <select
                  value={selectedPoseStateId}
                  onChange={(e) => setSelectedPoseStateId(e.target.value)}
                  disabled={!poseEligibleStates.length}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white disabled:opacity-60"
                >
                  {!poseEligibleStates.length && <option value="">No descriptor-ready states available</option>}
                  {poseEligibleStates.map((state) => (
                    <option key={state.state_id} value={state.state_id}>
                      {state.name || state.state_id}
                    </option>
                  ))}
                </select>
              </div>
              {!selectedAnalysisModelId && (
                <p className="text-[11px] text-amber-300">Select an analysis in the sidebar first.</p>
              )}
              {!!selectedAnalysisModelId && !baseEnergyAnalysesForModel.length && (
                <p className="text-[11px] text-amber-300">
                  Run the complete Potts analysis for this model first. The state energy is appended to that model context.
                </p>
              )}
              <button
                type="button"
                onClick={handleAppendStatePoseEnergy}
                disabled={!selectedAnalysisModelId || !selectedPoseStateId || !baseEnergyAnalysesForModel.length}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-gray-700 hover:bg-gray-600 text-white text-sm disabled:opacity-60 disabled:cursor-not-allowed"
              >
                <Play className="h-4 w-4" />
                Add energy
              </button>
            </div>

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

            <div>
              <p className="text-xs font-semibold text-gray-300">Lambda sweeps</p>
              {lambdaSweepSamples.length === 0 && (
                <p className="text-[11px] text-gray-500 mt-1">No lambda sweep samples yet.</p>
              )}
              {lambdaSweepSeries.length > 0 && (
                <div className="space-y-2 mt-2">
                  {lambdaSweepSeries.map((group) => (
                    <div key={group.series_id} className="rounded-md border border-gray-800 bg-gray-950/30 p-2">
                      <p className="text-[11px] text-gray-200 font-semibold">{group.label}</p>
                      <p className="text-[10px] text-gray-500">{group.samples.length} samples</p>
                      <div className="space-y-1 mt-2">
                        {group.samples.map((sample) => (
                          <div
                            key={sample.sample_id || sample.path}
                            className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/40 px-2 py-1 text-[11px] text-gray-300"
                          >
                            <button
                              type="button"
                              onClick={() => setSelectedSampleId(sample.sample_id)}
                              className={`truncate text-left ${selectedSampleId === sample.sample_id ? 'text-cyan-200' : ''}`}
                            >
                              {typeof sample.lambda === 'number' && Number.isFinite(sample.lambda)
                                ? `λ=${sample.lambda.toFixed(3)}`
                                : sample.name || 'Lambda sample'}{' '}
                              • {sample.created_at || ''}
                            </button>
                            <div className="flex items-center gap-2">
                              <button
                                type="button"
                                onClick={() => toggleInfo(sample.sample_id)}
                                className="text-gray-400 hover:text-gray-200"
                                aria-label={`Show info for ${sample.name || 'lambda sweep sample'}`}
                              >
                                <Info className="h-4 w-4" />
                              </button>
                              <button
                                type="button"
                                onClick={() => handleDeleteSample(sample.sample_id)}
                                className="text-gray-400 hover:text-red-300"
                                aria-label={`Delete ${sample.name || 'lambda sweep sample'}`}
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            </div>
                          </div>
                        ))}
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

        <aside className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
            <div>
              <p className="text-xs font-semibold text-gray-300">JS pairs</p>
              <p className="mt-1 text-[11px] text-gray-500">
                For each selected model, choose one or more MD/sample pairs to display in the Node or Edge JS tab.
              </p>
            </div>
            {!selectedAnalysisGroups.length && (
              <p className="text-[11px] text-gray-500">Select at least one analysis on the left.</p>
            )}
            {selectedAnalysisGroups.map((group) => {
              const pairs = Array.isArray(pairSelections[group.modelId]) ? pairSelections[group.modelId] : [];
              return (
                <div key={`pairs:${group.modelId}`} className="rounded-md border border-gray-800 bg-gray-950/40 p-2 space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="min-w-0 truncate text-xs font-semibold text-gray-100">{group.modelName}</p>
                    <button
                      type="button"
                      onClick={() => setPairSelections((prev) => ({
                        ...prev,
                        [group.modelId]: [...(prev[group.modelId] || []), makeDefaultPair(group.modelId)],
                      }))}
                      className="rounded border border-gray-700 px-2 py-1 text-[11px] text-gray-200 hover:border-cyan-500"
                    >
                      Add pair
                    </button>
                  </div>
                  {pairs.map((pair, idx) => {
                    const meta = findMdVsMeta(group.modelId, pair.mdSampleId, pair.sampleId);
                    return (
                      <div key={pair.id || idx} className="space-y-1 rounded border border-gray-800 bg-gray-900/50 p-2">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[10px] uppercase tracking-wide text-gray-500">Pair {idx + 1}</span>
                          <button
                            type="button"
                            onClick={() => setPairSelections((prev) => ({
                              ...prev,
                              [group.modelId]: (prev[group.modelId] || []).filter((_, row) => row !== idx),
                            }))}
                            className="text-gray-500 hover:text-red-300"
                            disabled={pairs.length <= 1}
                          >
                            Remove
                          </button>
                        </div>
                        <select
                          value={pair.mdSampleId}
                          onChange={(e) => setPairSelections((prev) => ({
                            ...prev,
                            [group.modelId]: (prev[group.modelId] || []).map((row, rowIdx) => rowIdx === idx ? { ...row, mdSampleId: e.target.value } : row),
                          }))}
                          className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-xs text-white"
                        >
                          {mdSamples.map((sample) => <option key={sample.sample_id} value={sample.sample_id}>{sample.name || sample.sample_id}</option>)}
                        </select>
                        <select
                          value={pair.sampleId}
                          onChange={(e) => setPairSelections((prev) => ({
                            ...prev,
                            [group.modelId]: (prev[group.modelId] || []).map((row, rowIdx) => rowIdx === idx ? { ...row, sampleId: e.target.value } : row),
                          }))}
                          className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1.5 text-xs text-white"
                        >
                          {selectableSamples.map((sample) => <option key={sample.sample_id} value={sample.sample_id}>{sample.name || sample.sample_id}</option>)}
                        </select>
                        {!meta && (
                          <p className="text-[10px] text-amber-300">No stored JS analysis for this pair/model.</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </aside>

        <main className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-gray-800 bg-gray-900/40 p-2">
            <button
              type="button"
              onClick={() => setActiveExplorerTab('js')}
              className={`rounded-md px-3 py-2 text-sm ${activeExplorerTab === 'js' ? 'bg-cyan-600 text-white' : 'text-gray-300 hover:bg-gray-800'}`}
            >
              Node/Edge JS
            </button>
            <button
              type="button"
              onClick={() => setActiveExplorerTab('energies')}
              className={`rounded-md px-3 py-2 text-sm ${activeExplorerTab === 'energies' ? 'bg-cyan-600 text-white' : 'text-gray-300 hover:bg-gray-800'}`}
            >
              Energies
            </button>
          </div>

          {activeExplorerTab === 'js' && (
          <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-gray-200">Node/Edge JS</h2>
                <p className="text-[11px] text-gray-500">
                  Select analyses on the left and MD/sample pairs in the middle panel. Each pair becomes one item in the plots.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setActiveJsSubtab('nodes')}
                  className={`rounded-md px-3 py-1.5 text-xs ${activeJsSubtab === 'nodes' ? 'bg-cyan-600 text-white' : 'border border-gray-700 text-gray-300 hover:bg-gray-800'}`}
                >
                  Nodes
                </button>
                <button
                  type="button"
                  onClick={() => setActiveJsSubtab('edges')}
                  className={`rounded-md px-3 py-1.5 text-xs ${activeJsSubtab === 'edges' ? 'bg-cyan-600 text-white' : 'border border-gray-700 text-gray-300 hover:bg-gray-800'}`}
                >
                  Edges
                </button>
              </div>
            </div>

            {analysisJobStatus?.status === 'finished' && analysisSummary && (
              <div className="rounded-md border border-cyan-800 bg-cyan-950/20 p-3 text-[12px] text-cyan-100 space-y-1">
                <div>
                  Wrote {analysisSummary.comparisons_written ?? 0} MD-vs-sample analyses, {analysisSummary.energies_written ?? 0}{' '}
                  energy analyses, and {analysisSummary.pose_energies_written ?? 0} single-pose energies.
                </div>
                {!!analysisSkippedSamples.length && (
                  <div className="text-cyan-200/90">
                    Skipped: {analysisSkippedSamples
                      .slice(0, 5)
                      .map((item) => `${item.sample_name || item.sample_id} (${item.stage}: ${item.reason})`)
                      .join(', ')}
                    {analysisSkippedSamples.length > 5 ? ` +${analysisSkippedSamples.length - 5} more` : ''}
                  </div>
                )}
              </div>
            )}
            {comparisonError && <ErrorMessage message={comparisonError} />}
            {comparisonLoading && <p className="text-sm text-gray-400">Loading selected JS analyses...</p>}
            {!selectedAnalysisGroups.length && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                Select one or more analyses in the left panel.
              </div>
            )}
            {!!selectedAnalysisGroups.length && !jsLoadedItems.length && !comparisonLoading && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                No matching MD-vs-sample analysis is available for the selected pair setup.
              </div>
            )}

            {activeJsSubtab === 'nodes' && !!multiNodePlot && (
              <>
                <div className="rounded-md border border-gray-800 bg-white p-3">
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <p className="text-xs font-semibold text-gray-800">Node JS by residue and selected pair</p>
                    <button
                      type="button"
                      className="text-[11px] text-gray-600 hover:text-gray-800"
                      onClick={() => setOverlayPlot({ ...multiNodePlot, title: 'Node JS by residue and selected pair' })}
                    >
                      Maximize
                    </button>
                  </div>
                  <Plot
                    data={multiNodePlot.data}
                    layout={multiNodePlot.layout}
                    config={multiNodePlot.config}
                    useResizeHandler
                    style={{ width: '100%', height: 'min(68vw, 620px)' }}
                  />
                </div>
                <div className="rounded-md border border-gray-800 bg-gray-950/40 p-3">
                  <p className="text-xs font-semibold text-gray-200">Top residues</p>
                  <div className="mt-2 grid grid-cols-1 lg:grid-cols-2 gap-x-6 gap-y-1 text-[11px] text-gray-300">
                    {multiTopResidues.map((row) => (
                      <div key={`${row.item}:${row.label}`} className="flex items-center justify-between gap-3">
                        <span className="min-w-0 truncate">{row.label} · {row.item}</span>
                        <span className="font-mono text-gray-100">{row.value.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {activeJsSubtab === 'edges' && !!jsLoadedItems.length && (
              <>
                <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4">
                  {jsLoadedItems.map((item) => {
                    const hasValues = item.edgeMatrix?.some((row) => row?.some((val) => Number.isFinite(val)));
                    return (
                      <div key={`edge-matrix:${item.key}`} className="rounded-md border border-gray-800 bg-white p-3">
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <p className="min-w-0 truncate text-xs font-semibold text-gray-800">{item.label}</p>
                          {hasValues && (
                            <button
                              type="button"
                              className="text-[11px] text-gray-600 hover:text-gray-800"
                              onClick={() => setOverlayPlot({
                                title: item.label,
                                data: [{
                                  z: item.edgeMatrix,
                                  x: residueLabels,
                                  y: residueLabels,
                                  type: 'heatmap',
                                  colorscale: 'Viridis',
                                  zmin: 0,
                                  zmax: globalEdgeJsMax,
                                  hovertemplate: 'x: %{x}<br>y: %{y}<br>JS: %{z:.4f}<extra></extra>',
                                }],
                                layout: {
                                  margin: { l: 80, r: 20, t: 20, b: 120 },
                                  paper_bgcolor: '#ffffff',
                                  plot_bgcolor: '#ffffff',
                                  font: { color: '#111827' },
                                  xaxis: { tickangle: -45, automargin: true, tickfont: { size: 9 } },
                                  yaxis: { automargin: true, scaleanchor: 'x', scaleratio: 1, tickfont: { size: 9 } },
                                },
                              })}
                            >
                              Maximize
                            </button>
                          )}
                        </div>
                        {!hasValues ? (
                          <p className="py-12 text-center text-sm text-gray-500">No edge JS values.</p>
                        ) : (
                          <Plot
                            data={[{
                              z: item.edgeMatrix,
                              x: residueLabels,
                              y: residueLabels,
                              type: 'heatmap',
                              colorscale: 'Viridis',
                              zmin: 0,
                              zmax: globalEdgeJsMax,
                              colorbar: { title: 'JS' },
                              hovertemplate: 'x: %{x}<br>y: %{y}<br>JS: %{z:.4f}<extra></extra>',
                            }]}
                            layout={{
                              margin: { l: 70, r: 20, t: 10, b: 105 },
                              paper_bgcolor: '#ffffff',
                              plot_bgcolor: '#ffffff',
                              font: { color: '#111827' },
                              xaxis: { tickangle: -45, automargin: true, tickfont: { size: 8 } },
                              yaxis: { automargin: true, scaleanchor: 'x', scaleratio: 1, tickfont: { size: 8 } },
                            }}
                            config={{ displayModeBar: false, responsive: true }}
                            useResizeHandler
                            style={{ width: '100%', height: 'min(58vw, 520px)' }}
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
                <div className="rounded-md border border-gray-800 bg-gray-950/40 p-3">
                  <p className="text-xs font-semibold text-gray-200">Top edges</p>
                  <div className="mt-2 grid grid-cols-1 lg:grid-cols-2 gap-x-6 gap-y-1 text-[11px] text-gray-300">
                    {multiTopEdges.map((row) => (
                      <div key={`${row.item}:${row.label}`} className="flex items-center justify-between gap-3">
                        <span className="min-w-0 truncate">{row.label} · {row.item}</span>
                        <span className="font-mono text-gray-100">{row.value.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </section>

          )}

          {activeExplorerTab === 'energies' && (
          <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-2">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-gray-200">Energies</h2>
                <p className="text-[11px] text-gray-500">
                  One graph is shown for each selected Potts model. All graphs share the same x-axis energy range.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={energyGraphMode}
                  onChange={(e) => setEnergyGraphMode(e.target.value)}
                  className="rounded border border-gray-700 bg-gray-950 px-2 py-1.5 text-xs text-gray-100"
                >
                  <option value="histogram">histograms + fitted curves</option>
                  <option value="curves">fitted curves only</option>
                </select>
                <span className="text-[11px] text-gray-500">{energyGraphs.length} model graph(s)</span>
              </div>
            </div>

            {!selectedAnalysisGroups.length && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                Select one or more analyses in the left panel to load energies.
              </div>
            )}
            {energyError && <ErrorMessage message={energyError} />}
            {energyLoading && <p className="text-sm text-gray-400">Loading energies...</p>}
            {!!selectedAnalysisGroups.length && !energyLoading && !energyGraphs.some((graph) => graph.series.length) && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                No energy analyses found for the selected models/settings. Run a new analysis first.
              </div>
            )}

            <div className="space-y-4">
              {energyGraphs.filter((graph) => graph.series.length).map((graph) => {
                const plot = buildEnergyDistributionPlot({
                  series: graph.series,
                  mode: energyGraphMode,
                  title: `${graph.modelName} · ${energyGraphMode === 'curves' ? 'energy fitted curves' : 'energy distributions'}`,
                  xTitle: 'Energy',
                  height: 300,
                  background: 'white',
                  xRange: globalEnergyRange,
                });
                return (
                  <div key={`energy-graph:${graph.modelId}`} className="rounded-md border border-gray-800 bg-white p-3">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-xs font-semibold text-gray-800">{graph.modelName}</p>
                        <p className="text-[11px] text-gray-500">{graph.series.length} trajectories · shared energy axis</p>
                      </div>
                      <button
                        type="button"
                        className="text-[11px] text-gray-600 hover:text-gray-800"
                        disabled={!plot}
                        onClick={() => setOverlayPlot({ ...plot, title: `${graph.modelName} energy distributions` })}
                      >
                        Maximize
                      </button>
                    </div>
                    {plot ? <EnergyDistributionPlot plot={plot} height={300} /> : <p className="py-12 text-center text-sm text-gray-500">No energy data.</p>}
                  </div>
                );
              })}
            </div>
          </section>
          )}
        </main>
      </div>
    </div>
  );
}
