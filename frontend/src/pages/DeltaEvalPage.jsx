import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Play, RefreshCw } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchPottsClusterInfo, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitDeltaEvalJob } from '../api/jobs';

const palette = ['#22d3ee', '#f97316', '#10b981', '#f43f5e', '#60a5fa', '#f59e0b', '#a855f7', '#84cc16'];
function pickColor(idx) {
  return palette[idx % palette.length];
}

function topKAbs(values, labels, k = 12) {
  const pairs = values.map((v, i) => [v, Math.abs(v), labels[i] ?? String(i), i]);
  pairs.sort((a, b) => b[1] - a[1]);
  return pairs.slice(0, k);
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

export default function DeltaEvalPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [clusterInfo, setClusterInfo] = useState(null);
  const [clusterInfoError, setClusterInfoError] = useState(null);

  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [keepInvalid, setKeepInvalid] = useState(false);

  const [analyses, setAnalyses] = useState([]);
  const [analysesError, setAnalysesError] = useState(null);
  const [analysisDataCache, setAnalysisDataCache] = useState({});

  const [mdSampleId, setMdSampleId] = useState('');
  const [modelAId, setModelAId] = useState('');
  const [modelBId, setModelBId] = useState('');

  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);

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

  const deltaModels = useMemo(() => {
    return pottsModels.filter((m) => {
      const params = m.params || {};
      if (params.fit_mode === 'delta') return true;
      const kind = params.delta_kind || '';
      return typeof kind === 'string' && kind.startsWith('delta_');
    });
  }, [pottsModels]);

  useEffect(() => {
    if (!clusterOptions.length) return;
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterId]);

  useEffect(() => {
    if (!mdSamples.length) {
      setMdSampleId('');
      return;
    }
    if (!mdSampleId || !mdSamples.some((s) => s.sample_id === mdSampleId)) {
      setMdSampleId(mdSamples[0].sample_id);
    }
  }, [mdSamples, mdSampleId]);

  useEffect(() => {
    if (!deltaModels.length) {
      setModelAId('');
      setModelBId('');
      return;
    }
    if (!modelAId || !deltaModels.some((m) => m.model_id === modelAId)) {
      setModelAId(deltaModels[0].model_id);
    }
    if (!modelBId || !deltaModels.some((m) => m.model_id === modelBId) || modelBId === modelAId) {
      const fallback = deltaModels.find((m) => m.model_id !== (modelAId || deltaModels[0].model_id));
      setModelBId(fallback?.model_id || '');
    }
  }, [deltaModels, modelAId, modelBId]);

  const loadClusterInfo = useCallback(async () => {
    if (!selectedClusterId) return;
    setClusterInfoError(null);
    try {
      // Use model A to get the correct edge set.
      const data = await fetchPottsClusterInfo(projectId, systemId, selectedClusterId, { modelId: modelAId || undefined });
      setClusterInfo(data);
    } catch (err) {
      setClusterInfoError(err.message || 'Failed to load cluster info.');
      setClusterInfo(null);
    }
  }, [projectId, systemId, selectedClusterId, modelAId]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_eval' });
      setAnalyses(Array.isArray(data?.analyses) ? data.analyses : []);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    setAnalysisDataCache({});
    loadClusterInfo();
    loadAnalyses();
  }, [selectedClusterId, loadClusterInfo, loadAnalyses]);

  useEffect(() => {
    if (!selectedClusterId) return;
    loadClusterInfo();
  }, [modelAId, selectedClusterId, loadClusterInfo]);

  const loadAnalysisData = useCallback(
    async (analysisId) => {
      if (!analysisId) return null;
      const cacheKey = `delta_eval:${analysisId}`;
      if (analysisDataCache[cacheKey]) return analysisDataCache[cacheKey];
      const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'delta_eval', analysisId);
      setAnalysisDataCache((prev) => ({ ...prev, [cacheKey]: payload }));
      return payload;
    },
    [analysisDataCache, projectId, systemId, selectedClusterId]
  );

  const dropInvalid = !keepInvalid;
  const selectedMeta = useMemo(() => {
    if (!mdSampleId || !modelAId || !modelBId) return null;
    return (
      analyses.find((a) => {
        const mode = (a.md_label_mode || 'assigned').toLowerCase();
        return (
          a.md_sample_id === mdSampleId &&
          a.model_a_id === modelAId &&
          a.model_b_id === modelBId &&
          mode === mdLabelMode &&
          Boolean(a.drop_invalid) === Boolean(dropInvalid)
        );
      }) || null
    );
  }, [analyses, mdSampleId, modelAId, modelBId, mdLabelMode, dropInvalid]);

  const [data, setData] = useState(null);
  const [dataError, setDataError] = useState(null);
  const [dataLoading, setDataLoading] = useState(false);
  useEffect(() => {
    const run = async () => {
      setData(null);
      setDataError(null);
      if (!selectedMeta) return;
      setDataLoading(true);
      try {
        const payload = await loadAnalysisData(selectedMeta.analysis_id);
        setData(payload);
      } catch (err) {
        setDataError(err.message || 'Failed to load data.');
      } finally {
        setDataLoading(false);
      }
    };
    run();
  }, [selectedMeta, loadAnalysisData]);

  const handleRun = useCallback(async () => {
    if (!selectedClusterId || !mdSampleId || !modelAId || !modelBId) return;
    setJobError(null);
    setJob(null);
    setJobStatus(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        md_sample_id: mdSampleId,
        model_a_id: modelAId,
        model_b_id: modelBId,
        md_label_mode: mdLabelMode,
        keep_invalid: keepInvalid,
      };
      const res = await submitDeltaEvalJob(payload);
      setJob(res);
    } catch (err) {
      setJobError(err.message || 'Failed to submit delta eval job.');
    }
  }, [projectId, systemId, selectedClusterId, mdSampleId, modelAId, modelBId, mdLabelMode, keepInvalid]);

  useEffect(() => {
    if (!job?.job_id) return;
    let cancelled = false;
    const terminal = new Set(['finished', 'failed', 'canceled']);
    const poll = async () => {
      try {
        const status = await fetchJobStatus(job.job_id);
        if (cancelled) return;
        setJobStatus(status);
        if (terminal.has(status?.status)) {
          clearInterval(timer);
          if (status?.status === 'finished') await loadAnalyses();
        }
      } catch (err) {
        if (!cancelled) setJobError(err.message || 'Failed to poll job.');
      }
    };
    const timer = setInterval(poll, 2000);
    poll();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job, loadAnalyses]);

  const residueLabels = useMemo(() => {
    const keys = clusterInfo?.residue_keys || [];
    if (Array.isArray(keys) && keys.length) return keys;
    const n = clusterInfo?.n_residues || 0;
    return Array.from({ length: n }, (_, i) => String(i));
  }, [clusterInfo]);

  const deltaEnergy = useMemo(() => (Array.isArray(data?.data?.delta_energy) ? data.data.delta_energy : []), [data]);
  const deltaResMean = useMemo(
    () => (Array.isArray(data?.data?.delta_residue_mean) ? data.data.delta_residue_mean : []),
    [data]
  );
  const deltaEdges = useMemo(() => (Array.isArray(data?.data?.edges) ? data.data.edges : []), [data]);
  const deltaEdgeMean = useMemo(
    () => (Array.isArray(data?.data?.delta_edge_mean) ? data.data.delta_edge_mean : []),
    [data]
  );

  const topResidues = useMemo(() => topKAbs(deltaResMean, residueLabels, 12), [deltaResMean, residueLabels]);
  const edgeMatrix = useMemo(
    () => buildEdgeMatrix(residueLabels.length, deltaEdges, deltaEdgeMean),
    [residueLabels, deltaEdges, deltaEdgeMean]
  );
  const edgeMatrixHasValues = useMemo(
    () => edgeMatrix?.some((row) => row?.some((val) => Number.isFinite(val))),
    [edgeMatrix]
  );

  if (loadingSystem) return <Loader message="Loading delta evaluation..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Delta Potts Evaluation</h1>
          <p className="text-sm text-gray-400">
            Computes per-residue/edge preferences on an MD sample for two selected delta Potts fits (Point 4 in{' '}
            <code>validation_ladder2.MD</code>).
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/visualize`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Back to sampling
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
              {clusterInfoError && <p className="text-[11px] text-red-300">{clusterInfoError}</p>}
              {clusterInfo && (
                <p className="text-[11px] text-gray-500">
                  Residues: {clusterInfo.n_residues} · Edges: {clusterInfo.n_edges} {clusterInfo.edges_source ? `(${clusterInfo.edges_source})` : ''}
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
              <label className="block text-xs text-gray-400">MD sample</label>
              <select
                value={mdSampleId}
                onChange={(e) => setMdSampleId(e.target.value)}
                disabled={!mdSamples.length}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white disabled:opacity-60"
              >
                {!mdSamples.length && <option value="">No MD samples</option>}
                {mdSamples.map((s) => (
                  <option key={s.sample_id} value={s.sample_id}>
                    {s.name || s.sample_id}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Model A (negative = A-favored)</label>
              <select
                value={modelAId}
                onChange={(e) => setModelAId(e.target.value)}
                disabled={!deltaModels.length}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white disabled:opacity-60"
              >
                {!deltaModels.length && <option value="">No delta models</option>}
                {deltaModels.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.name || m.model_id}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Model B (positive = B-favored)</label>
              <select
                value={modelBId}
                onChange={(e) => setModelBId(e.target.value)}
                disabled={!deltaModels.length}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white disabled:opacity-60"
              >
                {!deltaModels.length && <option value="">No delta models</option>}
                {deltaModels.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.name || m.model_id}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleRun}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white text-sm disabled:opacity-60"
                disabled={!selectedClusterId || !mdSampleId || !modelAId || !modelBId || modelAId === modelBId}
              >
                <Play className="h-4 w-4" />
                Run eval
              </button>
              <button
                type="button"
                onClick={async () => {
                  await loadClusterInfo();
                  await loadAnalyses();
                }}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 text-gray-200 text-sm hover:border-gray-500"
              >
                <RefreshCw className="h-4 w-4" />
                Refresh
              </button>
            </div>

            {job?.job_id && (
              <div className="text-[11px] text-gray-300">
                Job: <span className="text-gray-200">{job.job_id}</span>{' '}
                {jobStatus?.meta?.status ? `· ${jobStatus.meta.status}` : ''}
                {typeof jobStatus?.meta?.progress === 'number' ? ` · ${jobStatus.meta.progress}%` : ''}
              </div>
            )}
            {jobError && <ErrorMessage message={jobError} />}
            {analysesError && <ErrorMessage message={analysesError} />}
          </div>
        </aside>

        <main className="space-y-4">
          <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
            <div>
              <h2 className="text-sm font-semibold text-gray-200">Frame-level preference</h2>
              <p className="text-[11px] text-gray-500">ΔE(t) = E_A(s_t) − E_B(s_t). Negative means A-favored.</p>
            </div>
            {!selectedMeta && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                No evaluation found for this selection. Click <span className="font-semibold">Run eval</span>.
              </div>
            )}
            {dataError && <ErrorMessage message={dataError} />}
            {dataLoading && <p className="text-sm text-gray-400">Loading…</p>}
            {!!deltaEnergy.length && (
              <div className="rounded-md border border-gray-800 bg-white p-3">
                <Plot
                  data={[
                    {
                      x: deltaEnergy,
                      type: 'histogram',
                      opacity: 0.75,
                      marker: { color: pickColor(0) },
                      nbinsx: 60,
                      name: 'ΔE',
                    },
                  ]}
                  layout={{
                    height: 260,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { title: 'ΔE (A - B)', color: '#111827' },
                    yaxis: { title: 'Count', color: '#111827' },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '260px' }}
                />
              </div>
            )}
          </section>

          <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
            <div>
              <h2 className="text-sm font-semibold text-gray-200">Per-residue map</h2>
              <p className="text-[11px] text-gray-500">Mean δ_i = ⟨h^A_i(s_i) − h^B_i(s_i)⟩ over MD frames.</p>
            </div>
            {!!deltaResMean.length && (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div className="rounded-md border border-gray-800 bg-white p-3">
                  <Plot
                    data={[
                      {
                        x: residueLabels,
                        y: deltaResMean,
                        type: 'bar',
                        marker: { color: '#22d3ee' },
                        hovertemplate: 'res: %{x}<br>mean δ: %{y:.4f}<extra></extra>',
                      },
                    ]}
                    layout={{
                      margin: { l: 40, r: 10, t: 10, b: 80 },
                      paper_bgcolor: '#ffffff',
                      plot_bgcolor: '#ffffff',
                      font: { color: '#111827' },
                      xaxis: { tickfont: { size: 9 }, color: '#111827' },
                      yaxis: { title: 'mean δ (A - B)', color: '#111827' },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    useResizeHandler
                    style={{ width: '100%', height: '260px' }}
                  />
                </div>
                <div className="rounded-md border border-gray-800 bg-white p-3 text-[11px] text-gray-700">
                  <p className="text-xs font-semibold text-gray-800">Top |δ| residues</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2">
                    {topResidues.map(([v, , label]) => (
                      <div key={label} className="flex items-center justify-between gap-2">
                        <span className="truncate">{label}</span>
                        <span className="font-mono">{Number(v).toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </section>

          <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
            <div>
              <h2 className="text-sm font-semibold text-gray-200">Per-edge map</h2>
              <p className="text-[11px] text-gray-500">Mean δ_ij = ⟨J^A_ij(s_i,s_j) − J^B_ij(s_i,s_j)⟩ over frames.</p>
            </div>
            {!edgeMatrixHasValues && <p className="text-[11px] text-gray-500">No edge data available.</p>}
            {edgeMatrixHasValues && (
              <div className="rounded-md border border-gray-800 bg-white p-3">
                <Plot
                  data={[
                    {
                      z: edgeMatrix,
                      x: residueLabels,
                      y: residueLabels,
                      type: 'heatmap',
                      colorscale: 'RdBu',
                      reversescale: true,
                      zmid: 0,
                      hovertemplate: 'x: %{x}<br>y: %{y}<br>mean δ: %{z:.4f}<extra></extra>',
                    },
                  ]}
                  layout={{
                    margin: { l: 60, r: 10, t: 10, b: 60 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '320px' }}
                />
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}

