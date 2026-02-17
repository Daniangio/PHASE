import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, Play, RefreshCw } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitGibbsRelaxationJob } from '../api/jobs';

function topKBy(values, labels, k = 10, descending = true) {
  const arr = (values || []).map((v, i) => ({ value: Number(v), label: labels[i] ?? String(i), index: i }));
  const filtered = arr.filter((x) => Number.isFinite(x.value));
  filtered.sort((a, b) => (descending ? b.value - a.value : a.value - b.value));
  return filtered.slice(0, k);
}

export default function GibbsRelaxationPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [modelId, setModelId] = useState('');
  const [startSampleId, setStartSampleId] = useState('');

  const [beta, setBeta] = useState(1.0);
  const [nStartFrames, setNStartFrames] = useState(100);
  const [gibbsSweeps, setGibbsSweeps] = useState(1000);
  const [workers, setWorkers] = useState(0);
  const [seed, setSeed] = useState(0);
  const [startLabelMode, setStartLabelMode] = useState('assigned');
  const [keepInvalid, setKeepInvalid] = useState(false);

  const [analyses, setAnalyses] = useState([]);
  const [analysesLoading, setAnalysesLoading] = useState(false);
  const [analysesError, setAnalysesError] = useState(null);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisDataLoading, setAnalysisDataLoading] = useState(false);
  const [analysisDataError, setAnalysisDataError] = useState(null);
  const analysisDataCacheRef = useRef({});
  const analysisDataInFlightRef = useRef({});

  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);

  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    const loadSystem = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
      } catch (err) {
        setSystemError(err.message || 'Failed to load system.');
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
  const mdSamples = useMemo(
    () => (selectedCluster?.samples || []).filter((s) => s.type === 'md_eval'),
    [selectedCluster]
  );
  const modelOptions = useMemo(() => {
    const models = selectedCluster?.potts_models || [];
    return models.filter((m) => {
      const params = m?.params || {};
      const dk = String(params?.delta_kind || '').toLowerCase();
      return !dk.startsWith('delta');
    });
  }, [selectedCluster]);

  useEffect(() => {
    if (!clusterOptions.length) return;
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterId]);

  useEffect(() => {
    if (!modelOptions.length) {
      setModelId('');
      return;
    }
    if (!modelId || !modelOptions.some((m) => m.model_id === modelId)) {
      setModelId(modelOptions[0].model_id);
    }
  }, [modelOptions, modelId]);

  useEffect(() => {
    if (!mdSamples.length) {
      setStartSampleId('');
      return;
    }
    if (!startSampleId || !mdSamples.some((s) => s.sample_id === startSampleId)) {
      setStartSampleId(mdSamples[0].sample_id);
    }
  }, [mdSamples, startSampleId]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return [];
    setAnalysesLoading(true);
    setAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'gibbs_relaxation' });
      const list = Array.isArray(data?.analyses) ? data.analyses : [];
      setAnalyses(list);
      setSelectedAnalysisId((prev) => {
        if (prev && list.some((a) => a.analysis_id === prev)) return prev;
        return list[0]?.analysis_id || '';
      });
      return list;
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
      return [];
    } finally {
      setAnalysesLoading(false);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    analysisDataCacheRef.current = {};
    analysisDataInFlightRef.current = {};
    setSelectedAnalysisId('');
    setAnalysisData(null);
    loadAnalyses();
  }, [selectedClusterId, loadAnalyses]);

  const loadAnalysisData = useCallback(
    async (analysisId) => {
      if (!analysisId) return null;
      const cacheKey = `gibbs_relaxation:${analysisId}`;
      const cached = analysisDataCacheRef.current;
      if (Object.prototype.hasOwnProperty.call(cached, cacheKey)) return cached[cacheKey];
      const inflight = analysisDataInFlightRef.current;
      if (inflight[cacheKey]) return inflight[cacheKey];

      const p = fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'gibbs_relaxation', analysisId)
        .then((payload) => {
          analysisDataCacheRef.current = { ...analysisDataCacheRef.current, [cacheKey]: payload };
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

  useEffect(() => {
    const run = async () => {
      setAnalysisDataError(null);
      setAnalysisData(null);
      if (!selectedAnalysisId) return;
      setAnalysisDataLoading(true);
      try {
        const payload = await loadAnalysisData(selectedAnalysisId);
        setAnalysisData(payload);
      } catch (err) {
        setAnalysisDataError(err.message || 'Failed to load analysis.');
      } finally {
        setAnalysisDataLoading(false);
      }
    };
    run();
  }, [selectedAnalysisId, loadAnalysisData]);

  const handleSubmit = useCallback(async () => {
    if (!selectedClusterId || !modelId || !startSampleId) return;
    setJobError(null);
    setJob(null);
    setJobStatus(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        model_id: modelId,
        start_sample_id: startSampleId,
        beta: Number(beta),
        n_start_frames: Number(nStartFrames),
        gibbs_sweeps: Number(gibbsSweeps),
        workers: Number(workers),
        seed: Number(seed),
        start_label_mode: startLabelMode,
        keep_invalid: keepInvalid,
      };
      const res = await submitGibbsRelaxationJob(payload);
      setJob(res);
    } catch (err) {
      setJobError(err.message || 'Failed to submit relaxation analysis.');
    }
  }, [
    projectId,
    systemId,
    selectedClusterId,
    modelId,
    startSampleId,
    beta,
    nStartFrames,
    gibbsSweeps,
    workers,
    seed,
    startLabelMode,
    keepInvalid,
  ]);

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
          if (status?.status === 'finished') {
            const list = await loadAnalyses();
            if (Array.isArray(list) && list.length) setSelectedAnalysisId(list[0].analysis_id);
            const data = await fetchSystem(projectId, systemId);
            if (!cancelled) setSystem(data);
          }
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
  }, [job, loadAnalyses, projectId, systemId]);

  const selectedAnalysisMeta = useMemo(
    () => analyses.find((a) => a.analysis_id === selectedAnalysisId) || null,
    [analyses, selectedAnalysisId]
  );

  const parsed = useMemo(() => {
    const d = analysisData?.data || {};
    const residueKeys = Array.isArray(d.residue_keys) && d.residue_keys.length
      ? d.residue_keys.map((v) => String(v))
      : Array.from({ length: Array.isArray(d.mean_first_flip_steps) ? d.mean_first_flip_steps.length : 0 }, (_, i) => `res_${i}`);
    const meanFirst = Array.isArray(d.mean_first_flip_steps) ? d.mean_first_flip_steps.map((v) => Number(v)) : [];
    const pctFast = Array.isArray(d.flip_percentile_fast) ? d.flip_percentile_fast.map((v) => Number(v)) : [];
    const everFlip = Array.isArray(d.ever_flip_rate) ? d.ever_flip_rate.map((v) => Number(v)) : [];
    const earlyFlip = Array.isArray(d.early_flip_rate) ? d.early_flip_rate.map((v) => Number(v)) : [];
    const meanFlipByStep = Array.isArray(d.mean_flip_fraction_by_step) ? d.mean_flip_fraction_by_step.map((v) => Number(v)) : [];
    const energyMean = Array.isArray(d.energy_mean) ? d.energy_mean.map((v) => Number(v)) : [];
    const energyStd = Array.isArray(d.energy_std) ? d.energy_std.map((v) => Number(v)) : [];
    const flipProbTime = Array.isArray(d.flip_prob_time) ? d.flip_prob_time : [];
    return { residueKeys, meanFirst, pctFast, everFlip, earlyFlip, meanFlipByStep, energyMean, energyStd, flipProbTime };
  }, [analysisData]);

  const topFast = useMemo(
    () => topKBy(parsed.pctFast, parsed.residueKeys, 10, true),
    [parsed.pctFast, parsed.residueKeys]
  );
  const topSlow = useMemo(
    () => topKBy(parsed.pctFast, parsed.residueKeys, 10, false),
    [parsed.pctFast, parsed.residueKeys]
  );

  if (loadingSystem) return <Loader message="Loading Gibbs relaxation..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Gibbs Relaxation: Help"
        docPath="/docs/gibbs_relaxation_help.md"
        onClose={() => setHelpOpen(false)}
      />

      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Gibbs Relaxation</h1>
          <p className="text-sm text-gray-400">
            Start from random MD frames and relax under a selected Potts Hamiltonian. Residue coloring uses flip-time percentiles.
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
            onClick={() =>
              navigate(
                `/projects/${projectId}/systems/${systemId}/sampling/gibbs_relaxation_3d?cluster_id=${encodeURIComponent(
                  selectedClusterId || ''
                )}&analysis_id=${encodeURIComponent(selectedAnalysisId || '')}`
              )
            }
            className="text-xs px-3 py-2 rounded-md border border-cyan-700 text-cyan-200 hover:border-cyan-500"
          >
            Open 3D viewer
          </button>
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

      <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-4">
        <aside className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Cluster</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
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
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Potts model (target Hamiltonian)</label>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                {modelOptions.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.name || m.model_id}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-xs text-gray-400 mb-1">Starting MD sample</label>
              <select
                value={startSampleId}
                onChange={(e) => setStartSampleId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                {mdSamples.map((s) => (
                  <option key={s.sample_id} value={s.sample_id}>
                    {s.name || s.sample_id}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">β</label>
                <input
                  type="number"
                  step={0.05}
                  min={0.01}
                  value={beta}
                  onChange={(e) => setBeta(Number(e.target.value))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Workers (0 = auto)</label>
                <input
                  type="number"
                  min={0}
                  value={workers}
                  onChange={(e) => setWorkers(Number(e.target.value))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Random starts</label>
                <input
                  type="number"
                  min={1}
                  value={nStartFrames}
                  onChange={(e) => setNStartFrames(Number(e.target.value))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Gibbs sweeps / run</label>
                <input
                  type="number"
                  min={1}
                  value={gibbsSweeps}
                  onChange={(e) => setGibbsSweeps(Number(e.target.value))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Seed</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Start label mode</label>
                <select
                  value={startLabelMode}
                  onChange={(e) => setStartLabelMode(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                >
                  <option value="assigned">assigned</option>
                  <option value="halo">halo</option>
                </select>
              </div>
            </div>

            <label className="flex items-center gap-2 text-xs text-gray-300">
              <input
                type="checkbox"
                checked={keepInvalid}
                onChange={(e) => setKeepInvalid(e.target.checked)}
              />
              Keep invalid frames
            </label>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleSubmit}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white text-sm"
                disabled={!selectedClusterId || !modelId || !startSampleId}
              >
                <Play className="h-4 w-4" />
                Run analysis
              </button>
              <button
                type="button"
                onClick={async () => {
                  await loadAnalyses();
                  const data = await fetchSystem(projectId, systemId);
                  setSystem(data);
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
          </div>

          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-semibold text-gray-200">Saved analyses</p>
              {analysesLoading && <p className="text-[11px] text-gray-500">Loading…</p>}
            </div>
            {analysesError && <ErrorMessage message={analysesError} />}
            {analyses.length === 0 && <p className="text-[11px] text-gray-500">No Gibbs relaxation analyses yet.</p>}
            {analyses.length > 0 && (
              <select
                value={selectedAnalysisId}
                onChange={(e) => setSelectedAnalysisId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                {analyses.map((a) => (
                  <option key={a.analysis_id} value={a.analysis_id}>
                    {(a.model_name || a.model_id || 'model')} · {a.start_sample_name || a.start_sample_id} · {a.created_at || a.analysis_id}
                  </option>
                ))}
              </select>
            )}
            {selectedAnalysisMeta && (
              <div className="text-[11px] text-gray-400 space-y-1">
                <div>
                  <span className="text-gray-500">model:</span> {selectedAnalysisMeta.model_name || selectedAnalysisMeta.model_id}
                </div>
                <div>
                  <span className="text-gray-500">starts:</span> {selectedAnalysisMeta.n_start_frames_used || selectedAnalysisMeta.summary?.n_start_frames_used || selectedAnalysisMeta.n_start_frames_requested || selectedAnalysisMeta.summary?.n_samples || '—'}
                </div>
              </div>
            )}
          </div>
        </aside>

        <main className="space-y-4">
          {analysisDataLoading && <Loader message="Loading analysis…" />}
          {analysisDataError && <ErrorMessage message={analysisDataError} />}

          {analysisData && (
            <>
              <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
                <h2 className="text-sm font-semibold text-gray-200">Residue Flip Percentiles</h2>
                <div className="rounded-md border border-gray-800 bg-white p-3">
                  <Plot
                    data={[
                      {
                        x: parsed.residueKeys,
                        y: parsed.meanFirst,
                        type: 'bar',
                        marker: {
                          color: parsed.pctFast,
                          cmin: 0,
                          cmax: 1,
                          colorscale: [
                            [0.0, '#1d4ed8'],
                            [0.5, '#f3f4f6'],
                            [1.0, '#dc2626'],
                          ],
                          colorbar: { title: 'Fast flip percentile', thickness: 12 },
                        },
                        hovertemplate:
                          'Residue: %{x}<br>Mean first flip step: %{y:.2f}<br>Fast percentile: %{marker.color:.3f}<extra></extra>',
                      },
                    ]}
                    layout={{
                      height: 320,
                      margin: { l: 60, r: 30, t: 10, b: 90 },
                      paper_bgcolor: '#ffffff',
                      plot_bgcolor: '#ffffff',
                      font: { color: '#111827' },
                      xaxis: { title: 'Residue', tickfont: { size: 9 }, color: '#111827' },
                      yaxis: { title: 'Mean first flip step', color: '#111827' },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    useResizeHandler
                    style={{ width: '100%', height: '320px' }}
                  />
                </div>
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 text-[11px]">
                  <div className="rounded-md border border-gray-800 bg-gray-950/40 p-3">
                    <p className="text-gray-200 font-semibold mb-1">Fastest flippers (high percentile)</p>
                    <div className="space-y-1 text-gray-300">
                      {topFast.map((row) => (
                        <div key={`fast-${row.index}`} className="flex items-center justify-between gap-2">
                          <span>{row.label}</span>
                          <span className="font-mono">{row.value.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-md border border-gray-800 bg-gray-950/40 p-3">
                    <p className="text-gray-200 font-semibold mb-1">Slowest flippers (low percentile)</p>
                    <div className="space-y-1 text-gray-300">
                      {topSlow.map((row) => (
                        <div key={`slow-${row.index}`} className="flex items-center justify-between gap-2">
                          <span>{row.label}</span>
                          <span className="font-mono">{row.value.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </section>

              <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
                <h2 className="text-sm font-semibold text-gray-200">Relaxation Curves</h2>
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  <div className="rounded-md border border-gray-800 bg-white p-3">
                    <p className="text-xs font-semibold text-gray-800 mb-2">Mean fraction of flipped residues vs sweep</p>
                    <Plot
                      data={[
                        {
                          x: parsed.meanFlipByStep.map((_, i) => i + 1),
                          y: parsed.meanFlipByStep,
                          mode: 'lines',
                          line: { color: '#22d3ee' },
                          name: 'mean flip fraction',
                        },
                      ]}
                      layout={{
                        height: 260,
                        margin: { l: 50, r: 10, t: 10, b: 40 },
                        paper_bgcolor: '#ffffff',
                        plot_bgcolor: '#ffffff',
                        font: { color: '#111827' },
                        xaxis: { title: 'Sweep', color: '#111827' },
                        yaxis: { title: 'Flip fraction', color: '#111827', range: [0, 1] },
                      }}
                      config={{ displayModeBar: false, responsive: true }}
                      useResizeHandler
                      style={{ width: '100%', height: '260px' }}
                    />
                  </div>
                  <div className="rounded-md border border-gray-800 bg-white p-3">
                    <p className="text-xs font-semibold text-gray-800 mb-2">Energy trace (mean ± std)</p>
                    <Plot
                      data={[
                        {
                          x: parsed.energyMean.map((_, i) => i + 1),
                          y: parsed.energyMean.map((v, i) => v - (parsed.energyStd[i] || 0)),
                          mode: 'lines',
                          line: { color: 'rgba(34,211,238,0.2)' },
                          name: 'mean-std',
                          showlegend: false,
                        },
                        {
                          x: parsed.energyMean.map((_, i) => i + 1),
                          y: parsed.energyMean.map((v, i) => v + (parsed.energyStd[i] || 0)),
                          mode: 'lines',
                          fill: 'tonexty',
                          fillcolor: 'rgba(34,211,238,0.25)',
                          line: { color: 'rgba(34,211,238,0.2)' },
                          name: 'std band',
                        },
                        {
                          x: parsed.energyMean.map((_, i) => i + 1),
                          y: parsed.energyMean,
                          mode: 'lines',
                          line: { color: '#0891b2' },
                          name: 'mean energy',
                        },
                      ]}
                      layout={{
                        height: 260,
                        margin: { l: 50, r: 10, t: 10, b: 40 },
                        paper_bgcolor: '#ffffff',
                        plot_bgcolor: '#ffffff',
                        font: { color: '#111827' },
                        xaxis: { title: 'Sweep', color: '#111827' },
                        yaxis: { title: 'Energy', color: '#111827' },
                      }}
                      config={{ displayModeBar: false, responsive: true }}
                      useResizeHandler
                      style={{ width: '100%', height: '260px' }}
                    />
                  </div>
                </div>
              </section>

              {Array.isArray(parsed.flipProbTime) && parsed.flipProbTime.length > 0 && (
                <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
                  <h2 className="text-sm font-semibold text-gray-200">Flip Probability Heatmap</h2>
                  <div className="rounded-md border border-gray-800 bg-white p-3">
                    <Plot
                      data={[
                        {
                          z: parsed.flipProbTime[0]?.length
                            ? parsed.flipProbTime[0].map((_, col) => parsed.flipProbTime.map((row) => Number(row[col])))
                            : [],
                          x: parsed.meanFlipByStep.map((_, i) => i + 1),
                          y: parsed.residueKeys,
                          type: 'heatmap',
                          colorscale: 'Viridis',
                          zmin: 0,
                          zmax: 1,
                          hovertemplate: 'Residue: %{y}<br>Sweep: %{x}<br>Flip prob: %{z:.3f}<extra></extra>',
                        },
                      ]}
                      layout={{
                        height: 420,
                        margin: { l: 80, r: 20, t: 10, b: 40 },
                        paper_bgcolor: '#ffffff',
                        plot_bgcolor: '#ffffff',
                        font: { color: '#111827' },
                        xaxis: { title: 'Sweep', color: '#111827' },
                        yaxis: { title: 'Residue', color: '#111827', automargin: true },
                      }}
                      config={{ displayModeBar: false, responsive: true }}
                      useResizeHandler
                      style={{ width: '100%', height: '420px' }}
                    />
                  </div>
                </section>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
