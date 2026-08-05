import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, Play, RefreshCw } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitLambdaSweepJob } from '../api/jobs';

const palette = ['#22d3ee', '#f97316', '#10b981', '#f43f5e', '#60a5fa', '#f59e0b', '#a855f7', '#84cc16'];
function pickColor(idx) {
  return palette[idx % palette.length];
}

export default function LambdaSweepPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');

  const [analyses, setAnalyses] = useState([]);
  const [analysesLoading, setAnalysesLoading] = useState(false);
  const [analysesError, setAnalysesError] = useState(null);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [, setAnalysisDataCache] = useState({});
  const analysisDataCacheRef = useRef({});
  const analysisDataInFlightRef = useRef({});

  const [helpOpen, setHelpOpen] = useState(false);

  const [modelAId, setModelAId] = useState('');
  const [modelBId, setModelBId] = useState('');
  const [referenceAId, setReferenceAId] = useState('');
  const [referenceBId, setReferenceBId] = useState('');
  const [comparisonSampleIds, setComparisonSampleIds] = useState([]);

  const [seriesLabel, setSeriesLabel] = useState('');
  const [lambdaCount, setLambdaCount] = useState(11);
  const [alpha, setAlpha] = useState(0.5);
  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [keepInvalid, setKeepInvalid] = useState(false);

  const [gibbsMethod, setGibbsMethod] = useState('rex');
  const [beta, setBeta] = useState(1.0);
  const [rexRounds, setRexRounds] = useState(2000);
  const [rexBurninRounds, setRexBurninRounds] = useState(50);
  const [rexSweepsPerRound, setRexSweepsPerRound] = useState(2);
  const [rexThinRounds, setRexThinRounds] = useState(1);

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

  const sampleOptions = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);
  const pottsModels = useMemo(() => selectedCluster?.potts_models || [], [selectedCluster]);
  const sampleLabel = useCallback((sample) => {
    const name = sample?.name || sample?.sample_id || 'sample';
    const type = sample?.type || 'sample';
    return `${name} (${type})`;
  }, []);

  const endpointModelOptions = useMemo(() => {
    return pottsModels.filter((m) => {
      const params = m?.params || {};
      const kind = params.delta_kind || '';
      if (typeof kind === 'string' && kind.startsWith('delta')) return false; // delta-only, not sampleable
      return true;
    });
  }, [pottsModels]);

  useEffect(() => {
    if (!clusterOptions.length) return;
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterId]);

  useEffect(() => {
    if (!endpointModelOptions.length) {
      setModelAId('');
      setModelBId('');
      return;
    }
    const has = (id) => id && endpointModelOptions.some((m) => m.model_id === id);
    let a = has(modelAId) ? modelAId : endpointModelOptions[0].model_id;
    let b = has(modelBId) ? modelBId : '';
    if (!b || b === a) {
      const fallback = endpointModelOptions.find((m) => m.model_id !== a);
      b = fallback?.model_id || '';
    }
    if (a !== modelAId) setModelAId(a);
    if (b !== modelBId) setModelBId(b);
  }, [endpointModelOptions, modelAId, modelBId]);

  useEffect(() => {
    if (!sampleOptions.length) {
      setReferenceAId('');
      setReferenceBId('');
      setComparisonSampleIds([]);
      return;
    }
    const pickDistinct = (fallbackIndex, exclude) => {
      if (sampleOptions[fallbackIndex] && !exclude.includes(sampleOptions[fallbackIndex].sample_id)) {
        return sampleOptions[fallbackIndex].sample_id;
      }
      const found = sampleOptions.find((s) => !exclude.includes(s.sample_id));
      return found ? found.sample_id : '';
    };
    const has = (id) => id && sampleOptions.some((s) => s.sample_id === id);
    let nextA = has(referenceAId) ? referenceAId : pickDistinct(0, []);
    let nextB = has(referenceBId) && referenceBId !== nextA ? referenceBId : pickDistinct(1, [nextA]);
    if (nextB === nextA) nextB = pickDistinct(1, [nextA]);
    const validComparisons = comparisonSampleIds.filter(
      (id) => has(id) && id !== nextA && id !== nextB
    );
    const defaultComparison = pickDistinct(2, [nextA, nextB]);
    const nextComparisons = validComparisons.length ? validComparisons : (defaultComparison ? [defaultComparison] : []);
    if (nextA !== referenceAId) setReferenceAId(nextA);
    if (nextB !== referenceBId) setReferenceBId(nextB);
    const sameComparisons =
      nextComparisons.length === comparisonSampleIds.length &&
      nextComparisons.every((id, idx) => id === comparisonSampleIds[idx]);
    if (!sameComparisons) setComparisonSampleIds(nextComparisons);
  }, [sampleOptions, referenceAId, referenceBId, comparisonSampleIds]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesLoading(true);
    setAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'lambda_sweep' });
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
    setAnalysisDataCache({});
    analysisDataCacheRef.current = {};
    analysisDataInFlightRef.current = {};
    setSelectedAnalysisId('');
    loadAnalyses();
  }, [selectedClusterId, loadAnalyses]);

  const selectedAnalysisMeta = useMemo(
    () => analyses.find((a) => a.analysis_id === selectedAnalysisId) || null,
    [analyses, selectedAnalysisId]
  );

  const loadAnalysisData = useCallback(
    async (analysisId) => {
      if (!analysisId) return null;
      const cacheKey = `lambda_sweep:${analysisId}`;
      const cached = analysisDataCacheRef.current;
      if (Object.prototype.hasOwnProperty.call(cached, cacheKey)) return cached[cacheKey];

      const inflight = analysisDataInFlightRef.current;
      if (inflight[cacheKey]) return inflight[cacheKey];

      const p = fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'lambda_sweep', analysisId)
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

  const [analysisData, setAnalysisData] = useState(null);
  const [analysisDataError, setAnalysisDataError] = useState(null);
  const [analysisDataLoading, setAnalysisDataLoading] = useState(false);

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
    if (!selectedClusterId) return;
    setJobError(null);
    setJob(null);
    setJobStatus(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        model_a_id: modelAId,
        model_b_id: modelBId,
        reference_sample_id_a: referenceAId,
        reference_sample_id_b: referenceBId,
        comparison_sample_ids: comparisonSampleIds,
        series_label: seriesLabel || undefined,
        lambda_count: lambdaCount ? Number(lambdaCount) : undefined,
        alpha: alpha !== null && alpha !== undefined ? Number(alpha) : undefined,
        md_label_mode: mdLabelMode,
        keep_invalid: keepInvalid,
        gibbs_method: gibbsMethod,
        beta: beta ? Number(beta) : undefined,
        rex_rounds: gibbsMethod === 'rex' ? Number(rexRounds) : undefined,
        rex_burnin_rounds: gibbsMethod === 'rex' ? Number(rexBurninRounds) : undefined,
        rex_sweeps_per_round: gibbsMethod === 'rex' ? Number(rexSweepsPerRound) : undefined,
        rex_thin_rounds: gibbsMethod === 'rex' ? Number(rexThinRounds) : undefined,
      };
      const res = await submitLambdaSweepJob(payload);
      setJob(res);
    } catch (err) {
      setJobError(err.message || 'Failed to submit lambda sweep job.');
    }
  }, [
    projectId,
    systemId,
    selectedClusterId,
    modelAId,
    modelBId,
    referenceAId,
    referenceBId,
    comparisonSampleIds,
    seriesLabel,
    lambdaCount,
    alpha,
    mdLabelMode,
    keepInvalid,
    gibbsMethod,
    beta,
    rexRounds,
    rexBurninRounds,
    rexSweepsPerRound,
    rexThinRounds,
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

  const plots = useMemo(() => {
    const data = analysisData?.data || null;
    if (!data) return null;
    const lambdas = Array.isArray(data.lambdas) ? data.lambdas : [];
    const refNames = Array.isArray(data.reference_sample_names)
      ? data.reference_sample_names
      : Array.isArray(data.ref_md_sample_names)
        ? data.ref_md_sample_names
        : [];
    const comparisonNames = Array.isArray(data.comparison_sample_names)
      ? data.comparison_sample_names
      : refNames.slice(2);
    const nodeMean = Array.isArray(data.node_js_mean) ? data.node_js_mean : [];
    const edgeMean = Array.isArray(data.edge_js_mean) ? data.edge_js_mean : [];
    const combined = Array.isArray(data.combined_distance) ? data.combined_distance : [];

    const dMean = Array.isArray(data.deltaE_mean) ? data.deltaE_mean : [];
    const dQ25 = Array.isArray(data.deltaE_q25) ? data.deltaE_q25 : [];
    const dQ75 = Array.isArray(data.deltaE_q75) ? data.deltaE_q75 : [];

    const comparisonRefIndices = Array.isArray(data.comparison_ref_indices)
      ? data.comparison_ref_indices
      : refNames.length > 2
        ? refNames.slice(2).map((_, idx) => idx + 2)
        : [];
    const lambdaStarByReference = Array.isArray(data.lambda_star_by_reference)
      ? data.lambda_star_by_reference
      : data.lambda_star !== undefined
        ? [Array.isArray(data.lambda_star) ? data.lambda_star[0] : data.lambda_star]
        : [];
    const matchMinByReference = Array.isArray(data.match_min_by_reference)
      ? data.match_min_by_reference
      : data.match_min !== undefined
        ? [Array.isArray(data.match_min) ? data.match_min[0] : data.match_min]
        : [];

    const orderParam = {
      data: [
        {
          x: lambdas,
          y: dQ75,
          mode: 'lines',
          line: { color: 'rgba(34,211,238,0.1)' },
          name: 'IQR',
          showlegend: false,
        },
        {
          x: lambdas,
          y: dQ25,
          mode: 'lines',
          fill: 'tonexty',
          fillcolor: 'rgba(34,211,238,0.25)',
          line: { color: 'rgba(34,211,238,0.1)' },
          name: 'IQR',
        },
        {
          x: lambdas,
          y: dMean,
          mode: 'lines+markers',
          marker: { color: '#22d3ee' },
          line: { color: '#22d3ee' },
          name: 'mean ΔE',
        },
      ],
      layout: {
        height: 280,
        margin: { l: 50, r: 10, t: 10, b: 40 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#111827' },
        xaxis: { title: 'λ', color: '#111827' },
        yaxis: { title: 'ΔE = E_A - E_B (mean + IQR)', color: '#111827' },
      },
    };

    const nodePlot = {
      data: refNames.map((name, idx) => ({
        x: lambdas,
        y: Array.isArray(nodeMean?.[idx]) ? nodeMean[idx] : [],
        mode: 'lines+markers',
        name,
        marker: { color: pickColor(idx) },
        line: { color: pickColor(idx) },
      })),
      layout: {
        height: 260,
        margin: { l: 50, r: 10, t: 10, b: 40 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#111827' },
        xaxis: { title: 'λ', color: '#111827' },
        yaxis: { title: 'Mean node JS', color: '#111827' },
      },
    };

    const edgePlot = {
      data: refNames.map((name, idx) => ({
        x: lambdas,
        y: Array.isArray(edgeMean?.[idx]) ? edgeMean[idx] : [],
        mode: 'lines+markers',
        name,
        marker: { color: pickColor(idx) },
        line: { color: pickColor(idx) },
      })),
      layout: {
        height: 260,
        margin: { l: 50, r: 10, t: 10, b: 40 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#111827' },
        xaxis: { title: 'λ', color: '#111827' },
        yaxis: { title: 'Mean edge JS', color: '#111827' },
      },
    };

    const matchPlot = {
      data: comparisonRefIndices.flatMap((refIndex, idx) => {
        const color = pickColor(idx + 2);
        const curve = Array.isArray(combined?.[refIndex]) ? combined[refIndex] : [];
        const series = [
          {
            x: lambdas,
            y: curve,
            mode: 'lines+markers',
            name: comparisonNames[idx] || refNames[refIndex] || `comparison ${idx + 1}`,
            marker: { color },
            line: { color },
          },
        ];
        const bestX = lambdaStarByReference[idx];
        const bestY = matchMinByReference[idx];
        if (Number.isFinite(bestX) && Number.isFinite(bestY)) {
          series.push({
            x: [bestX],
            y: [bestY],
            mode: 'markers',
            name: `${comparisonNames[idx] || refNames[refIndex] || `comparison ${idx + 1}`} λ*`,
            marker: { color, size: 10, symbol: 'diamond' },
            showlegend: false,
          });
        }
        return series;
      }),
      layout: {
        height: 260,
        margin: { l: 50, r: 10, t: 10, b: 40 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        font: { color: '#111827' },
        xaxis: { title: 'λ', color: '#111827' },
        yaxis: { title: 'D(λ) vs comparison samples', color: '#111827' },
      },
    };

    return { orderParam, nodePlot, edgePlot, matchPlot };
  }, [analysisData]);

  if (loadingSystem) return <Loader message="Loading lambda sweep…" />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Lambda Sweep: How To Interpret"
        docPath="/docs/lambda_sweep_help.md"
        onClose={() => setHelpOpen(false)}
      />

      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Lambda Sweep</h1>
          <p className="text-sm text-gray-400">
            Sample from interpolated models <code>E_λ</code> between two endpoints and compare against flexible reference and comparison samples.
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
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/visualize`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Back to sampling
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-4">
        <aside className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Cluster</label>
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

            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Endpoint model A (λ=1)</label>
              <select
                value={modelAId}
                onChange={(e) => setModelAId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                {endpointModelOptions.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.name || m.model_id}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Endpoint model B (λ=0)</label>
              <select
                value={modelBId}
                onChange={(e) => setModelBId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                {endpointModelOptions.map((m) => (
                  <option key={m.model_id} value={m.model_id}>
                    {m.name || m.model_id}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-1 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Reference sample A</label>
                <select
                  value={referenceAId}
                  onChange={(e) => setReferenceAId(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  {sampleOptions.map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {sampleLabel(s)}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Reference sample B</label>
                <select
                  value={referenceBId}
                  onChange={(e) => setReferenceBId(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  {sampleOptions.map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {sampleLabel(s)}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Comparison samples</label>
                <select
                  multiple
                  value={comparisonSampleIds}
                  onChange={(e) => {
                    const values = Array.from(e.target.selectedOptions).map((option) => option.value);
                    const filtered = values.filter((id) => id !== referenceAId && id !== referenceBId);
                    setComparisonSampleIds(filtered);
                  }}
                  className="w-full min-h-[10rem] bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  {sampleOptions
                    .filter((s) => s.sample_id !== referenceAId && s.sample_id !== referenceBId)
                    .map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {sampleLabel(s)}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-[11px] text-gray-500">Hold Ctrl/Cmd to select multiple comparison samples.</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">λ count</label>
                <input
                  type="number"
                  min={2}
                  value={lambdaCount}
                  onChange={(e) => setLambdaCount(Number(e.target.value))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">α (node weight)</label>
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                />
              </div>
            </div>

            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Series label</label>
              <input
                value={seriesLabel}
                onChange={(e) => setSeriesLabel(e.target.value)}
                placeholder="Optional"
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              />
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">MD reference label mode</label>
                <select
                  value={mdLabelMode}
                  onChange={(e) => setMdLabelMode(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                >
                  <option value="assigned">assigned</option>
                  <option value="halo">halo</option>
                </select>
              </div>
              <label className="flex items-center gap-2 text-xs text-gray-300 mt-6">
                <input
                  type="checkbox"
                  checked={keepInvalid}
                  onChange={(e) => setKeepInvalid(e.target.checked)}
                />
                Keep invalid
              </label>
            </div>

            <div className="rounded-md border border-gray-800 bg-gray-950/50 p-3 space-y-2">
              <p className="text-xs font-semibold text-gray-200">Sampler</p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Gibbs</label>
                  <select
                    value={gibbsMethod}
                    onChange={(e) => setGibbsMethod(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                  >
                    <option value="rex">replica-exchange</option>
                    <option value="single">single-site</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Target β</label>
                  <input
                    type="number"
                    step={0.05}
                    min={0.01}
                    value={beta}
                    onChange={(e) => setBeta(Number(e.target.value))}
                    className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                  />
                </div>
              </div>

              {gibbsMethod === 'rex' && (
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Rounds</label>
                    <input
                      type="number"
                      min={1}
                      value={rexRounds}
                      onChange={(e) => setRexRounds(Number(e.target.value))}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Burn-in</label>
                    <input
                      type="number"
                      min={1}
                      value={rexBurninRounds}
                      onChange={(e) => setRexBurninRounds(Number(e.target.value))}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Sweeps/round</label>
                    <input
                      type="number"
                      min={1}
                      value={rexSweepsPerRound}
                      onChange={(e) => setRexSweepsPerRound(Number(e.target.value))}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Thin</label>
                    <input
                      type="number"
                      min={1}
                      value={rexThinRounds}
                      onChange={(e) => setRexThinRounds(Number(e.target.value))}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleSubmit}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white text-sm"
                disabled={
                  !selectedClusterId ||
                  !modelAId ||
                  !modelBId ||
                  !referenceAId ||
                  !referenceBId ||
                  comparisonSampleIds.length < 1
                }
              >
                <Play className="h-4 w-4" />
                Run sweep
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
              <p className="text-xs font-semibold text-gray-200">Saved sweeps</p>
              {analysesLoading && <p className="text-[11px] text-gray-500">Loading…</p>}
            </div>
            {analysesError && <ErrorMessage message={analysesError} />}
            {analyses.length === 0 && <p className="text-[11px] text-gray-500">No lambda sweep analyses yet.</p>}
            {analyses.length > 0 && (
              <select
                value={selectedAnalysisId}
                onChange={(e) => setSelectedAnalysisId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                {analyses.map((a) => (
                  <option key={a.analysis_id} value={a.analysis_id}>
                    {(a.series_label || 'lambda_sweep')} · {a.created_at || a.analysis_id}
                  </option>
                ))}
              </select>
            )}
            {selectedAnalysisMeta && (
              <div className="text-[11px] text-gray-400 space-y-1">
                <div>
                  <span className="text-gray-500">models:</span> {selectedAnalysisMeta.model_a_name || selectedAnalysisMeta.model_a_id}{' '}
                  vs {selectedAnalysisMeta.model_b_name || selectedAnalysisMeta.model_b_id}
                </div>
                {Array.isArray(selectedAnalysisMeta.comparison_sample_names) && selectedAnalysisMeta.comparison_sample_names.length > 0 && (
                  <div>
                    <span className="text-gray-500">comparisons:</span> {selectedAnalysisMeta.comparison_sample_names.join(', ')}
                  </div>
                )}
                {selectedAnalysisMeta.summary?.lambda_star !== undefined && (
                  <div>
                    <span className="text-gray-500">λ*:</span> {Number(selectedAnalysisMeta.summary.lambda_star).toFixed(3)}
                  </div>
                )}
              </div>
            )}
          </div>
        </aside>

        <main className="space-y-4">
          {analysisDataLoading && <Loader message="Loading analysis…" />}
          {analysisDataError && <ErrorMessage message={analysisDataError} />}

          {plots && (
            <>
              <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
                <h2 className="text-sm font-semibold text-gray-200">Order Parameter</h2>
                <div className="rounded-md border border-gray-800 bg-white p-3">
                  <Plot
                    data={plots.orderParam.data}
                    layout={plots.orderParam.layout}
                    config={{ displayModeBar: false, responsive: true }}
                    useResizeHandler
                    style={{ width: '100%', height: '280px' }}
                  />
                </div>
              </section>

              <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
                <h2 className="text-sm font-semibold text-gray-200">JS Distances vs Reference Samples</h2>
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                  <div className="rounded-md border border-gray-800 bg-white p-3">
                    <p className="text-xs font-semibold text-gray-800 mb-2">Node JS (mean)</p>
                    <Plot
                      data={plots.nodePlot.data}
                      layout={plots.nodePlot.layout}
                      config={{ displayModeBar: false, responsive: true }}
                      useResizeHandler
                      style={{ width: '100%', height: '260px' }}
                    />
                  </div>
                  <div className="rounded-md border border-gray-800 bg-white p-3">
                    <p className="text-xs font-semibold text-gray-800 mb-2">Edge JS (mean)</p>
                    <Plot
                      data={plots.edgePlot.data}
                      layout={plots.edgePlot.layout}
                      config={{ displayModeBar: false, responsive: true }}
                      useResizeHandler
                      style={{ width: '100%', height: '260px' }}
                    />
                  </div>
                </div>
              </section>

              <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
                <h2 className="text-sm font-semibold text-gray-200">Match Curves</h2>
                <div className="rounded-md border border-gray-800 bg-white p-3">
                  <Plot
                    data={plots.matchPlot.data}
                    layout={plots.matchPlot.layout}
                    config={{ displayModeBar: false, responsive: true }}
                    useResizeHandler
                    style={{ width: '100%', height: '260px' }}
                  />
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
