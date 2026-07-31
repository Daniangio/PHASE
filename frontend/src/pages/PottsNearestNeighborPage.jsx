import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, Play, RefreshCw, Trash2 } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { buildKdeCurve } from '../components/common/EnergyDistributionPlot';
import { deleteClusterAnalysis, fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitPottsNearestNeighborJob } from '../api/jobs';

const palette = ['#22d3ee', '#f97316', '#10b981', '#f43f5e', '#60a5fa', '#f59e0b'];

function weightedQuantile(values, weights, q) {
  if (!Array.isArray(values) || !values.length) return null;
  const rows = values
    .map((value, idx) => ({ value: Number(value), weight: Math.max(0, Number(weights?.[idx] ?? 1)) }))
    .filter((row) => Number.isFinite(row.value) && row.weight > 0)
    .sort((a, b) => a.value - b.value);
  if (!rows.length) return null;
  const total = rows.reduce((sum, row) => sum + row.weight, 0);
  const target = Math.max(0, Math.min(1, Number(q))) * total;
  let acc = 0;
  for (const row of rows) {
    acc += row.weight;
    if (acc >= target) return row.value;
  }
  return rows[rows.length - 1].value;
}

export default function PottsNearestNeighborPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [modelId, setModelId] = useState('');
  const [sampleId, setSampleId] = useState('');
  const [mdSampleId, setMdSampleId] = useState('');
  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [useUnique, setUseUnique] = useState(true);
  const [normalize, setNormalize] = useState(true);
  const [computePerResidue, setComputePerResidue] = useState(true);
  const [alpha, setAlpha] = useState(0.75);
  const [betaNode, setBetaNode] = useState(1.0);
  const [betaEdge, setBetaEdge] = useState(1.0);
  const [topKCandidates, setTopKCandidates] = useState('0');
  const [chunkSize, setChunkSize] = useState(256);
  const [workers, setWorkers] = useState(0);
  const [distanceThresholds, setDistanceThresholds] = useState('0.05,0.1,0.2');
  const [rowCap, setRowCap] = useState('1500');
  const [distanceGraphMode, setDistanceGraphMode] = useState('histogram');

  const [analyses, setAnalyses] = useState([]);
  const [analysesLoading, setAnalysesLoading] = useState(false);
  const [analysesError, setAnalysesError] = useState(null);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [selectedUniqueIndex, setSelectedUniqueIndex] = useState('0');

  const [analysisData, setAnalysisData] = useState(null);
  const [analysisDataError, setAnalysisDataError] = useState(null);
  const cacheRef = useRef({});
  const inFlightRef = useRef({});

  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);
  const [deletingAnalysisId, setDeletingAnalysisId] = useState('');
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

  const clusters = useMemo(
    () => (system?.metastable_clusters || []).filter((run) => run?.cluster_id),
    [system]
  );
  const selectedCluster = useMemo(
    () => clusters.find((cluster) => cluster.cluster_id === selectedClusterId) || null,
    [clusters, selectedClusterId]
  );
  const sampleEntries = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);
  const mdSamples = useMemo(() => sampleEntries.filter((sample) => String(sample?.type || '') === 'md_eval'), [sampleEntries]);
  const candidateSamples = useMemo(
    () => sampleEntries.filter((sample) => String(sample?.sample_id || '').trim()),
    [sampleEntries]
  );
  const models = useMemo(() => selectedCluster?.potts_models || [], [selectedCluster]);

  useEffect(() => {
    if (!clusters.length) return;
    if (!selectedClusterId || !clusters.some((cluster) => cluster.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusters[0].cluster_id);
    }
  }, [clusters, selectedClusterId]);

  useEffect(() => {
    if (!models.length) {
      setModelId('');
      return;
    }
    if (!modelId || !models.some((model) => model.model_id === modelId)) {
      setModelId(models[0].model_id || '');
    }
  }, [models, modelId]);

  useEffect(() => {
    if (!candidateSamples.length) {
      setSampleId('');
      return;
    }
    if (!sampleId || !candidateSamples.some((sample) => sample.sample_id === sampleId)) {
      const preferred = candidateSamples.find((sample) => String(sample?.type || '') !== 'md_eval');
      setSampleId((preferred || candidateSamples[0]).sample_id || '');
    }
  }, [candidateSamples, sampleId]);

  useEffect(() => {
    if (!mdSamples.length) {
      setMdSampleId('');
      return;
    }
    if (!mdSampleId || !mdSamples.some((sample) => sample.sample_id === mdSampleId)) {
      setMdSampleId(mdSamples[0].sample_id || '');
    }
  }, [mdSamples, mdSampleId]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return [];
    setAnalysesLoading(true);
    setAnalysesError(null);
    try {
      const payload = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'potts_nn_mapping' });
      const list = Array.isArray(payload?.analyses) ? payload.analyses : [];
      setAnalyses(list);
      setSelectedAnalysisId((prev) => {
        if (prev && list.some((entry) => entry.analysis_id === prev)) return prev;
        return list[0]?.analysis_id || '';
      });
      return list;
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
      setSelectedAnalysisId('');
      return [];
    } finally {
      setAnalysesLoading(false);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    cacheRef.current = {};
    inFlightRef.current = {};
    setAnalysisData(null);
    setAnalysisDataError(null);
    setSelectedUniqueIndex('0');
    loadAnalyses();
  }, [selectedClusterId, loadAnalyses]);

  useEffect(() => {
    cacheRef.current = {};
    inFlightRef.current = {};
    setAnalysisData(null);
    setAnalysisDataError(null);
    setSelectedUniqueIndex('0');
  }, [rowCap]);

  const loadAnalysisData = useCallback(async (analysisId) => {
    if (!analysisId || !selectedClusterId) return null;
    const maxRows = Number(rowCap) > 0 ? Number(rowCap) : null;
    const key = `${selectedClusterId}:${analysisId}:${maxRows ?? 'all'}`;
    if (Object.prototype.hasOwnProperty.call(cacheRef.current, key)) return cacheRef.current[key];
    if (inFlightRef.current[key]) return inFlightRef.current[key];
    const promise = fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'potts_nn_mapping', analysisId, {
      maxRows,
      sampleSeed: 0,
    })
      .then((payload) => {
        cacheRef.current = { ...cacheRef.current, [key]: payload };
        delete inFlightRef.current[key];
        return payload;
      })
      .catch((err) => {
        delete inFlightRef.current[key];
        throw err;
      });
    inFlightRef.current[key] = promise;
    return promise;
  }, [projectId, rowCap, selectedClusterId, systemId]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedAnalysisId) {
      setAnalysisData(null);
      setAnalysisDataError(null);
      return undefined;
    }
    setAnalysisDataError(null);
    loadAnalysisData(selectedAnalysisId)
      .then((payload) => {
        if (cancelled) return;
        setAnalysisData(payload);
      })
      .catch((err) => {
        if (cancelled) return;
        setAnalysisData(null);
        setAnalysisDataError(err.message || 'Failed to load analysis data.');
      });
    return () => {
      cancelled = true;
    };
  }, [selectedAnalysisId, loadAnalysisData]);

  useEffect(() => {
    const total = Number(analysisData?.data?.sample_unique_sequences?.length || 0);
    if (!total) {
      setSelectedUniqueIndex('0');
      return;
    }
    const current = Number(selectedUniqueIndex);
    if (!Number.isFinite(current) || current < 0 || current >= total) {
      setSelectedUniqueIndex('0');
    }
  }, [analysisData, selectedUniqueIndex]);

  useEffect(() => {
    if (!job?.job_id) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await fetchJobStatus(job.job_id);
        if (cancelled) return;
        setJobStatus(status);
        const state = String(status?.status || '').toLowerCase();
        if (['finished', 'failed', 'stopped'].includes(state)) {
          if (state === 'failed') setJobError(status?.error || 'Analysis failed.');
          const list = await loadAnalyses();
          const analysisId = status?.result?.results?.analysis_id || list[0]?.analysis_id || '';
          if (analysisId) setSelectedAnalysisId(analysisId);
          return;
        }
        window.setTimeout(poll, 1500);
      } catch (err) {
        if (cancelled) return;
        setJobError(err.message || 'Failed to poll job status.');
      }
    };
    poll();
    return () => {
      cancelled = true;
    };
  }, [job, loadAnalyses]);

  const handleRun = async () => {
    if (!selectedClusterId || !modelId || !sampleId || !mdSampleId) {
      setJobError('Select cluster, Potts model, sample, and MD sample.');
      return;
    }
    setJobError(null);
    setJobStatus(null);
    try {
      const payload = await submitPottsNearestNeighborJob({
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        model_id: modelId,
        sample_id: sampleId,
        md_sample_id: mdSampleId,
        md_label_mode: mdLabelMode,
        keep_invalid: false,
        use_unique: !!useUnique,
        normalize: !!normalize,
        compute_per_residue: !!computePerResidue,
        alpha: Number(alpha),
        beta_node: Number(betaNode),
        beta_edge: Number(betaEdge),
        top_k_candidates: Number(topKCandidates) > 0 ? Number(topKCandidates) : null,
        chunk_size: Number(chunkSize),
        workers: Number(workers) > 0 ? Number(workers) : null,
      });
      setJob(payload);
    } catch (err) {
      setJobError(err.message || 'Failed to submit analysis.');
    }
  };

  const handleDeleteAnalysis = async (analysisId) => {
    const ok = window.confirm('Delete this Potts NN mapping analysis?');
    if (!ok) return;
    setDeletingAnalysisId(analysisId);
    setJobError(null);
    try {
      await deleteClusterAnalysis(projectId, systemId, selectedClusterId, 'potts_nn_mapping', analysisId);
      if (selectedAnalysisId === analysisId) {
        setSelectedAnalysisId('');
        setAnalysisData(null);
      }
      await loadAnalyses();
    } catch (err) {
      setJobError(err.message || 'Failed to delete analysis.');
    } finally {
      setDeletingAnalysisId('');
    }
  };

  const histogram = useMemo(() => {
    const global = Array.isArray(analysisData?.data?.nn_dist_global) ? analysisData.data.nn_dist_global.map(Number) : [];
    const counts = Array.isArray(analysisData?.data?.sample_unique_counts) ? analysisData.data.sample_unique_counts.map(Number) : [];
    if (!global.length) return null;
    const median = weightedQuantile(global, counts, 0.5);
    const q9 = weightedQuantile(global, counts, 0.9);
    const curve = buildKdeCurve(global, { weights: counts.length === global.length ? counts : null });
    const data = [];
    if (distanceGraphMode === 'histogram') {
      data.push({
        type: 'histogram',
        histnorm: 'probability density',
        x: global,
        weights: counts.length === global.length ? counts : undefined,
        marker: { color: palette[0] },
        autobinx: true,
        opacity: 0.28,
        name: 'Nearest MD distance histogram',
        showlegend: false,
      });
    }
    if (curve.x.length) {
      data.push({
        type: 'scatter',
        mode: 'lines',
        x: curve.x,
        y: curve.y,
        line: { color: palette[0], width: 2.5 },
        name: 'Fitted distance curve',
      });
    }
    return {
      data,
      layout: {
        paper_bgcolor: '#111827',
        plot_bgcolor: '#111827',
        font: { color: '#e5e7eb' },
        margin: { t: 28, r: 16, b: 48, l: 48 },
        title: distanceGraphMode === 'histogram' ? 'Weighted nearest-neighbor distance distribution' : 'Weighted nearest-neighbor fitted distance curve',
        xaxis: { title: 'Distance' },
        yaxis: { title: 'Probability density' },
        shapes: [
          median != null
            ? { type: 'line', x0: median, x1: median, y0: 0, y1: 1, yref: 'paper', line: { color: '#f97316', width: 2, dash: 'dash' } }
            : null,
          q9 != null
            ? { type: 'line', x0: q9, x1: q9, y0: 0, y1: 1, yref: 'paper', line: { color: '#f43f5e', width: 2, dash: 'dot' } }
            : null,
        ].filter(Boolean),
      },
      config: { responsive: true, displaylogo: false },
    };
  }, [analysisData, distanceGraphMode]);

  const thresholdRows = useMemo(() => {
    const global = Array.isArray(analysisData?.data?.nn_dist_global) ? analysisData.data.nn_dist_global.map(Number) : [];
    const counts = Array.isArray(analysisData?.data?.sample_unique_counts) ? analysisData.data.sample_unique_counts.map(Number) : [];
    if (!global.length) return [];
    const values = String(distanceThresholds || '')
      .split(',')
      .map((value) => Number(value.trim()))
      .filter((value) => Number.isFinite(value))
      .sort((a, b) => a - b);
    const total = counts.length === global.length
      ? counts.reduce((sum, value) => sum + Math.max(0, Number(value || 0)), 0)
      : global.length;
    return values.map((value) => {
      const covered = global.reduce((sum, dist, idx) => {
        if (!(dist <= value)) return sum;
        return sum + (counts.length === global.length ? Math.max(0, Number(counts[idx] || 0)) : 1);
      }, 0);
      return {
        value: Number(value),
        coverage: total > 0 ? covered / total : 0,
      };
    });
  }, [analysisData, distanceThresholds]);

  const residueKeys = useMemo(() => {
    const n = Array.isArray(analysisData?.data?.per_residue_mean) ? analysisData.data.per_residue_mean.length : 0;
    return Array.from({ length: n }, (_, idx) => `res_${idx + 1}`);
  }, [analysisData]);

  const perResidueSeries = useMemo(() => {
    const node = Array.isArray(analysisData?.data?.per_residue_mean_node) ? analysisData.data.per_residue_mean_node.map(Number) : [];
    const edge = Array.isArray(analysisData?.data?.per_residue_mean_edge) ? analysisData.data.per_residue_mean_edge.map(Number) : [];
    if (!node.length && !edge.length) return null;
    const combined = residueKeys.map((_, idx) => (1 - alpha) * Number(node[idx] ?? 0) + alpha * Number(edge[idx] ?? 0));
    const topRows = residueKeys
      .map((key, idx) => ({ key, idx, combined: Number(combined[idx] ?? 0), node: Number(node[idx] ?? 0), edge: Number(edge[idx] ?? 0) }))
      .sort((a, b) => b.combined - a.combined)
      .slice(0, 20);
    return { topRows };
  }, [analysisData, alpha, residueKeys]);

  const selectedUniqueDetails = useMemo(() => {
    const row = Number(selectedUniqueIndex || 0);
    const sampleRows = analysisData?.data?.sample_unique_sequences || [];
    if (!sampleRows.length || row < 0 || row >= sampleRows.length) return null;
    const mdRows = analysisData?.data?.md_unique_sequences || [];
    const mdIdx = Number(analysisData?.data?.nn_md_unique_idx?.[row] ?? -1);
    const residueNode = Array.isArray(analysisData?.data?.nn_dist_residue_node?.[row]) ? analysisData.data.nn_dist_residue_node[row].map(Number) : [];
    const residueEdge = Array.isArray(analysisData?.data?.nn_dist_residue_edge?.[row]) ? analysisData.data.nn_dist_residue_edge[row].map(Number) : [];
    const residueCombined = residueKeys.map((_, idx) => (1 - alpha) * Number(residueNode[idx] ?? 0) + alpha * Number(residueEdge[idx] ?? 0));
    const ranked = residueKeys
      .map((key, idx) => ({ key, idx, combined: Number(residueCombined[idx] ?? 0), node: Number(residueNode[idx] ?? 0), edge: Number(residueEdge[idx] ?? 0) }))
      .sort((a, b) => b.combined - a.combined)
      .slice(0, 15);
    return {
      mdIdx,
      global: Number(analysisData?.data?.nn_dist_global?.[row] ?? NaN),
      node: Number(analysisData?.data?.nn_dist_node?.[row] ?? NaN),
      edge: Number(analysisData?.data?.nn_dist_edge?.[row] ?? NaN),
      count: Number(analysisData?.data?.sample_unique_counts?.[row] ?? 0),
      mdFrameIdx: Number(analysisData?.data?.nn_md_rep_frame_idx?.[row] ?? NaN),
      ranked,
      sample: sampleRows[row],
      md: mdIdx >= 0 ? mdRows[mdIdx] : [],
    };
  }, [analysisData, selectedUniqueIndex, residueKeys, alpha]);

  const selectedResiduePlot = useMemo(() => {
    if (!selectedUniqueDetails?.ranked?.length) return null;
    return {
      data: [
        { type: 'bar', x: selectedUniqueDetails.ranked.map((row) => row.key), y: selectedUniqueDetails.ranked.map((row) => row.combined), marker: { color: palette[2] }, name: 'Combined' },
        { type: 'scatter', mode: 'lines+markers', x: selectedUniqueDetails.ranked.map((row) => row.key), y: selectedUniqueDetails.ranked.map((row) => row.node), marker: { color: palette[1] }, line: { color: palette[1] }, name: 'Node' },
        { type: 'scatter', mode: 'lines+markers', x: selectedUniqueDetails.ranked.map((row) => row.key), y: selectedUniqueDetails.ranked.map((row) => row.edge), marker: { color: palette[4] }, line: { color: palette[4] }, name: 'Edge' },
      ],
      layout: {
        paper_bgcolor: '#111827',
        plot_bgcolor: '#111827',
        font: { color: '#e5e7eb' },
        margin: { t: 28, r: 16, b: 64, l: 48 },
        title: 'Top mismatched residues for selected unique sample',
        xaxis: { tickangle: -45 },
        yaxis: { title: 'Distance' },
      },
      config: { responsive: true, displaylogo: false },
    };
  }, [selectedUniqueDetails]);

  const orderedDistancePlot = useMemo(() => {
    const global = Array.isArray(analysisData?.data?.nn_dist_global) ? analysisData.data.nn_dist_global.map(Number) : [];
    const counts = Array.isArray(analysisData?.data?.sample_unique_counts) ? analysisData.data.sample_unique_counts.map(Number) : [];
    const mdIdx = Array.isArray(analysisData?.data?.nn_md_unique_idx) ? analysisData.data.nn_md_unique_idx.map(Number) : [];
    const mdFrameIdx = Array.isArray(analysisData?.data?.nn_md_rep_frame_idx) ? analysisData.data.nn_md_rep_frame_idx.map(Number) : [];
    const node = Array.isArray(analysisData?.data?.nn_dist_node) ? analysisData.data.nn_dist_node.map(Number) : [];
    const edge = Array.isArray(analysisData?.data?.nn_dist_edge) ? analysisData.data.nn_dist_edge.map(Number) : [];
    if (!global.length) return null;
    const rows = global
      .map((value, idx) => ({
        idx,
        value,
        count: Number(counts[idx] ?? 0),
        mdIdx: Number(mdIdx[idx] ?? -1),
        mdFrameIdx: Number(mdFrameIdx[idx] ?? NaN),
        node: Number(node[idx] ?? NaN),
        edge: Number(edge[idx] ?? NaN),
      }))
      .filter((row) => Number.isFinite(row.value))
      .sort((a, b) => a.value - b.value);
    return {
      data: [
        {
          type: rows.length > 1500 ? 'scattergl' : 'scatter',
          mode: 'lines+markers',
          x: rows.map((_, idx) => idx + 1),
          y: rows.map((row) => row.value),
          customdata: rows.map((row) => [row.idx, row.count, row.mdIdx, row.mdFrameIdx, row.node, row.edge]),
          marker: {
            color: rows.map((row) => row.count),
            colorscale: 'Viridis',
            size: rows.map((row) => Math.max(6, Math.min(18, 6 + Math.log10(Math.max(1, row.count)) * 4))),
            colorbar: { title: 'Count' },
            line: { width: 0 },
          },
          line: { color: '#60a5fa', width: 1.5 },
          hovertemplate:
            'Rank %{x}<br>' +
            'Distance %{y:.4f}<br>' +
            'Unique row %{customdata[0]}<br>' +
            'Count %{customdata[1]}<br>' +
            'Nearest MD unique %{customdata[2]}<br>' +
            'Nearest MD frame %{customdata[3]}<br>' +
            'Node %{customdata[4]:.4f}<br>' +
            'Edge %{customdata[5]:.4f}<extra></extra>',
        },
      ],
      layout: {
        paper_bgcolor: '#111827',
        plot_bgcolor: '#111827',
        font: { color: '#e5e7eb' },
        margin: { t: 28, r: 16, b: 52, l: 56 },
        title: 'Unique sample rows ordered by nearest distance',
        xaxis: { title: 'Rank (closest to farthest)' },
        yaxis: { title: 'Nearest MD distance' },
      },
      config: { responsive: true, displaylogo: false },
      rows,
    };
  }, [analysisData]);

  if (loadingSystem) return <Loader message="Loading Potts nearest-neighbor mapping..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Potts NN Mapping: Help"
        docPath="/docs/potts_nn_mapping_help.md"
        onClose={() => setHelpOpen(false)}
      />

      <div className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Potts NN Mapping</h1>
          <p className="text-sm text-gray-400">Map one sample ensemble to its nearest MD frames in cluster space using Potts-weighted node and edge mismatches.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/potts_nn_mapping_graph${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)}
            className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
          >
            Graph view
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/potts_nn_mapping_compare${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)}
            className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
          >
            Compare experiments
          </button>
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
          >
            <CircleHelp className="h-4 w-4" /> Help
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/visualize`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
          >
            Sampling Explorer
          </button>
          <button
            type="button"
            onClick={loadAnalyses}
            className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
          >
            <RefreshCw className="h-4 w-4" /> Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[320px,minmax(0,1fr)] gap-4">
        <aside className="space-y-4 rounded-lg border border-gray-800 bg-gray-900/70 p-4 h-fit xl:sticky xl:top-6">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-gray-500">Run analysis</p>
          </div>
          <label className="block text-sm text-gray-300">
            Cluster
            <select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white">
              {clusters.map((cluster) => (
                <option key={cluster.cluster_id} value={cluster.cluster_id}>{cluster.name || cluster.cluster_id}</option>
              ))}
            </select>
          </label>
          <label className="block text-sm text-gray-300">
            Potts model
            <select value={modelId} onChange={(e) => setModelId(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white">
              {models.map((model) => (
                <option key={model.model_id} value={model.model_id}>{model.name || model.model_id}</option>
              ))}
            </select>
          </label>
          <label className="block text-sm text-gray-300">
            Sample to map
            <select value={sampleId} onChange={(e) => setSampleId(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white">
              {candidateSamples.map((sample) => (
                <option key={sample.sample_id} value={sample.sample_id}>{sample.name || sample.sample_id} ({sample.type || 'sample'})</option>
              ))}
            </select>
          </label>
          <label className="block text-sm text-gray-300">
            MD sample
            <select value={mdSampleId} onChange={(e) => setMdSampleId(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white">
              {mdSamples.map((sample) => (
                <option key={sample.sample_id} value={sample.sample_id}>{sample.name || sample.sample_id}</option>
              ))}
            </select>
          </label>
          <label className="block text-sm text-gray-300">
            MD label mode
            <select value={mdLabelMode} onChange={(e) => setMdLabelMode(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white">
              <option value="assigned">Assigned</option>
              <option value="halo">Halo</option>
            </select>
          </label>
          <div className="grid grid-cols-2 gap-2 text-sm text-gray-300">
            <label className="flex items-center gap-2"><input type="checkbox" checked={useUnique} onChange={(e) => setUseUnique(e.target.checked)} /> Use unique</label>
            <label className="flex items-center gap-2"><input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} /> Normalize</label>
            <label className="flex items-center gap-2 col-span-2"><input type="checkbox" checked={computePerResidue} onChange={(e) => setComputePerResidue(e.target.checked)} /> Compute per-residue outputs</label>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <label className="block text-sm text-gray-300">Beta node<input type="number" step="0.1" value={betaNode} onChange={(e) => setBetaNode(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white" /></label>
            <label className="block text-sm text-gray-300">Beta edge<input type="number" step="0.1" value={betaEdge} onChange={(e) => setBetaEdge(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white" /></label>
            <label className="block text-sm text-gray-300">Alpha<input type="number" step="0.05" min="0" max="1" value={alpha} onChange={(e) => setAlpha(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white" /></label>
            <label className="block text-sm text-gray-300">Top-K candidates<input type="number" min="0" value={topKCandidates} onChange={(e) => setTopKCandidates(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white" /></label>
            <label className="block text-sm text-gray-300">Chunk size<input type="number" min="1" value={chunkSize} onChange={(e) => setChunkSize(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white" /></label>
            <label className="block text-sm text-gray-300">Workers<input type="number" min="0" value={workers} onChange={(e) => setWorkers(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white" /></label>
          </div>
          <label className="block text-sm text-gray-300">
            Distance thresholds
            <input value={distanceThresholds} onChange={(e) => setDistanceThresholds(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white" />
          </label>
          <label className="block text-sm text-gray-300">
            Max unique rows to load
            <select value={rowCap} onChange={(e) => setRowCap(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white">
              <option value="500">500</option>
              <option value="1000">1000</option>
              <option value="1500">1500</option>
              <option value="3000">3000</option>
              <option value="5000">5000</option>
              <option value="0">All</option>
            </select>
            <p className="mt-1 text-xs text-gray-500">Applies server-side random downsampling to row-wise NN payloads. Aggregate summaries stay exact.</p>
          </label>
          <button
            type="button"
            onClick={handleRun}
            className="w-full inline-flex items-center justify-center gap-2 rounded-md bg-cyan-500 text-slate-950 px-3 py-2 text-sm font-semibold hover:bg-cyan-400"
          >
            <Play className="h-4 w-4" /> Run mapping
          </button>
          {jobStatus && (
            <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3 text-xs text-gray-300">
              <div className="flex items-center justify-between gap-3">
                <span>{jobStatus.status || 'running'}</span>
                <span>{Number(jobStatus.progress ?? 0)}%</span>
              </div>
              <p className="mt-2 text-gray-400">{jobStatus.status_message || jobStatus.status || 'Working...'}</p>
            </div>
          )}
          {(jobError || analysesError) && <ErrorMessage message={jobError || analysesError} />}

          <div className="border-t border-gray-800 pt-4">
            <p className="text-xs uppercase tracking-[0.2em] text-gray-500 mb-2">Saved analyses</p>
            {analysesLoading && <p className="text-xs text-gray-400">Loading…</p>}
            {!analysesLoading && analyses.length === 0 && <p className="text-xs text-gray-500">No analyses yet.</p>}
            <div className="space-y-2 max-h-[50vh] overflow-auto pr-1">
              {analyses.map((analysis) => {
                const active = analysis.analysis_id === selectedAnalysisId;
                return (
                  <button
                    key={analysis.analysis_id}
                    type="button"
                    onClick={() => setSelectedAnalysisId(analysis.analysis_id)}
                    className={`w-full text-left rounded-md border px-3 py-2 ${active ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-800 bg-gray-950/60 hover:bg-gray-800/70'}`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <p className="text-sm text-white">{analysis.model_name || analysis.model_id || 'Potts model'}</p>
                        <p className="text-xs text-gray-400">{analysis.sample_name || analysis.sample_id} → {analysis.md_sample_name || analysis.md_sample_id}</p>
                        <p className="text-[11px] text-gray-500 mt-1">{analysis.created_at || analysis.analysis_id}</p>
                      </div>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          handleDeleteAnalysis(analysis.analysis_id);
                        }}
                        disabled={deletingAnalysisId === analysis.analysis_id}
                        className="text-gray-500 hover:text-rose-300 disabled:opacity-50"
                        aria-label={`Delete analysis ${analysis.analysis_id}`}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </aside>

        <main className="space-y-4 min-w-0">
          {analysisDataError && <ErrorMessage message={analysisDataError} />}
          {!selectedAnalysisId && <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-6 text-sm text-gray-400">Run or select a Potts NN mapping analysis.</div>}
          {selectedAnalysisId && !analysisData && !analysisDataError && <Loader message="Loading analysis…" />}
          {analysisData && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-4"><p className="text-xs text-gray-500 uppercase tracking-[0.2em]">Unique sample rows</p><p className="mt-2 text-2xl text-white font-semibold">{analysisData?.metadata?.summary?.n_sample_unique ?? analysisData?.data?.sample_unique_sequences?.length ?? 0}</p></div>
                <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-4"><p className="text-xs text-gray-500 uppercase tracking-[0.2em]">Unique MD rows</p><p className="mt-2 text-2xl text-white font-semibold">{analysisData?.metadata?.summary?.n_md_unique ?? analysisData?.data?.md_unique_sequences?.length ?? 0}</p></div>
                <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-4"><p className="text-xs text-gray-500 uppercase tracking-[0.2em]">Mean distance</p><p className="mt-2 text-2xl text-white font-semibold">{Number(analysisData?.metadata?.summary?.distance_mean ?? 0).toFixed(3)}</p></div>
                <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-4"><p className="text-xs text-gray-500 uppercase tracking-[0.2em]">Distance max</p><p className="mt-2 text-2xl text-white font-semibold">{Number(analysisData?.metadata?.summary?.distance_max ?? 0).toFixed(3)}</p></div>
              </div>

              <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-4 text-sm text-gray-300">
                {Number(analysisData?.data?.downsampled?.[0] || 0) > 0 ? (
                  <span>Loaded {Number(analysisData?.data?.sampled_unique_row_count?.[0] || analysisData?.data?.sample_unique_sequences?.length || 0)} of {Number(analysisData?.data?.original_unique_row_count?.[0] || analysisData?.metadata?.summary?.n_sample_unique || 0)} unique rows using server-side random sampling.</span>
                ) : (
                  <span>Loaded full analysis payload: {Number(analysisData?.data?.sampled_unique_row_count?.[0] || analysisData?.data?.sample_unique_sequences?.length || 0)} unique rows.</span>
                )}
              </div>

              {histogram && (
                <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-3">
                  <div className="mb-2 flex items-center justify-end gap-2">
                    <label className="text-xs text-gray-400" htmlFor="nn-distance-graph-mode">Distance display</label>
                    <select
                      id="nn-distance-graph-mode"
                      value={distanceGraphMode}
                      onChange={(event) => setDistanceGraphMode(event.target.value)}
                      className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1 text-xs text-gray-100"
                    >
                      <option value="histogram">Histogram + fitted curve</option>
                      <option value="curves">Fitted curve only</option>
                    </select>
                  </div>
                  <Plot data={histogram.data} layout={histogram.layout} config={histogram.config} className="w-full" useResizeHandler style={{ width: '100%', height: 360 }} />
                </div>
              )}

              {orderedDistancePlot && (
                <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-3">
                  <Plot
                    data={orderedDistancePlot.data}
                    layout={orderedDistancePlot.layout}
                    config={orderedDistancePlot.config}
                    className="w-full"
                    useResizeHandler
                    style={{ width: '100%', height: 360 }}
                    onClick={(event) => {
                      const clickedRow = event?.points?.[0]?.customdata?.[0];
                      if (clickedRow != null) {
                        setSelectedUniqueIndex(String(clickedRow));
                      }
                    }}
                  />
                </div>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-[320px,minmax(0,1fr)] gap-4">
                <section className="rounded-lg border border-gray-800 bg-gray-900/70 p-4 space-y-4">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-gray-500">Threshold coverage</p>
                    <div className="mt-3 space-y-2">
                      {thresholdRows.map((row) => (
                        <div key={row.value} className="flex items-center justify-between text-sm text-gray-300">
                          <span>d ≤ {row.value.toFixed(3)}</span>
                          <span>{(100 * row.coverage).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="border-t border-gray-800 pt-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-gray-500">Per-residue alpha</p>
                    <input type="range" min="0" max="1" step="0.01" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="mt-3 w-full" />
                    <p className="mt-2 text-sm text-gray-300">alpha = {Number(alpha).toFixed(2)}</p>
                  </div>
                  {perResidueSeries && (
                    <div className="border-t border-gray-800 pt-4">
                      <p className="text-xs uppercase tracking-[0.2em] text-gray-500 mb-2">Top mean mismatched residues</p>
                      <div className="space-y-2 max-h-80 overflow-auto pr-1">
                        {perResidueSeries.topRows.map((row) => (
                          <div key={row.key} className="flex items-center justify-between gap-3 text-sm text-gray-300">
                            <span>{row.key}</span>
                            <span>{row.combined.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </section>

                <section className="space-y-4 min-w-0">
                  <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-4 space-y-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-xs uppercase tracking-[0.2em] text-gray-500">Selected unique sample</p>
                      <select value={selectedUniqueIndex} onChange={(e) => setSelectedUniqueIndex(e.target.value)} className="rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white">
                        {(analysisData?.data?.sample_unique_sequences || []).map((_, idx) => (
                          <option key={idx} value={idx}>{`unique ${idx} (count ${analysisData?.data?.sample_unique_counts?.[idx] ?? 0})`}</option>
                        ))}
                      </select>
                    </div>
                    {selectedUniqueDetails && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                        <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3"><p className="text-gray-500">Global</p><p className="mt-1 text-white font-semibold">{selectedUniqueDetails.global.toFixed(3)}</p></div>
                        <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3"><p className="text-gray-500">Node</p><p className="mt-1 text-white font-semibold">{selectedUniqueDetails.node.toFixed(3)}</p></div>
                        <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3"><p className="text-gray-500">Edge</p><p className="mt-1 text-white font-semibold">{selectedUniqueDetails.edge.toFixed(3)}</p></div>
                        <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3"><p className="text-gray-500">Nearest MD frame</p><p className="mt-1 text-white font-semibold">{Number.isFinite(selectedUniqueDetails.mdFrameIdx) ? selectedUniqueDetails.mdFrameIdx : '-'}</p></div>
                      </div>
                    )}
                  </div>
                  {selectedResiduePlot && (
                    <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-3">
                      <Plot data={selectedResiduePlot.data} layout={selectedResiduePlot.layout} config={selectedResiduePlot.config} className="w-full" useResizeHandler style={{ width: '100%', height: 420 }} />
                    </div>
                  )}
                </section>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
