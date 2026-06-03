import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, Play, RefreshCw } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import EnergyDistributionPlot, {
  EnergySeriesSelectorButton,
  buildEnergyDistributionPlot,
  useEnergySeriesSelection,
} from '../components/common/EnergyDistributionPlot';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchPottsClusterInfo, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitDeltaEnergyJob, submitEndpointFrustrationJob } from '../api/jobs';

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function rgbToHex(r, g, b) {
  const to = (v) => {
    const x = Math.max(0, Math.min(255, Math.round(v)));
    return x.toString(16).padStart(2, '0');
  };
  return `#${to(r)}${to(g)}${to(b)}`;
}

function colorDiverging(value, maxAbs = 0.5) {
  const scale = Math.max(1e-6, Number(maxAbs) || 0.5);
  const x = Math.max(-1, Math.min(1, Number(value) / scale));
  if (x >= 0) {
    return rgbToHex(lerp(245, 190, x), lerp(245, 24, x), lerp(245, 24, x));
  }
  return rgbToHex(lerp(245, 49, -x), lerp(245, 112, -x), lerp(245, 204, -x));
}

function colorDivergingGreenRed(value, maxAbs = 0.5) {
  const scale = Math.max(1e-6, Number(maxAbs) || 0.5);
  const x = Math.max(-1, Math.min(1, Number(value) / scale));
  if (x >= 0) {
    return rgbToHex(lerp(245, 239, x), lerp(245, 68, x), lerp(245, 68, x));
  }
  return rgbToHex(lerp(245, 34, -x), lerp(245, 197, -x), lerp(245, 94, -x));
}

function colorSequentialGreenRed(value, minValue, maxValue) {
  const minV = Number.isFinite(Number(minValue)) ? Number(minValue) : 0;
  const maxV = Number.isFinite(Number(maxValue)) ? Number(maxValue) : 1;
  const span = Math.max(1e-6, maxV - minV);
  const t = clamp01((Number(value) - minV) / span);
  if (t <= 0.5) {
    const u = t / 0.5;
    return rgbToHex(lerp(34, 250, u), lerp(197, 204, u), lerp(94, 21, u));
  }
  const u = (t - 0.5) / 0.5;
  return rgbToHex(lerp(250, 239, u), lerp(204, 68, u), lerp(21, 68, u));
}

function edgeLabel(edge, residueLabels) {
  if (!Array.isArray(edge) || edge.length < 2) return '';
  const r = Number(edge[0]);
  const s = Number(edge[1]);
  const a = residueLabels[r] ?? String(r);
  const b = residueLabels[s] ?? String(s);
  return `${a}-${b}`;
}

function buildSamplingSuffix(clusterId) {
  const params = new URLSearchParams();
  if (clusterId) params.set('cluster_id', clusterId);
  const q = params.toString();
  return q ? `?${q}` : '';
}

function parseSearchClusterId(search) {
  const params = new URLSearchParams(search || '');
  return String(params.get('cluster_id') || '').trim();
}

function blendResidueMetric(values, edgeValues, topEdgeIndices, edges, dEdge, alpha) {
  if (!Array.isArray(values) || !values.length) return [];
  if (!Array.isArray(edgeValues) || !edgeValues.length || !Array.isArray(topEdgeIndices) || !Array.isArray(edges)) {
    return values.map((v) => Number(v));
  }
  const nResidues = values.length;
  const strength = clamp01(Number(alpha));
  if (strength <= 0) return values.map((v) => Number(v));
  const sumW = new Array(nResidues).fill(0);
  const sumWV = new Array(nResidues).fill(0);
  for (let col = 0; col < topEdgeIndices.length && col < edgeValues.length; col += 1) {
    const eidx = Number(topEdgeIndices[col]);
    const edge = edges[eidx];
    if (!Array.isArray(edge) || edge.length < 2) continue;
    const r = Number(edge[0]);
    const s = Number(edge[1]);
    if (!Number.isInteger(r) || !Number.isInteger(s) || r < 0 || s < 0 || r >= nResidues || s >= nResidues) continue;
    const v = Number(edgeValues[col]);
    if (!Number.isFinite(v)) continue;
    const wRaw = Array.isArray(dEdge) && Number.isFinite(Number(dEdge[eidx])) ? Math.abs(Number(dEdge[eidx])) : 1;
    const w = wRaw > 1e-9 ? wRaw : 1;
    sumW[r] += w;
    sumWV[r] += w * v;
    sumW[s] += w;
    sumWV[s] += w * v;
  }
  return values.map((raw, idx) => {
    const base = Number(raw);
    if (!Number.isFinite(base)) return NaN;
    const edgeMean = sumW[idx] > 0 ? sumWV[idx] / sumW[idx] : base;
    return (1 - strength) * base + strength * edgeMean;
  });
}

function metricScale(values, centered = false) {
  const finite = (values || []).map(Number).filter((v) => Number.isFinite(v));
  if (!finite.length) return centered ? 0.5 : 1;
  const maxAbs = Math.max(...finite.map((v) => Math.abs(v)));
  const maxVal = Math.max(...finite);
  return centered ? Math.max(0.05, maxAbs) : Math.max(0.1, maxVal);
}

function metricMinMax(values) {
  const finite = (values || []).map(Number).filter((v) => Number.isFinite(v));
  if (!finite.length) return { min: 0, max: 1 };
  return { min: Math.min(...finite), max: Math.max(...finite) };
}

function makeHorizontalBar(values, labels, colors, title, xTitle, limit) {
  const rows = labels.map((label, idx) => ({ label, value: Number(values[idx]), color: colors[idx] })).filter((row) => Number.isFinite(row.value));
  rows.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const shown = rows.slice(0, Math.max(1, Number(limit) || rows.length));
  return {
    data: [
      {
        type: 'bar',
        orientation: 'h',
        x: shown.map((row) => row.value).reverse(),
        y: shown.map((row) => row.label).reverse(),
        marker: { color: shown.map((row) => row.color).reverse() },
        hovertemplate: '%{y}<br>%{x:.4f}<extra></extra>',
      },
    ],
    layout: {
      title,
      paper_bgcolor: '#0f172a',
      plot_bgcolor: '#0f172a',
      font: { color: '#e5e7eb' },
      height: Math.max(320, shown.length * 18),
      margin: { l: 140, r: 24, t: 40, b: 48 },
      xaxis: { title: xTitle, gridcolor: '#1f2937', zerolinecolor: '#374151' },
      yaxis: { automargin: true, tickfont: { size: 11 } },
    },
    config: { displaylogo: false, responsive: true },
  };
}

export default function DeltaEvalPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [clusterInfo, setClusterInfo] = useState(null);
  const [clusterInfoError, setClusterInfoError] = useState(null);

  const [modelAId, setModelAId] = useState('');
  const [modelBId, setModelBId] = useState('');
  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [keepInvalid, setKeepInvalid] = useState(false);
  const [topKEdges, setTopKEdges] = useState(2000);
  const [workers, setWorkers] = useState(0);
  const [selectedSampleIds, setSelectedSampleIds] = useState([]);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [selectedSampleId, setSelectedSampleId] = useState('');
  const [commitmentCentered, setCommitmentCentered] = useState(true);
  const [frustrationChannel, setFrustrationChannel] = useState('sym');
  const [frustrationDisplayMode, setFrustrationDisplayMode] = useState('raw');
  const [referenceSampleIds, setReferenceSampleIds] = useState([]);
  const [edgeBlendEnabled, setEdgeBlendEnabled] = useState(false);
  const [edgeBlendStrength, setEdgeBlendStrength] = useState(0.75);
  const [residueLimit, setResidueLimit] = useState(60);
  const [edgeLimit, setEdgeLimit] = useState(80);
  const [deltaEnergyGraphMode, setDeltaEnergyGraphMode] = useState('histogram');
  const [activeTab, setActiveTab] = useState('delta_energy');
  const [runPanelOpen, setRunPanelOpen] = useState(false);
  const [deltaEnergySeed, setDeltaEnergySeed] = useState(0);
  const [deltaEnergyBins, setDeltaEnergyBins] = useState(80);
  const [sampleFrameModes, setSampleFrameModes] = useState({});
  const [sampleFrameLimits, setSampleFrameLimits] = useState({});
  const [helpOpen, setHelpOpen] = useState(false);

  const [analyses, setAnalyses] = useState([]);
  const [analysesError, setAnalysesError] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisDataError, setAnalysisDataError] = useState(null);
  const [analysisDataLoading, setAnalysisDataLoading] = useState(false);
  const analysisDataCacheRef = useRef({});

  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);

  useEffect(() => {
    const loadSystem = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        setSystem(await fetchSystem(projectId, systemId));
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
  const sampleEntries = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);
  const sampleFrameCounts = useMemo(() => {
    const out = {};
    sampleEntries.forEach((sample) => {
      const sid = String(sample?.sample_id || '');
      if (!sid) return;
      const count =
        Number(sample?.n_frames) ||
        Number(sample?.frame_count) ||
        Number(sample?.frames) ||
        Number(sample?.metadata?.n_frames) ||
        Number(sample?.summary?.n_frames) ||
        0;
      out[sid] = Number.isFinite(count) && count > 0 ? Math.round(count) : 0;
    });
    return out;
  }, [sampleEntries]);
  const pottsModels = useMemo(() => selectedCluster?.potts_models || [], [selectedCluster]);
  const deltaModels = useMemo(
    () =>
      pottsModels.filter((m) => {
        const params = m.params || {};
        if (params.fit_mode === 'delta') return true;
        const kind = params.delta_kind || '';
        return typeof kind === 'string' && kind.startsWith('delta_');
      }),
    [pottsModels]
  );

  useEffect(() => {
    if (!clusterOptions.length) return;
    const requested = parseSearchClusterId(location.search);
    if (requested && clusterOptions.some((c) => c.cluster_id === requested)) {
      if (requested !== selectedClusterId) setSelectedClusterId(requested);
      return;
    }
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterId, location.search]);

  useEffect(() => {
    if (!deltaModels.length) {
      setModelAId('');
      setModelBId('');
      return;
    }
    const ids = new Set(deltaModels.map((m) => String(m.model_id)));
    let nextA = ids.has(String(modelAId)) ? String(modelAId) : String(deltaModels[0].model_id);
    let nextB = ids.has(String(modelBId)) ? String(modelBId) : '';
    if (!nextB || nextA === nextB) {
      const alt = deltaModels.find((m) => String(m.model_id) !== nextA);
      nextB = String((alt || deltaModels[0]).model_id);
    }
    if (nextA !== modelAId) setModelAId(nextA);
    if (nextB !== modelBId) setModelBId(nextB);
  }, [deltaModels, modelAId, modelBId]);

  useEffect(() => {
    if (!sampleEntries.length || selectedSampleIds.length) return;
    const md = sampleEntries.filter((s) => String(s?.type || '').toLowerCase() === 'md_eval').map((s) => String(s.sample_id));
    setSelectedSampleIds(md.length ? md : sampleEntries.slice(0, 3).map((s) => String(s.sample_id)));
  }, [sampleEntries, selectedSampleIds.length]);

  const loadClusterInfo = useCallback(async () => {
    if (!selectedClusterId) return;
    setClusterInfoError(null);
    try {
      const info = await fetchPottsClusterInfo(projectId, systemId, selectedClusterId, { modelId: modelAId || undefined });
      setClusterInfo(info);
    } catch (err) {
      setClusterInfoError(err.message || 'Failed to load cluster info.');
      setClusterInfo(null);
    }
  }, [projectId, systemId, selectedClusterId, modelAId]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesError(null);
    try {
      const res = await fetchClusterAnalyses(projectId, systemId, selectedClusterId);
      setAnalyses(Array.isArray(res?.analyses) ? res.analyses : []);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    analysisDataCacheRef.current = {};
    setAnalysisData(null);
    setSelectedAnalysisId('');
    setSelectedSampleId('');
    loadClusterInfo();
    loadAnalyses();
  }, [selectedClusterId, loadClusterInfo, loadAnalyses]);

  const activeAnalysisType = activeTab === 'delta_energy' ? 'delta_energy' : 'endpoint_frustration';

  const matchingAnalyses = useMemo(() => {
    return analyses
      .filter((a) => String(a?.analysis_type || '') === activeAnalysisType)
      .sort((x, y) => {
        const tx = Date.parse(String(x?.updated_at || x?.created_at || ''));
        const ty = Date.parse(String(y?.updated_at || y?.created_at || ''));
        return (Number.isFinite(ty) ? ty : 0) - (Number.isFinite(tx) ? tx : 0);
      });
  }, [analyses, activeAnalysisType]);

  useEffect(() => {
    if (!matchingAnalyses.length) {
      setSelectedAnalysisId('');
      return;
    }
    if (!matchingAnalyses.some((a) => String(a.analysis_id) === String(selectedAnalysisId))) {
      setSelectedAnalysisId(String(matchingAnalyses[0].analysis_id));
    }
  }, [matchingAnalyses, selectedAnalysisId]);

  const selectedAnalysisMeta = useMemo(
    () => matchingAnalyses.find((a) => String(a.analysis_id) === String(selectedAnalysisId)) || matchingAnalyses[0] || null,
    [matchingAnalyses, selectedAnalysisId]
  );

  const loadAnalysisData = useCallback(
    async (analysisId) => {
      if (!analysisId) return null;
      const cacheKey = `${activeAnalysisType}:${analysisId}`;
      if (Object.prototype.hasOwnProperty.call(analysisDataCacheRef.current, cacheKey)) {
        return analysisDataCacheRef.current[cacheKey];
      }
      const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, activeAnalysisType, analysisId);
      analysisDataCacheRef.current = { ...analysisDataCacheRef.current, [cacheKey]: payload };
      return payload;
    },
    [projectId, systemId, selectedClusterId, activeAnalysisType]
  );

  useEffect(() => {
    const run = async () => {
      setAnalysisData(null);
      setAnalysisDataError(null);
      if (!selectedAnalysisMeta?.analysis_id) return;
      setAnalysisDataLoading(true);
      try {
        setAnalysisData(await loadAnalysisData(selectedAnalysisMeta.analysis_id));
      } catch (err) {
        setAnalysisDataError(err.message || 'Failed to load analysis data.');
      } finally {
        setAnalysisDataLoading(false);
      }
    };
    run();
  }, [selectedAnalysisMeta, loadAnalysisData]);

  useEffect(() => {
    if (!job?.job_id) return;
    let cancelled = false;
    const timer = setInterval(async () => {
      try {
        const status = await fetchJobStatus(job.job_id);
        if (cancelled) return;
        setJobStatus(status);
        if (['finished', 'failed', 'canceled'].includes(String(status?.status || ''))) {
          clearInterval(timer);
          if (status?.status === 'finished') {
            await loadAnalyses();
          }
        }
      } catch (err) {
        if (!cancelled) setJobError(err.message || 'Failed to poll job.');
      }
    }, 2000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job, loadAnalyses]);

  const handleRun = useCallback(async () => {
    setJobError(null);
    try {
      const basePayload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        model_a_id: modelAId,
        model_b_id: modelBId,
        sample_ids: selectedSampleIds,
        md_label_mode: mdLabelMode,
        keep_invalid: keepInvalid,
      };
      const frameLimits = {};
      selectedSampleIds.forEach((sid) => {
        if (sampleFrameModes[sid] === 'random') {
          const limit = Math.max(0, Number(sampleFrameLimits[sid]) || 0);
          if (limit > 0) frameLimits[sid] = limit;
        }
      });
      const res = activeTab === 'delta_energy' ? await submitDeltaEnergyJob({
        ...basePayload,
        frame_limits: frameLimits,
        seed: Math.max(0, Number(deltaEnergySeed) || 0),
        energy_bins: Math.max(5, Number(deltaEnergyBins) || 80),
        workers: Math.max(0, Number(workers) || 0),
      }) : await submitEndpointFrustrationJob({
        ...basePayload,
        top_k_edges: Number(topKEdges),
        workers: Math.max(0, Number(workers) || 0),
      });
      setJob(res);
      setJobStatus(null);
      setRunPanelOpen(false);
    } catch (err) {
      setJobError(err.message || 'Failed to submit analysis job.');
    }
  }, [
    activeTab,
    projectId,
    systemId,
    selectedClusterId,
    modelAId,
    modelBId,
    selectedSampleIds,
    mdLabelMode,
    keepInvalid,
    sampleFrameModes,
    sampleFrameLimits,
    deltaEnergySeed,
    deltaEnergyBins,
    topKEdges,
    workers,
  ]);

  const jobProgress = Number(jobStatus?.meta?.progress ?? jobStatus?.progress ?? 0);
  const jobStatusLabel = String(jobStatus?.meta?.status || jobStatus?.status || 'queued');

  const residueLabels = useMemo(() => {
    const keys = clusterInfo?.residue_keys || [];
    if (Array.isArray(keys) && keys.length) return keys.map(String);
    const n = Number(clusterInfo?.n_residues || 0);
    return Array.from({ length: n }, (_, idx) => `res_${idx}`);
  }, [clusterInfo]);

  const sampleIds = useMemo(() => (Array.isArray(analysisData?.data?.sample_ids) ? analysisData.data.sample_ids.map(String) : []), [analysisData]);
  const sampleLabels = useMemo(() => {
    const raw = Array.isArray(analysisData?.data?.sample_labels) ? analysisData.data.sample_labels.map(String) : [];
    return sampleIds.map((sid, idx) => raw[idx] || sid);
  }, [analysisData, sampleIds]);
  const sampleTypes = useMemo(() => {
    const raw = Array.isArray(analysisData?.data?.sample_types) ? analysisData.data.sample_types.map(String) : [];
    return sampleIds.map((sid, idx) => raw[idx] || '');
  }, [analysisData, sampleIds]);
  const sampleIndexById = useMemo(() => {
    const map = new Map();
    sampleIds.forEach((sid, idx) => map.set(String(sid), idx));
    return map;
  }, [sampleIds]);

  useEffect(() => {
    if (!sampleIds.length) {
      setSelectedSampleId('');
      return;
    }
    if (!selectedSampleId || !sampleIndexById.has(String(selectedSampleId))) {
      setSelectedSampleId(String(sampleIds[0]));
    }
  }, [sampleIds, selectedSampleId, sampleIndexById]);

  useEffect(() => {
    if (frustrationDisplayMode !== 'centered') return;
    if (referenceSampleIds.length) return;
    if (!sampleIds.length) return;
    const md = sampleIds.filter((sid, idx) => String(sampleTypes[idx] || '').toLowerCase().includes('md'));
    if (md.length) setReferenceSampleIds(md);
    else setReferenceSampleIds([sampleIds[0]]);
  }, [frustrationDisplayMode, referenceSampleIds.length, sampleIds, sampleTypes]);

  const selectedSampleIndex = useMemo(() => sampleIndexById.get(String(selectedSampleId)), [sampleIndexById, selectedSampleId]);
  const frustrationReferenceIdxs = useMemo(
    () =>
      referenceSampleIds
        .map((sid) => sampleIndexById.get(String(sid)))
        .filter((idx) => Number.isInteger(idx) && idx >= 0),
    [referenceSampleIds, sampleIndexById]
  );
  const topEdgeIndices = useMemo(
    () => (Array.isArray(analysisData?.data?.top_edge_indices) ? analysisData.data.top_edge_indices.map((v) => Number(v)) : []),
    [analysisData]
  );
  const edgesAll = useMemo(() => (Array.isArray(analysisData?.data?.edges) ? analysisData.data.edges : []), [analysisData]);
  const dEdge = useMemo(() => (Array.isArray(analysisData?.data?.D_edge) ? analysisData.data.D_edge : []), [analysisData]);

  const commitmentResidue = useMemo(() => {
    if (!Number.isInteger(selectedSampleIndex)) return [];
    const matrix = Array.isArray(analysisData?.data?.q_residue_all) ? analysisData.data.q_residue_all : [];
    const row = Array.isArray(matrix[selectedSampleIndex]) ? matrix[selectedSampleIndex].map((v) => Number(v)) : [];
    return commitmentCentered ? row.map((v) => (Number.isFinite(v) ? v - 0.5 : NaN)) : row;
  }, [analysisData, selectedSampleIndex, commitmentCentered]);

  const commitmentEdge = useMemo(() => {
    if (!Number.isInteger(selectedSampleIndex)) return [];
    const matrix = Array.isArray(analysisData?.data?.q_edge) ? analysisData.data.q_edge : [];
    const row = Array.isArray(matrix[selectedSampleIndex]) ? matrix[selectedSampleIndex].map((v) => Number(v)) : [];
    return commitmentCentered ? row.map((v) => (Number.isFinite(v) ? v - 0.5 : NaN)) : row;
  }, [analysisData, selectedSampleIndex, commitmentCentered]);

  const frustrationResidueRaw = useMemo(() => {
    if (!Number.isInteger(selectedSampleIndex)) return [];
    const key = frustrationChannel === 'pol' ? 'frustration_node_pol_mean' : 'frustration_node_sym_mean';
    const matrix = Array.isArray(analysisData?.data?.[key]) ? analysisData.data[key] : [];
    const row = Array.isArray(matrix[selectedSampleIndex]) ? matrix[selectedSampleIndex].map((v) => Number(v)) : [];
    if (frustrationDisplayMode !== 'centered' || !frustrationReferenceIdxs.length) return row;
    const ref = new Array(row.length).fill(0);
    const counts = new Array(row.length).fill(0);
    for (const ridx of frustrationReferenceIdxs) {
      const refRow = Array.isArray(matrix[ridx]) ? matrix[ridx] : null;
      if (!refRow) continue;
      for (let i = 0; i < Math.min(row.length, refRow.length); i += 1) {
        const v = Number(refRow[i]);
        if (!Number.isFinite(v)) continue;
        ref[i] += v;
        counts[i] += 1;
      }
    }
    return row.map((v, i) => {
      const mean = counts[i] > 0 ? ref[i] / counts[i] : NaN;
      return Number.isFinite(v) && Number.isFinite(mean) ? v - mean : v;
    });
  }, [analysisData, selectedSampleIndex, frustrationChannel, frustrationDisplayMode, frustrationReferenceIdxs]);

  const frustrationEdgeRaw = useMemo(() => {
    if (!Number.isInteger(selectedSampleIndex)) return [];
    const key = frustrationChannel === 'pol' ? 'frustration_edge_pol_mean' : 'frustration_edge_sym_mean';
    const matrix = Array.isArray(analysisData?.data?.[key]) ? analysisData.data[key] : [];
    const row = Array.isArray(matrix[selectedSampleIndex]) ? matrix[selectedSampleIndex].map((v) => Number(v)) : [];
    if (frustrationDisplayMode !== 'centered' || !frustrationReferenceIdxs.length) return row;
    const ref = new Array(row.length).fill(0);
    const counts = new Array(row.length).fill(0);
    for (const ridx of frustrationReferenceIdxs) {
      const refRow = Array.isArray(matrix[ridx]) ? matrix[ridx] : null;
      if (!refRow) continue;
      for (let i = 0; i < Math.min(row.length, refRow.length); i += 1) {
        const v = Number(refRow[i]);
        if (!Number.isFinite(v)) continue;
        ref[i] += v;
        counts[i] += 1;
      }
    }
    return row.map((v, i) => {
      const mean = counts[i] > 0 ? ref[i] / counts[i] : NaN;
      return Number.isFinite(v) && Number.isFinite(mean) ? v - mean : v;
    });
  }, [analysisData, selectedSampleIndex, frustrationChannel, frustrationDisplayMode, frustrationReferenceIdxs]);

  const commitmentResidueBlended = useMemo(
    () =>
      edgeBlendEnabled
        ? blendResidueMetric(commitmentResidue, commitmentEdge, topEdgeIndices, edgesAll, dEdge, edgeBlendStrength)
        : commitmentResidue,
    [commitmentResidue, commitmentEdge, topEdgeIndices, edgesAll, dEdge, edgeBlendEnabled, edgeBlendStrength]
  );
  const frustrationResidueBlended = useMemo(
    () =>
      edgeBlendEnabled
        ? blendResidueMetric(frustrationResidueRaw, frustrationEdgeRaw, topEdgeIndices, edgesAll, dEdge, edgeBlendStrength)
        : frustrationResidueRaw,
    [frustrationResidueRaw, frustrationEdgeRaw, topEdgeIndices, edgesAll, dEdge, edgeBlendEnabled, edgeBlendStrength]
  );

  const commitmentResidueColors = useMemo(() => {
    const scale = metricScale(commitmentResidueBlended, true);
    return commitmentResidueBlended.map((v) => colorDiverging(v, scale));
  }, [commitmentResidueBlended]);
  const commitmentEdgeColors = useMemo(() => {
    const scale = metricScale(commitmentEdge, true);
    return commitmentEdge.map((v) => colorDiverging(v, scale));
  }, [commitmentEdge]);
  const frustrationResidueColors = useMemo(() => {
    if (frustrationChannel === 'pol') {
      const scale = metricScale(frustrationResidueBlended, true);
      return frustrationResidueBlended.map((v) => colorDiverging(-v, scale));
    }
    if (frustrationDisplayMode === 'centered') {
      const scale = metricScale(frustrationResidueBlended, true);
      return frustrationResidueBlended.map((v) => colorDivergingGreenRed(v, scale));
    }
    const { min, max } = metricMinMax(frustrationResidueBlended);
    return frustrationResidueBlended.map((v) => colorSequentialGreenRed(v, min, max));
  }, [frustrationResidueBlended, frustrationChannel, frustrationDisplayMode]);
  const frustrationEdgeColors = useMemo(() => {
    if (frustrationChannel === 'pol') {
      const scale = metricScale(frustrationEdgeRaw, true);
      return frustrationEdgeRaw.map((v) => colorDiverging(-v, scale));
    }
    if (frustrationDisplayMode === 'centered') {
      const scale = metricScale(frustrationEdgeRaw, true);
      return frustrationEdgeRaw.map((v) => colorDivergingGreenRed(v, scale));
    }
    const { min, max } = metricMinMax(frustrationEdgeRaw);
    return frustrationEdgeRaw.map((v) => colorSequentialGreenRed(v, min, max));
  }, [frustrationEdgeRaw, frustrationChannel, frustrationDisplayMode]);

  const edgeLabels = useMemo(
    () => topEdgeIndices.map((raw) => edgeLabel(edgesAll[Number(raw)], residueLabels) || `edge_${raw}`),
    [topEdgeIndices, edgesAll, residueLabels]
  );

  const frustrationAxisLabel = useMemo(() => {
    if (frustrationChannel === 'pol') {
      return frustrationDisplayMode === 'centered'
        ? 'centered polarity frustration (red = more A-like, blue = more B-like)'
        : 'polarity frustration (red = more A-like, blue = more B-like)';
    }
    return frustrationDisplayMode === 'centered'
      ? 'centered symmetric frustration (green = lower than ref, red = higher than ref)'
      : 'symmetric frustration (green = low, red = high)';
  }, [frustrationChannel, frustrationDisplayMode]);

  const residueCommitmentPlot = useMemo(
    () =>
      makeHorizontalBar(
        commitmentResidueBlended,
        residueLabels,
        commitmentResidueColors,
        edgeBlendEnabled ? 'Residue commitment (edge-weighted blend)' : 'Residue commitment',
        commitmentCentered ? 'centered commitment (q - 0.5)' : 'commitment q',
        residueLimit
      ),
    [commitmentResidueBlended, residueLabels, commitmentResidueColors, edgeBlendEnabled, commitmentCentered, residueLimit]
  );
  const edgeCommitmentPlot = useMemo(
    () =>
      makeHorizontalBar(
        commitmentEdge,
        edgeLabels,
        commitmentEdgeColors,
        'Edge commitment',
        commitmentCentered ? 'centered commitment (q - 0.5)' : 'commitment q',
        edgeLimit
      ),
    [commitmentEdge, edgeLabels, commitmentEdgeColors, commitmentCentered, edgeLimit]
  );
  const residueFrustrationPlot = useMemo(
    () =>
      makeHorizontalBar(
        frustrationResidueBlended,
        residueLabels,
        frustrationResidueColors,
        edgeBlendEnabled ? 'Residue frustration (edge-weighted blend)' : 'Residue frustration',
        frustrationAxisLabel,
        residueLimit
      ),
    [frustrationResidueBlended, residueLabels, frustrationResidueColors, edgeBlendEnabled, frustrationAxisLabel, residueLimit]
  );
  const edgeFrustrationPlot = useMemo(
    () =>
      makeHorizontalBar(
        frustrationEdgeRaw,
        edgeLabels,
        frustrationEdgeColors,
        'Edge frustration',
        frustrationAxisLabel,
        edgeLimit
      ),
    [frustrationEdgeRaw, edgeLabels, frustrationEdgeColors, frustrationAxisLabel, edgeLimit]
  );

  const selectedSampleName = useMemo(() => {
    if (!Number.isInteger(selectedSampleIndex)) return '';
    return sampleLabels[selectedSampleIndex] || selectedSampleId;
  }, [selectedSampleIndex, sampleLabels, selectedSampleId]);

  const deltaEnergySeries = useMemo(() => {
    const bins = Array.isArray(analysisData?.data?.delta_energy_bins)
      ? analysisData.data.delta_energy_bins.map(Number)
      : [];
    const hist = Array.isArray(analysisData?.data?.delta_energy_hist) ? analysisData.data.delta_energy_hist : [];
    if (bins.length < 2 || !hist.length) return [];
    return hist.map((row, idx) => ({
      id: sampleIds[idx] || `sample-${idx + 1}`,
      sample_id: sampleIds[idx] || '',
      label: sampleLabels[idx] || sampleIds[idx] || `sample ${idx + 1}`,
      kind: sampleTypes[idx] || 'sample',
      type: sampleTypes[idx] || 'sample',
      bins,
      density: Array.isArray(row) ? row.map(Number) : [],
    }));
  }, [analysisData, sampleLabels, sampleIds, sampleTypes]);

  const {
    selectedIds: selectedDeltaEnergySeriesIds,
    setSelectedIds: setSelectedDeltaEnergySeriesIds,
    selectedSeries: visibleDeltaEnergySeries,
  } = useEnergySeriesSelection(deltaEnergySeries);

  const deltaEnergyPlot = useMemo(() => buildEnergyDistributionPlot({
    series: visibleDeltaEnergySeries,
    mode: deltaEnergyGraphMode,
    title: 'ΔE distributions over selected trajectories',
    xTitle: 'ΔE = E_model_A - E_model_B',
    height: 340,
    background: 'dark',
  }), [visibleDeltaEnergySeries, deltaEnergyGraphMode]);

  const analysisSummary = selectedAnalysisMeta?.summary || {};

  if (loadingSystem) return <Loader message="Loading model-pair analysis..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-6">
      <HelpDrawer
        open={helpOpen}
        onClose={() => setHelpOpen(false)}
        title="Model-pair analysis help"
        docPath="/docs/endpoint_frustration.md"
      />

      {runPanelOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-lg border border-gray-700 bg-gray-900 p-4 shadow-xl">
            <div className="flex items-start justify-between gap-3 border-b border-gray-800 pb-3">
              <div>
                <h2 className="text-lg font-semibold text-white">Run {activeTab === 'delta_energy' ? 'delta-energy' : 'commitment/frustration'} analysis</h2>
                <p className="text-xs text-gray-400 mt-1">Select the model pair and trajectories for the active tab.</p>
              </div>
              <button type="button" onClick={() => setRunPanelOpen(false)} className="text-sm text-gray-400 hover:text-gray-100">
                Close
              </button>
            </div>
            <div className="grid grid-cols-1 gap-3 pt-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Cluster</label>
                <select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100">
                  {clusterOptions.map((cluster) => (
                    <option key={cluster.cluster_id} value={cluster.cluster_id}>{cluster.name || cluster.cluster_id}</option>
                  ))}
                </select>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Model A</label>
                  <select value={modelAId} onChange={(e) => setModelAId(e.target.value)} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100">
                    {deltaModels.map((model) => <option key={`modal-a:${model.model_id}`} value={model.model_id}>{model.name || model.model_id}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Model B</label>
                  <select value={modelBId} onChange={(e) => setModelBId(e.target.value)} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100">
                    {deltaModels.map((model) => <option key={`modal-b:${model.model_id}`} value={model.model_id}>{model.name || model.model_id}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Samples to analyze</label>
                <select multiple value={selectedSampleIds} onChange={(e) => setSelectedSampleIds(Array.from(e.target.selectedOptions).map((o) => String(o.value)))} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 h-40">
                  {sampleEntries.map((sample) => (
                    <option key={`modal-s:${sample.sample_id}`} value={sample.sample_id}>
                      {sample.name || sample.sample_id}{sample.type ? ` (${sample.type})` : ''}
                    </option>
                  ))}
                </select>
              </div>
              {activeTab === 'delta_energy' ? (
                <div className="rounded-md border border-gray-700 bg-gray-950/50 p-3 space-y-2">
                  <h3 className="text-sm font-semibold text-gray-100">Frame selection</h3>
                  {selectedSampleIds.map((sid) => {
                    const sample = sampleEntries.find((s) => String(s.sample_id) === String(sid));
                    const total = Number(sampleFrameCounts[sid] || 0);
                    const mode = sampleFrameModes[sid] || 'all';
                    return (
                      <div key={`frame:${sid}`} className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_120px_140px] gap-2 items-end">
                        <div className="text-xs text-gray-300 truncate">{sample?.name || sid}<span className="text-gray-500"> · {total || '?'} frames</span></div>
                        <select value={mode} onChange={(e) => setSampleFrameModes((prev) => ({ ...prev, [sid]: e.target.value }))} className="bg-gray-950 border border-gray-800 rounded-md px-2 py-1.5 text-xs text-gray-100">
                          <option value="all">all frames</option>
                          <option value="random">random subset</option>
                        </select>
                        <input type="number" min={1} max={total || undefined} disabled={mode !== 'random'} value={sampleFrameLimits[sid] || ''} placeholder={total ? `< ${total}` : 'frames'} onChange={(e) => setSampleFrameLimits((prev) => ({ ...prev, [sid]: Number(e.target.value) }))} className="bg-gray-950 border border-gray-800 rounded-md px-2 py-1.5 text-xs text-gray-100 disabled:opacity-50" />
                      </div>
                    );
                  })}
                </div>
              ) : null}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">MD labels</label>
                  <select value={mdLabelMode} onChange={(e) => setMdLabelMode(e.target.value)} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100">
                    <option value="assigned">assigned</option>
                    <option value="halo">halo</option>
                  </select>
                </div>
                {activeTab !== 'delta_energy' ? (
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Top edges stored</label>
                    <input type="number" min={1} step={1} value={topKEdges} onChange={(e) => setTopKEdges(Number(e.target.value))} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100" />
                  </div>
                ) : (
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Energy bins</label>
                    <input type="number" min={5} step={1} value={deltaEnergyBins} onChange={(e) => setDeltaEnergyBins(Number(e.target.value))} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100" />
                  </div>
                )}
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Workers</label>
                  <input type="number" min={0} step={1} value={workers} onChange={(e) => setWorkers(Number(e.target.value))} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100" />
                </div>
              </div>
              {activeTab === 'delta_energy' ? (
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Random seed</label>
                  <input type="number" min={0} step={1} value={deltaEnergySeed} onChange={(e) => setDeltaEnergySeed(Number(e.target.value))} className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100" />
                </div>
              ) : null}
              <label className="flex items-center gap-2 text-sm text-gray-200">
                <input type="checkbox" checked={keepInvalid} onChange={(e) => setKeepInvalid(e.target.checked)} className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950" />
                Keep invalid frames
              </label>
              <button type="button" onClick={handleRun} disabled={!selectedClusterId || !modelAId || !modelBId || !selectedSampleIds.length} className="inline-flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-cyan-500 text-black font-semibold disabled:opacity-50">
                <Play className="h-4 w-4" />
                Run analysis
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-gray-500">Pairwise Potts model analyses</p>
          <h1 className="text-2xl font-semibold text-white mt-2">Model-pair analysis</h1>
          <p className="text-sm text-gray-400 mt-2 max-w-4xl">
            For a fixed pair of Potts models, inspect delta-energy distributions, local commitment,
            and trajectory-normalized frustration summaries.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)}
            className="px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800"
          >
            Back to system
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_commitment_3d${buildSamplingSuffix(selectedClusterId)}`)}
            className="px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800"
          >
            Open 3D viewer
          </button>
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800"
          >
            <CircleHelp className="h-4 w-4" />
            Help
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-6">
        <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white">Analysis browser</h2>
            <button
              type="button"
              onClick={() => setRunPanelOpen(true)}
              className="inline-flex items-center gap-2 px-2 py-1 rounded-md bg-cyan-500 text-xs font-semibold text-black"
            >
              <Play className="h-3.5 w-3.5" />
              Run analysis
            </button>
          </div>
          <div className="flex items-center justify-end">
            <button
              type="button"
              onClick={loadAnalyses}
              className="inline-flex items-center gap-2 px-2 py-1 rounded-md border border-gray-700 text-xs text-gray-200 hover:bg-gray-700/40"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Refresh
            </button>
          </div>

          <div className="hidden">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Cluster</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {clusterOptions.map((cluster) => (
                  <option key={cluster.cluster_id} value={cluster.cluster_id}>
                    {cluster.name || cluster.cluster_id}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Model A</label>
                <select
                  value={modelAId}
                  onChange={(e) => setModelAId(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  {deltaModels.map((model) => (
                    <option key={`a:${model.model_id}`} value={model.model_id}>
                      {model.name || model.model_id}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Model B</label>
                <select
                  value={modelBId}
                  onChange={(e) => setModelBId(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  {deltaModels.map((model) => (
                    <option key={`b:${model.model_id}`} value={model.model_id}>
                      {model.name || model.model_id}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Samples to analyze</label>
              <select
                multiple
                value={selectedSampleIds}
                onChange={(e) => setSelectedSampleIds(Array.from(e.target.selectedOptions).map((o) => String(o.value)))}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 h-40"
              >
                {sampleEntries.map((sample) => (
                  <option key={sample.sample_id} value={sample.sample_id}>
                    {sample.name || sample.sample_id}
                    {sample.type ? ` (${sample.type})` : ''}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">MD labels</label>
                <select
                  value={mdLabelMode}
                  onChange={(e) => setMdLabelMode(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  <option value="assigned">assigned</option>
                  <option value="halo">halo</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Top edges stored</label>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={topKEdges}
                  onChange={(e) => setTopKEdges(Number(e.target.value))}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Workers</label>
              <input
                type="number"
                min={0}
                step={1}
                value={workers}
                onChange={(e) => setWorkers(Number(e.target.value))}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              />
              <p className="text-[11px] text-gray-500 mt-1"><code>0</code> uses an automatic sample-level worker count.</p>
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-200">
              <input
                type="checkbox"
                checked={keepInvalid}
                onChange={(e) => setKeepInvalid(e.target.checked)}
                className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
              />
              Keep invalid frames
            </label>

            <button
              type="button"
              onClick={handleRun}
              disabled={!selectedClusterId || !modelAId || !modelBId || !selectedSampleIds.length}
              className="inline-flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-cyan-500 text-black font-semibold disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              Run analysis
            </button>
          </div>

          {jobError ? <ErrorMessage message={jobError} /> : null}
          {jobStatus ? (
            <div className="rounded-md border border-gray-700 bg-gray-900/60 p-3 text-sm text-gray-200 space-y-2">
              <div className="flex items-center justify-between">
                <span>{jobStatusLabel}</span>
                <span>{jobProgress}%</span>
              </div>
              <div className="w-full bg-gray-900 rounded-full h-2 overflow-hidden">
                <div className="h-full bg-cyan-400" style={{ width: `${clamp01(jobProgress / 100) * 100}%` }} />
              </div>
              {jobStatus?.meta?.status ? <p className="text-xs text-gray-400">{jobStatus.meta.status}</p> : null}
            </div>
          ) : null}

          <div className="border-t border-gray-700 pt-4 space-y-2">
            <h3 className="text-sm font-semibold text-white">Available analyses</h3>
            {analysesError ? <ErrorMessage message={analysesError} /> : null}
            <div className="space-y-2 max-h-72 overflow-auto pr-1">
              {matchingAnalyses.map((analysis) => (
                <button
                  key={analysis.analysis_id}
                  type="button"
                  onClick={() => setSelectedAnalysisId(String(analysis.analysis_id))}
                  className={`w-full text-left rounded-md border px-3 py-2 ${
                    String(selectedAnalysisMeta?.analysis_id) === String(analysis.analysis_id)
                      ? 'border-cyan-500 bg-cyan-500/10'
                      : 'border-gray-700 bg-gray-900/40 hover:bg-gray-900/70'
                  }`}
                >
                  <div className="text-sm text-gray-100">{analysis.model_a_name || analysis.model_a_id} vs {analysis.model_b_name || analysis.model_b_id}</div>
                  <div className="text-xs text-gray-400 mt-1">
                    {analysis.analysis_type || activeAnalysisType} · samples: {analysis?.summary?.n_samples ?? 0}
                    {analysis?.summary?.n_selected_edges || analysis?.top_k_edges ? ` · edges: ${analysis?.summary?.n_selected_edges ?? analysis?.top_k_edges}` : ''}
                  </div>
                </button>
              ))}
              {!matchingAnalyses.length ? <p className="text-xs text-gray-500">No {activeAnalysisType.replace('_', ' ')} analyses yet.</p> : null}
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <div className="flex flex-wrap gap-2 rounded-lg border border-gray-700 bg-gray-800 p-2">
            {[
              ['delta_energy', 'Delta energy distributions'],
              ['commitment', 'Commitment'],
              ['frustration', 'Frustration'],
            ].map(([tab, label]) => (
              <button
                key={tab}
                type="button"
                onClick={() => setActiveTab(tab)}
                className={`rounded-md px-3 py-2 text-sm ${
                  activeTab === tab ? 'bg-cyan-500 text-black font-semibold' : 'bg-gray-900 text-gray-200 hover:bg-gray-700'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {clusterInfoError ? <ErrorMessage message={clusterInfoError} /> : null}
          {analysisDataError ? <ErrorMessage message={analysisDataError} /> : null}
          {analysisDataLoading ? <Loader message="Loading model-pair analysis..." /> : null}
          {!analysisDataLoading && !selectedAnalysisMeta ? (
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 text-sm text-gray-300">
              Select an existing analysis from the list, or run a new analysis for this tab.
            </div>
          ) : null}
          {!analysisDataLoading && analysisData ? (
            <>
              {activeTab !== 'delta_energy' ? (
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
                <div className="flex flex-wrap items-center gap-4">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Displayed sample</label>
                    <select
                      value={selectedSampleId}
                      onChange={(e) => setSelectedSampleId(e.target.value)}
                      className="bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                    >
                      {sampleIds.map((sid, idx) => (
                        <option key={sid} value={sid}>
                          {sampleLabels[idx] || sid}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Commitment display</label>
                    <select
                      value={commitmentCentered ? 'centered' : 'prob'}
                      onChange={(e) => setCommitmentCentered(e.target.value === 'centered')}
                      className="bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                    >
                      <option value="centered">Centered (q - 0.5)</option>
                      <option value="prob">Probability q</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Frustration channel</label>
                    <select
                      value={frustrationChannel}
                      onChange={(e) => setFrustrationChannel(e.target.value)}
                      className="bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                    >
                      <option value="sym">Symmetric</option>
                      <option value="pol">Polarity</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Frustration display</label>
                    <select
                      value={frustrationDisplayMode}
                      onChange={(e) => setFrustrationDisplayMode(e.target.value)}
                      className="bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                    >
                      <option value="raw">Raw normalized</option>
                      <option value="centered">Centered vs reference MD</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Residues shown</label>
                    <input
                      type="number"
                      min={10}
                      step={5}
                      value={residueLimit}
                      onChange={(e) => setResidueLimit(Number(e.target.value))}
                      className="bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 w-24"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Edges shown</label>
                    <input
                      type="number"
                      min={10}
                      step={5}
                      value={edgeLimit}
                      onChange={(e) => setEdgeLimit(Number(e.target.value))}
                      className="bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 w-24"
                    />
                  </div>
                </div>

                {frustrationDisplayMode === 'centered' ? (
                  <div className="rounded-md border border-gray-700 bg-gray-900/50 p-3 space-y-2">
                    <label className="block text-xs text-gray-400">Reference ensemble(s)</label>
                    <select
                      multiple
                      value={referenceSampleIds}
                      onChange={(e) => setReferenceSampleIds(Array.from(e.target.selectedOptions).map((o) => String(o.value)))}
                      className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 h-24"
                    >
                      {sampleIds.map((sid, idx) => (
                        <option key={`fr-ref:${sid}`} value={sid}>
                          {sampleTypes[idx] ? `${sampleLabels[idx] || sid} (${sampleTypes[idx]})` : sampleLabels[idx] || sid}
                        </option>
                      ))}
                    </select>
                    <p className="text-[11px] text-gray-500">
                      The selected reference trajectories define the baseline that is subtracted residue-by-residue and edge-by-edge.
                    </p>
                  </div>
                ) : null}

                <div className="rounded-md border border-gray-700 bg-gray-900/50 p-3 space-y-2">
                  <label className="flex items-center gap-2 text-sm text-gray-200">
                    <input
                      type="checkbox"
                      checked={edgeBlendEnabled}
                      onChange={(e) => setEdgeBlendEnabled(e.target.checked)}
                      className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                    />
                    Edge-weighted residue coloring
                  </label>
                  <div className="flex items-center gap-3">
                    <label className="text-xs text-gray-400">Blend weight</label>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={edgeBlendStrength}
                      onChange={(e) => setEdgeBlendStrength(Number(e.target.value))}
                      disabled={!edgeBlendEnabled}
                      className="flex-1"
                    />
                    <span className="text-xs text-gray-300 w-12 text-right">{edgeBlendStrength.toFixed(2)}</span>
                  </div>
                  <p className="text-[11px] text-gray-500">
                    When enabled, residue commitment/frustration is blended with the mean value of incident top edges,
                    weighted by <span className="font-mono">|ΔJ|</span>. Default stored weight is 0.75, but the blend
                    is intentionally off by default on webserver.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                  <div className="rounded-md border border-gray-700 bg-gray-900/40 p-3">
                    <div className="text-xs uppercase tracking-[0.15em] text-gray-500">Analysis</div>
                    <div className="text-gray-100 mt-2">{selectedAnalysisMeta?.analysis_id}</div>
                    <div className="text-xs text-gray-400 mt-1">samples: {analysisSummary?.n_samples ?? sampleIds.length}</div>
                  </div>
                  <div className="rounded-md border border-gray-700 bg-gray-900/40 p-3">
                    <div className="text-xs uppercase tracking-[0.15em] text-gray-500">Displayed sample</div>
                    <div className="text-gray-100 mt-2">{selectedSampleName || 'n/a'}</div>
                    <div className="text-xs text-gray-400 mt-1">cluster: {selectedCluster?.name || selectedClusterId}</div>
                  </div>
                  <div className="rounded-md border border-gray-700 bg-gray-900/40 p-3">
                    <div className="text-xs uppercase tracking-[0.15em] text-gray-500">Stored scope</div>
                    <div className="text-gray-100 mt-2">{analysisSummary?.n_residues ?? residueLabels.length} residues · {analysisSummary?.n_selected_edges ?? topEdgeIndices.length} edges</div>
                    <div className="text-xs text-gray-400 mt-1">label mode: {selectedAnalysisMeta?.md_label_mode || mdLabelMode}</div>
                  </div>
                </div>
                </div>
              ) : null}

              {activeTab === 'delta_energy' && !!deltaEnergySeries.length ? (
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 overflow-hidden">
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                    <div>
                      <h2 className="text-sm font-semibold text-gray-100">Delta energy distributions</h2>
                      <p className="text-xs text-gray-400">
                        Global endpoint score per frame: ΔE = E_model_A - E_model_B. Negative values favor model A.
                      </p>
                    </div>
                    <select
                      value={deltaEnergyGraphMode}
                      onChange={(e) => setDeltaEnergyGraphMode(e.target.value)}
                      className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5 text-xs text-gray-100"
                    >
                      <option value="histogram">histograms + fitted curves</option>
                      <option value="curves">fitted curves only</option>
                    </select>
                    <EnergySeriesSelectorButton
                      series={deltaEnergySeries}
                      selectedIds={selectedDeltaEnergySeriesIds}
                      onChange={setSelectedDeltaEnergySeriesIds}
                      dark
                    />
                  </div>
                  {deltaEnergyPlot ? (
                    <EnergyDistributionPlot plot={deltaEnergyPlot} height={340} />
                  ) : (
                    <p className="py-16 text-center text-sm text-gray-400">No selected trajectories.</p>
                  )}
                </div>
              ) : null}

              {activeTab === 'commitment' ? (
                <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4">
                  <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 overflow-hidden">
                    <Plot data={residueCommitmentPlot.data} layout={residueCommitmentPlot.layout} config={residueCommitmentPlot.config} style={{ width: '100%' }} />
                  </div>
                  <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 overflow-hidden">
                    <Plot data={edgeCommitmentPlot.data} layout={edgeCommitmentPlot.layout} config={edgeCommitmentPlot.config} style={{ width: '100%' }} />
                  </div>
                </div>
              ) : null}
              {activeTab === 'frustration' ? (
                <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4">
                  <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 overflow-hidden">
                    <Plot data={residueFrustrationPlot.data} layout={residueFrustrationPlot.layout} config={residueFrustrationPlot.config} style={{ width: '100%' }} />
                  </div>
                  <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 overflow-hidden">
                    <Plot data={edgeFrustrationPlot.data} layout={edgeFrustrationPlot.layout} config={edgeFrustrationPlot.config} style={{ width: '100%' }} />
                  </div>
                </div>
              ) : null}
            </>
          ) : null}
        </section>
      </div>
    </div>
  );
}
