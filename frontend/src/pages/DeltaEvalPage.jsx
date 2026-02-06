import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, Play, RefreshCw } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchPottsClusterInfo, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitDeltaEvalJob, submitDeltaTransitionJob } from '../api/jobs';

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

  const [transitionAnalyses, setTransitionAnalyses] = useState([]);
  const [transitionAnalysesError, setTransitionAnalysesError] = useState(null);
  const [transitionDataCache, setTransitionDataCache] = useState({});

  const [mdSampleId, setMdSampleId] = useState('');
  const [activeMdSampleId, setActiveMdSampleId] = useState('');
  const [inactiveMdSampleId, setInactiveMdSampleId] = useState('');
  const [pasMdSampleId, setPasMdSampleId] = useState('');
  const [modelAId, setModelAId] = useState('');
  const [modelBId, setModelBId] = useState('');

  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);

  const [transitionJob, setTransitionJob] = useState(null);
  const [transitionJobStatus, setTransitionJobStatus] = useState(null);
  const [transitionJobError, setTransitionJobError] = useState(null);

  const [bandFraction, setBandFraction] = useState(0.1);
  const [topKResidues, setTopKResidues] = useState(20);
  const [topKEdges, setTopKEdges] = useState(30);
  const [helpOpen, setHelpOpen] = useState(false);
  const [helpTopic, setHelpTopic] = useState('interpretation');

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
    if (!mdSamples.length) {
      setActiveMdSampleId('');
      setInactiveMdSampleId('');
      setPasMdSampleId('');
      return;
    }

    const has = (id) => id && mdSamples.some((s) => s.sample_id === id);
    const pickDistinct = (fallbackIndex, exclude) => {
      if (mdSamples[fallbackIndex] && !exclude.includes(mdSamples[fallbackIndex].sample_id)) return mdSamples[fallbackIndex].sample_id;
      const found = mdSamples.find((s) => !exclude.includes(s.sample_id));
      return found ? found.sample_id : mdSamples[0].sample_id;
    };

    let e1 = has(activeMdSampleId) ? activeMdSampleId : pickDistinct(0, []);
    let e2 = has(inactiveMdSampleId) ? inactiveMdSampleId : pickDistinct(1, [e1]);
    if (e2 === e1) e2 = pickDistinct(1, [e1]);
    let e3 = has(pasMdSampleId) ? pasMdSampleId : pickDistinct(2, [e1, e2]);
    if (e3 === e1 || e3 === e2) e3 = pickDistinct(2, [e1, e2]);

    if (e1 !== activeMdSampleId) setActiveMdSampleId(e1);
    if (e2 !== inactiveMdSampleId) setInactiveMdSampleId(e2);
    if (e3 !== pasMdSampleId) setPasMdSampleId(e3);
  }, [mdSamples, activeMdSampleId, inactiveMdSampleId, pasMdSampleId]);

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

  const loadTransitionAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setTransitionAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_transition' });
      setTransitionAnalyses(Array.isArray(data?.analyses) ? data.analyses : []);
    } catch (err) {
      setTransitionAnalysesError(err.message || 'Failed to load analyses.');
      setTransitionAnalyses([]);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    setAnalysisDataCache({});
    setTransitionDataCache({});
    loadClusterInfo();
    loadAnalyses();
    loadTransitionAnalyses();
  }, [selectedClusterId, loadClusterInfo, loadAnalyses, loadTransitionAnalyses]);

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

  const loadTransitionData = useCallback(
    async (analysisId) => {
      if (!analysisId) return null;
      const cacheKey = `delta_transition:${analysisId}`;
      if (transitionDataCache[cacheKey]) return transitionDataCache[cacheKey];
      const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'delta_transition', analysisId);
      setTransitionDataCache((prev) => ({ ...prev, [cacheKey]: payload }));
      return payload;
    },
    [transitionDataCache, projectId, systemId, selectedClusterId]
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

  const selectedTransitionMeta = useMemo(() => {
    if (!activeMdSampleId || !inactiveMdSampleId || !pasMdSampleId || !modelAId || !modelBId) return null;
    const wantBand = Number(bandFraction);
    const wantTopK = Number(topKResidues);
    const wantTopKEdges = Number(topKEdges);
    return (
      transitionAnalyses.find((a) => {
        const mode = (a.md_label_mode || 'assigned').toLowerCase();
        const band = typeof a.band_fraction === 'number' ? a.band_fraction : Number(a.band_fraction);
        const topK = typeof a.top_k_residues === 'number' ? a.top_k_residues : Number(a.top_k_residues);
        const topKE = typeof a.top_k_edges === 'number' ? a.top_k_edges : Number(a.top_k_edges);
        return (
          a.active_md_sample_id === activeMdSampleId &&
          a.inactive_md_sample_id === inactiveMdSampleId &&
          a.pas_md_sample_id === pasMdSampleId &&
          a.model_a_id === modelAId &&
          a.model_b_id === modelBId &&
          mode === mdLabelMode &&
          Boolean(a.drop_invalid) === Boolean(dropInvalid) &&
          (Number.isFinite(wantBand) ? Math.abs((band || 0) - wantBand) < 1e-6 : true) &&
          (Number.isFinite(wantTopK) ? topK === wantTopK : true) &&
          (Number.isFinite(wantTopKEdges) ? topKE === wantTopKEdges : true)
        );
      }) || null
    );
  }, [
    transitionAnalyses,
    activeMdSampleId,
    inactiveMdSampleId,
    pasMdSampleId,
    modelAId,
    modelBId,
    mdLabelMode,
    dropInvalid,
    bandFraction,
    topKResidues,
    topKEdges,
  ]);

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

  const [transitionData, setTransitionData] = useState(null);
  const [transitionDataError, setTransitionDataError] = useState(null);
  const [transitionDataLoading, setTransitionDataLoading] = useState(false);
  useEffect(() => {
    const run = async () => {
      setTransitionData(null);
      setTransitionDataError(null);
      if (!selectedTransitionMeta) return;
      setTransitionDataLoading(true);
      try {
        const payload = await loadTransitionData(selectedTransitionMeta.analysis_id);
        setTransitionData(payload);
      } catch (err) {
        setTransitionDataError(err.message || 'Failed to load data.');
      } finally {
        setTransitionDataLoading(false);
      }
    };
    run();
  }, [selectedTransitionMeta, loadTransitionData]);

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

  const handleRunTransition = useCallback(async () => {
    if (
      !selectedClusterId ||
      !activeMdSampleId ||
      !inactiveMdSampleId ||
      !pasMdSampleId ||
      !modelAId ||
      !modelBId ||
      modelAId === modelBId
    )
      return;
    setTransitionJobError(null);
    setTransitionJob(null);
    setTransitionJobStatus(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        active_md_sample_id: activeMdSampleId,
        inactive_md_sample_id: inactiveMdSampleId,
        pas_md_sample_id: pasMdSampleId,
        model_a_id: modelAId,
        model_b_id: modelBId,
        md_label_mode: mdLabelMode,
        keep_invalid: keepInvalid,
        band_fraction: Number(bandFraction),
        top_k_residues: Number(topKResidues),
        top_k_edges: Number(topKEdges),
      };
      const res = await submitDeltaTransitionJob(payload);
      setTransitionJob(res);
    } catch (err) {
      setTransitionJobError(err.message || 'Failed to submit delta transition job.');
    }
  }, [
    projectId,
    systemId,
    selectedClusterId,
    activeMdSampleId,
    inactiveMdSampleId,
    pasMdSampleId,
    modelAId,
    modelBId,
    mdLabelMode,
    keepInvalid,
    bandFraction,
    topKResidues,
    topKEdges,
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

  useEffect(() => {
    if (!transitionJob?.job_id) return;
    let cancelled = false;
    const terminal = new Set(['finished', 'failed', 'canceled']);
    const poll = async () => {
      try {
        const status = await fetchJobStatus(transitionJob.job_id);
        if (cancelled) return;
        setTransitionJobStatus(status);
        if (terminal.has(status?.status)) {
          clearInterval(timer);
          if (status?.status === 'finished') await loadTransitionAnalyses();
        }
      } catch (err) {
        if (!cancelled) setTransitionJobError(err.message || 'Failed to poll job.');
      }
    };
    const timer = setInterval(poll, 2000);
    poll();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [transitionJob, loadTransitionAnalyses]);

  const residueLabels = useMemo(() => {
    const keys = clusterInfo?.residue_keys || [];
    if (Array.isArray(keys) && keys.length) return keys;
    const n = clusterInfo?.n_residues || 0;
    return Array.from({ length: n }, (_, i) => String(i));
  }, [clusterInfo]);

  const mdSampleById = useMemo(() => {
    const out = {};
    mdSamples.forEach((s) => {
      out[s.sample_id] = s;
    });
    return out;
  }, [mdSamples]);

  const ensemble1Name = useMemo(() => mdSampleById[activeMdSampleId]?.name || activeMdSampleId || '', [mdSampleById, activeMdSampleId]);
  const ensemble2Name = useMemo(
    () => mdSampleById[inactiveMdSampleId]?.name || inactiveMdSampleId || '',
    [mdSampleById, inactiveMdSampleId]
  );
  const ensemble3Name = useMemo(() => mdSampleById[pasMdSampleId]?.name || pasMdSampleId || '', [mdSampleById, pasMdSampleId]);

  const deltaEnergy = useMemo(() => (Array.isArray(data?.data?.delta_energy) ? data.data.delta_energy : []), [data]);
  const deltaEnergyPottsA = useMemo(
    () => (Array.isArray(data?.data?.delta_energy_potts_a) ? data.data.delta_energy_potts_a : []),
    [data]
  );
  const deltaEnergyPottsB = useMemo(
    () => (Array.isArray(data?.data?.delta_energy_potts_b) ? data.data.delta_energy_potts_b : []),
    [data]
  );
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

  const deltaEnergySeries = useMemo(() => {
    const modelAName = data?.metadata?.model_a_name || 'Potts (A)';
    const modelBName = data?.metadata?.model_b_name || 'Potts (B)';
    const series = [
      { key: 'md', name: mdSampleById[mdSampleId]?.name || 'MD', x: deltaEnergy, color: pickColor(0), opacity: 0.75 },
      { key: 'pottsA', name: modelAName, x: deltaEnergyPottsA, color: pickColor(1), opacity: 0.55 },
      { key: 'pottsB', name: modelBName, x: deltaEnergyPottsB, color: pickColor(2), opacity: 0.55 },
    ].filter((s) => Array.isArray(s.x) && s.x.length);
    if (!series.length) return null;
    let min = Infinity;
    let max = -Infinity;
    series.forEach((s) => {
      s.x.forEach((v) => {
        if (!Number.isFinite(v)) return;
        if (v < min) min = v;
        if (v > max) max = v;
      });
    });
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { series, xbins: null };
    if (min === max) {
      min -= 1;
      max += 1;
    }
    const nbins = 60;
    const size = (max - min) / nbins;
    const xbins = size > 0 ? { start: min, end: max, size } : null;
    return { series, xbins };
  }, [deltaEnergy, deltaEnergyPottsA, deltaEnergyPottsB, data, mdSampleById, mdSampleId]);

  const tsZActive = useMemo(
    () => (Array.isArray(transitionData?.data?.z_active) ? transitionData.data.z_active : []),
    [transitionData]
  );
  const tsZInactive = useMemo(
    () => (Array.isArray(transitionData?.data?.z_inactive) ? transitionData.data.z_inactive : []),
    [transitionData]
  );
  const tsZPas = useMemo(
    () => (Array.isArray(transitionData?.data?.z_pas) ? transitionData.data.z_pas : []),
    [transitionData]
  );
  const tsEdges = useMemo(
    () => (Array.isArray(transitionData?.data?.edges) ? transitionData.data.edges : []),
    [transitionData]
  );

  const tsTau = useMemo(() => {
    const raw = transitionData?.data?.tau;
    if (Array.isArray(raw)) return raw.length ? Number(raw[0]) : null;
    if (typeof raw === 'number') return raw;
    const v = Number(raw);
    return Number.isFinite(v) ? v : null;
  }, [transitionData]);
  const tsPTrain = useMemo(() => {
    const raw = transitionData?.data?.p_train;
    if (Array.isArray(raw)) return raw.length ? Number(raw[0]) : null;
    if (typeof raw === 'number') return raw;
    const v = Number(raw);
    return Number.isFinite(v) ? v : null;
  }, [transitionData]);
  const tsPPas = useMemo(() => {
    const raw = transitionData?.data?.p_pas;
    if (Array.isArray(raw)) return raw.length ? Number(raw[0]) : null;
    if (typeof raw === 'number') return raw;
    const v = Number(raw);
    return Number.isFinite(v) ? v : null;
  }, [transitionData]);
  const tsEnrichment = useMemo(() => {
    const raw = transitionData?.data?.enrichment;
    if (Array.isArray(raw)) return raw.length ? Number(raw[0]) : null;
    if (typeof raw === 'number') return raw;
    const v = Number(raw);
    return Number.isFinite(v) ? v : null;
  }, [transitionData]);

  const tsTopResidueIndices = useMemo(
    () => (Array.isArray(transitionData?.data?.top_residue_indices) ? transitionData.data.top_residue_indices : []),
    [transitionData]
  );
  const tsQResidue = useMemo(() => (Array.isArray(transitionData?.data?.q_residue) ? transitionData.data.q_residue : []), [
    transitionData,
  ]);
  const tsDResidue = useMemo(() => (Array.isArray(transitionData?.data?.D_residue) ? transitionData.data.D_residue : []), [
    transitionData,
  ]);
  const tsDEdge = useMemo(() => (Array.isArray(transitionData?.data?.D_edge) ? transitionData.data.D_edge : []), [transitionData]);
  const tsTopEdgeIndices = useMemo(
    () => (Array.isArray(transitionData?.data?.top_edge_indices) ? transitionData.data.top_edge_indices : []),
    [transitionData]
  );
  const tsQEdge = useMemo(() => (Array.isArray(transitionData?.data?.q_edge) ? transitionData.data.q_edge : []), [transitionData]);
  const tsTopResidueLabels = useMemo(() => {
    if (!Array.isArray(tsTopResidueIndices) || !tsTopResidueIndices.length) return [];
    return tsTopResidueIndices.map((raw) => {
      const idx = Number(raw);
      return residueLabels[idx] ?? String(raw);
    });
  }, [tsTopResidueIndices, residueLabels]);

  const tsTopDValues = useMemo(() => {
    if (!Array.isArray(tsTopResidueIndices) || !tsTopResidueIndices.length) return [];
    if (!Array.isArray(tsDResidue) || !tsDResidue.length) return [];
    return tsTopResidueIndices.map((raw) => {
      const idx = Number(raw);
      const v = tsDResidue[idx];
      return Number.isFinite(v) ? v : 0;
    });
  }, [tsTopResidueIndices, tsDResidue]);

  const tsTopEdgeLabels = useMemo(() => {
    if (!Array.isArray(tsTopEdgeIndices) || !tsTopEdgeIndices.length) return [];
    if (!Array.isArray(tsEdges) || !tsEdges.length) return [];
    return tsTopEdgeIndices.map((raw) => {
      const eidx = Number(raw);
      const edge = tsEdges[eidx];
      if (!Array.isArray(edge) || edge.length < 2) return String(raw);
      const r = Number(edge[0]);
      const s = Number(edge[1]);
      const a = residueLabels[r] ?? `res_${r}`;
      const b = residueLabels[s] ?? `res_${s}`;
      return `${a}–${b}`;
    });
  }, [tsTopEdgeIndices, tsEdges, residueLabels]);

  const tsTopDEdgeValues = useMemo(() => {
    if (!Array.isArray(tsTopEdgeIndices) || !tsTopEdgeIndices.length) return [];
    if (!Array.isArray(tsDEdge) || !tsDEdge.length) return [];
    return tsTopEdgeIndices.map((raw) => {
      const eidx = Number(raw);
      const v = tsDEdge[eidx];
      return Number.isFinite(v) ? v : 0;
    });
  }, [tsTopEdgeIndices, tsDEdge]);

  const tsHeatmapRows = useMemo(() => {
    return [ensemble1Name || 'Ensemble 1', ensemble2Name || 'Ensemble 2', ensemble3Name || 'Ensemble 3', 'TS-band'];
  }, [ensemble1Name, ensemble2Name, ensemble3Name]);

  const tsZSeries = useMemo(() => {
    const n1 = ensemble1Name || 'Ensemble 1';
    const n2 = ensemble2Name || 'Ensemble 2';
    const n3 = ensemble3Name || 'Ensemble 3';
    const series = [
      { key: 'e1', name: n1, x: tsZActive, color: pickColor(0), opacity: 0.55 },
      { key: 'e2', name: n2, x: tsZInactive, color: pickColor(3), opacity: 0.55 },
      { key: 'e3', name: n3, x: tsZPas, color: pickColor(1), opacity: 0.55 },
    ].filter((s) => Array.isArray(s.x) && s.x.length);
    if (!series.length) return null;
    let min = Infinity;
    let max = -Infinity;
    series.forEach((s) => {
      s.x.forEach((v) => {
        if (!Number.isFinite(v)) return;
        if (v < min) min = v;
        if (v > max) max = v;
      });
    });
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { series, xbins: null };
    if (min === max) {
      min -= 1;
      max += 1;
    }
    const nbins = 60;
    const size = (max - min) / nbins;
    const xbins = size > 0 ? { start: min, end: max, size } : null;
    return { series, xbins };
  }, [tsZActive, tsZInactive, tsZPas]);

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
            onClick={() => {
              setHelpTopic('interpretation');
              setHelpOpen(true);
            }}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
          >
            <CircleHelp className="h-4 w-4" />
            Help
          </button>
          <button
            type="button"
            onClick={() => {
              setHelpTopic('analyses');
              setHelpOpen(true);
            }}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Analyses
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

      <HelpDrawer
        open={helpOpen}
        title={helpTopic === 'analyses' ? 'Delta Potts Evaluation: Analyses & Goals' : 'Delta Potts Evaluation: How To Interpret Results'}
        docPath={helpTopic === 'analyses' ? '/docs/delta_eval_analyses.md' : '/docs/delta_eval_interpretation.md'}
        onClose={() => setHelpOpen(false)}
      />

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
                  await loadTransitionAnalyses();
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

          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
            <div>
              <p className="text-xs font-semibold text-gray-200">Transition-like analysis</p>
              <p className="text-[11px] text-gray-500">
                Computes a ΔE-derived reaction coordinate and a TS-band enrichment + per-residue commitment heatmap (
                <code>validation_ladder3.MD</code>).
              </p>
            </div>

            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Ensemble 1 MD sample</label>
              <select
                value={activeMdSampleId}
                onChange={(e) => setActiveMdSampleId(e.target.value)}
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
              <label className="block text-xs text-gray-400">Ensemble 2 MD sample</label>
              <select
                value={inactiveMdSampleId}
                onChange={(e) => setInactiveMdSampleId(e.target.value)}
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
              <label className="block text-xs text-gray-400">Ensemble 3 MD sample</label>
              <select
                value={pasMdSampleId}
                onChange={(e) => setPasMdSampleId(e.target.value)}
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

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400">Band fraction</label>
                <input
                  type="number"
                  min="0.01"
                  max="0.99"
                  step="0.01"
                  value={bandFraction}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setBandFraction(Number.isFinite(v) ? v : 0.1);
                  }}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400">Top residues</label>
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={topKResidues}
                  onChange={(e) => {
                    const v = parseInt(e.target.value, 10);
                    setTopKResidues(Number.isFinite(v) ? v : 20);
                  }}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs text-gray-400">Top edges</label>
              <input
                type="number"
                min="1"
                step="1"
                value={topKEdges}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  setTopKEdges(Number.isFinite(v) ? v : 30);
                }}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              />
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleRunTransition}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-cyan-700 hover:bg-cyan-600 text-white text-sm disabled:opacity-60"
                disabled={
                  !selectedClusterId ||
                  !activeMdSampleId ||
                  !inactiveMdSampleId ||
                  !pasMdSampleId ||
                  !modelAId ||
                  !modelBId ||
                  modelAId === modelBId
                }
              >
                <Play className="h-4 w-4" />
                Run TS analysis
              </button>
            </div>

            {transitionJob?.job_id && (
              <div className="text-[11px] text-gray-300">
                Job: <span className="text-gray-200">{transitionJob.job_id}</span>{' '}
                {transitionJobStatus?.meta?.status ? `· ${transitionJobStatus.meta.status}` : ''}
                {typeof transitionJobStatus?.meta?.progress === 'number' ? ` · ${transitionJobStatus.meta.progress}%` : ''}
              </div>
            )}
            {transitionJobError && <ErrorMessage message={transitionJobError} />}
            {transitionAnalysesError && <ErrorMessage message={transitionAnalysesError} />}
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
            {!!deltaEnergySeries?.series?.length && (
              <div className="rounded-md border border-gray-800 bg-white p-3">
                <Plot
                  data={deltaEnergySeries.series.map((s) => ({
                    x: s.x,
                    type: 'histogram',
                    opacity: s.opacity,
                    marker: { color: s.color },
                    nbinsx: 60,
                    xbins: deltaEnergySeries.xbins || undefined,
                    name: s.name,
                  }))}
                  layout={{
                    height: 260,
                    margin: { l: 40, r: 10, t: 10, b: 40 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    barmode: 'overlay',
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

          <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
            <div>
              <h2 className="text-sm font-semibold text-gray-200">Transition-like (TS-band) analysis</h2>
              <p className="text-[11px] text-gray-500">
                Reaction coordinate: <span className="font-mono">z = (ΔE − median(train)) / MAD(train)</span>. Band
                threshold τ chosen so that <span className="font-mono">P_train(|z| ≤ τ) ≈ band_fraction</span>. Enrichment
                is <span className="font-mono">log((p_3 + ε) / (p_train + ε))</span>.
              </p>
              <p className="text-[11px] text-gray-500 mt-1">
                Mapping: <span className="font-mono">Ensemble 1</span>={ensemble1Name || '—'} ·{' '}
                <span className="font-mono">Ensemble 2</span>={ensemble2Name || '—'} ·{' '}
                <span className="font-mono">Ensemble 3</span>={ensemble3Name || '—'}
              </p>
            </div>

            {!selectedTransitionMeta && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                No TS analysis found for this selection. Click <span className="font-semibold">Run TS analysis</span>.
              </div>
            )}

            {transitionDataError && <ErrorMessage message={transitionDataError} />}
            {transitionDataLoading && <p className="text-sm text-gray-400">Loading…</p>}

            {!!tsZSeries?.series?.length && (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div className="rounded-md border border-gray-800 bg-white p-3">
                  <Plot
                    data={tsZSeries.series.map((s) => ({
                      x: s.x,
                      type: 'histogram',
                      opacity: s.opacity,
                      marker: { color: s.color },
                      nbinsx: 60,
                      xbins: tsZSeries.xbins || undefined,
                      name: s.name,
                    }))}
                    layout={{
                      height: 260,
                      margin: { l: 40, r: 10, t: 10, b: 40 },
                      paper_bgcolor: '#ffffff',
                      plot_bgcolor: '#ffffff',
                      font: { color: '#111827' },
                      barmode: 'overlay',
                      xaxis: { title: 'z', color: '#111827' },
                      yaxis: { title: 'Count', color: '#111827' },
                      shapes:
                        Number.isFinite(tsTau) && tsTau !== null
                          ? [
                              {
                                type: 'line',
                                x0: -tsTau,
                                x1: -tsTau,
                                y0: 0,
                                y1: 1,
                                xref: 'x',
                                yref: 'paper',
                                line: { color: '#111827', width: 1, dash: 'dash' },
                              },
                              {
                                type: 'line',
                                x0: tsTau,
                                x1: tsTau,
                                y0: 0,
                                y1: 1,
                                xref: 'x',
                                yref: 'paper',
                                line: { color: '#111827', width: 1, dash: 'dash' },
                              },
                            ]
                          : [],
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    useResizeHandler
                    style={{ width: '100%', height: '260px' }}
                  />
                </div>

                <div className="rounded-md border border-gray-800 bg-white p-3">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-semibold text-gray-800">Enrichment in TS-band</p>
                    <p className="text-[11px] text-gray-600">
                      τ={Number.isFinite(tsTau) && tsTau !== null ? tsTau.toFixed(3) : '—'} · enrich=
                      {Number.isFinite(tsEnrichment) && tsEnrichment !== null ? tsEnrichment.toFixed(3) : '—'}
                    </p>
                  </div>
                  <Plot
                    data={[
                      {
                        x: [
                          `train (${ensemble1Name || 'ensemble 1'} + ${ensemble2Name || 'ensemble 2'})`,
                          `${ensemble3Name || 'ensemble 3'}`,
                        ],
                        y: [
                          Number.isFinite(tsPTrain) && tsPTrain !== null ? tsPTrain : 0,
                          Number.isFinite(tsPPas) && tsPPas !== null ? tsPPas : 0,
                        ],
                        type: 'bar',
                        marker: { color: ['#9ca3af', '#22d3ee'] },
                        hovertemplate: '%{x}: %{y:.4f}<extra></extra>',
                      },
                    ]}
                    layout={{
                      height: 220,
                      margin: { l: 40, r: 10, t: 10, b: 40 },
                      paper_bgcolor: '#ffffff',
                      plot_bgcolor: '#ffffff',
                      font: { color: '#111827' },
                      yaxis: { title: 'P(|z| ≤ τ)', color: '#111827', rangemode: 'tozero' },
                      xaxis: { color: '#111827' },
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    useResizeHandler
                    style={{ width: '100%', height: '220px' }}
                  />
                </div>
              </div>
            )}

            {!!tsQResidue?.length && !!tsTopResidueLabels?.length && !!tsHeatmapRows?.length && (
              <div className="rounded-md border border-gray-800 bg-white p-3">
                <p className="text-xs font-semibold text-gray-800 mb-2">
                  Top residues commitment heatmap <span className="font-normal text-gray-600">(q_i = Pr(δ_i &lt; 0))</span>
                </p>
                <Plot
                  data={[
                    {
                      z: tsQResidue,
                      x: tsTopResidueLabels,
                      y: tsHeatmapRows,
                      type: 'heatmap',
                      zmin: 0,
                      zmax: 1,
                      zmid: 0.5,
                      colorscale: 'RdBu',
                      reversescale: true,
                      hovertemplate: 'ensemble: %{y}<br>res: %{x}<br>q: %{z:.3f}<extra></extra>',
                    },
                  ]}
                  layout={{
                    height: 300,
                    margin: { l: 80, r: 10, t: 10, b: 90 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { tickfont: { size: 10 }, color: '#111827' },
                    yaxis: { color: '#111827' },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '300px' }}
                />
              </div>
            )}

            {!!tsTopDValues?.length && (
              <div className="rounded-md border border-gray-800 bg-white p-3">
                <p className="text-xs font-semibold text-gray-800 mb-2">
                  Discriminative power on training (fields-only): <span className="font-mono">D_i</span> for top residues
                </p>
                <Plot
                  data={[
                    {
                      x: tsTopResidueLabels,
                      y: tsTopDValues,
                      type: 'bar',
                      marker: {
                        color: tsTopDValues.map((v) => (v >= 0 ? '#f97316' : '#22d3ee')),
                      },
                      hovertemplate: 'res: %{x}<br>D: %{y:.4f}<extra></extra>',
                    },
                  ]}
                  layout={{
                    height: 260,
                    margin: { l: 40, r: 10, t: 10, b: 90 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { tickfont: { size: 10 }, color: '#111827' },
                    yaxis: {
                      title: `D_i (${ensemble1Name || 'ensemble 1'} - ${ensemble2Name || 'ensemble 2'})`,
                      color: '#111827',
                    },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '260px' }}
                />
              </div>
            )}

            {!!tsTopDEdgeValues?.length && (
              <div className="rounded-md border border-gray-800 bg-white p-3">
                <p className="text-xs font-semibold text-gray-800 mb-2">
                  Discriminative power on training (edges): <span className="font-mono">D_ij</span> for top edges
                </p>
                <Plot
                  data={[
                    {
                      x: tsTopEdgeLabels,
                      y: tsTopDEdgeValues,
                      type: 'bar',
                      marker: {
                        color: tsTopDEdgeValues.map((v) => (v >= 0 ? '#f97316' : '#22d3ee')),
                      },
                      hovertemplate: 'edge: %{x}<br>D: %{y:.4f}<extra></extra>',
                    },
                  ]}
                  layout={{
                    height: 260,
                    margin: { l: 40, r: 10, t: 10, b: 120 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { tickfont: { size: 10 }, tickangle: -45, color: '#111827' },
                    yaxis: {
                      title: `D_ij (${ensemble1Name || 'ensemble 1'} - ${ensemble2Name || 'ensemble 2'})`,
                      color: '#111827',
                    },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '260px' }}
                />
              </div>
            )}

            {!!tsQEdge?.length && !!tsTopEdgeLabels?.length && (
              <div className="rounded-md border border-gray-800 bg-white p-3">
                <p className="text-xs font-semibold text-gray-800 mb-2">
                  Top edges commitment heatmap <span className="font-normal text-gray-600">(q_ij = Pr(δ_ij &lt; 0))</span>
                </p>
                <Plot
                  data={[
                    {
                      z: tsQEdge,
                      x: tsTopEdgeLabels,
                      y: tsHeatmapRows,
                      type: 'heatmap',
                      zmin: 0,
                      zmax: 1,
                      zmid: 0.5,
                      colorscale: 'RdBu',
                      reversescale: true,
                      hovertemplate: 'ensemble: %{y}<br>edge: %{x}<br>q: %{z:.3f}<extra></extra>',
                    },
                  ]}
                  layout={{
                    height: 300,
                    margin: { l: 80, r: 10, t: 10, b: 140 },
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827' },
                    xaxis: { tickfont: { size: 10 }, tickangle: -45, color: '#111827' },
                    yaxis: { color: '#111827' },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: '100%', height: '300px' }}
                />
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
