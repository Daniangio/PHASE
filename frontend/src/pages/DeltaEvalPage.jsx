import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, Play, RefreshCw } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchPottsClusterInfo, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitDeltaCommitmentJob } from '../api/jobs';

const palette = ['#22d3ee', '#f97316', '#10b981', '#f43f5e', '#60a5fa', '#f59e0b', '#a855f7', '#84cc16'];
function pickColor(idx) {
  return palette[idx % palette.length];
}

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function sigmoid(x) {
  if (!Number.isFinite(x)) return 0.5;
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function clamp(x, lo, hi) {
  if (!Number.isFinite(x)) return lo;
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

function edgeLabel(edge, residueLabels) {
  if (!Array.isArray(edge) || edge.length < 2) return '';
  const r = Number(edge[0]);
  const s = Number(edge[1]);
  const a = residueLabels[r] ?? String(r);
  const b = residueLabels[s] ?? String(s);
  return `${a}–${b}`;
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

  const [modelAId, setModelAId] = useState('');
  const [modelBId, setModelBId] = useState('');

  const [commitmentMode, setCommitmentMode] = useState('prob'); // prob | centered | mu_sigmoid
  const [referenceSampleIds, setReferenceSampleIds] = useState([]); // used for centered mode
  const [edgeSmoothEnabled, setEdgeSmoothEnabled] = useState(false);
  const [edgeSmoothStrength, setEdgeSmoothStrength] = useState(0.75);
  const [hideSingleClusterResidues, setHideSingleClusterResidues] = useState(false);

  // Visualization-only filter: 0 means "show all residues". The analysis always stores q for all residues.
  const [topKResidues, setTopKResidues] = useState(0);
  // Edge commitment is stored only for the top-K edges by |ΔJ| to keep the stored analysis lightweight.
  // Set a generous default so other pages (e.g. 3D) can visualize hundreds of links without recomputing.
  const [topKEdges, setTopKEdges] = useState(2000);
  const [energyBins, setEnergyBins] = useState(80);

  const [selectedSampleIds, setSelectedSampleIds] = useState([]);
  const [plotSampleIds, setPlotSampleIds] = useState([]);

  const [analyses, setAnalyses] = useState([]);
  const [analysesError, setAnalysesError] = useState(null);

  const analysisDataCacheRef = useRef({});
  const analysisInFlightRef = useRef({});
  const [, setAnalysisDataCache] = useState({});

  const [data, setData] = useState(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState(null);

  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);

  const [helpOpen, setHelpOpen] = useState(false);
  const samplingSuffix = useMemo(() => {
    const params = new URLSearchParams();
    if (selectedClusterId) params.set('cluster_id', selectedClusterId);
    const s = params.toString();
    return s ? `?${s}` : '';
  }, [selectedClusterId]);

  useEffect(() => {
    const loadSystem = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        const sys = await fetchSystem(projectId, systemId);
        setSystem(sys);
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
    if (!deltaModels.length) {
      setModelAId('');
      setModelBId('');
      return;
    }
    const has = (id) => id && deltaModels.some((m) => m.model_id === id);
    let a = has(modelAId) ? modelAId : deltaModels[0].model_id;
    let b = has(modelBId) ? modelBId : '';
    if (!b || b === a) {
      const other = deltaModels.find((m) => m.model_id !== a);
      b = other?.model_id || a;
    }
    if (a !== modelAId) setModelAId(a);
    if (b !== modelBId) setModelBId(b);
  }, [deltaModels, modelAId, modelBId]);

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
      const res = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_commitment' });
      setAnalyses(Array.isArray(res?.analyses) ? res.analyses : []);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    analysisDataCacheRef.current = {};
    analysisInFlightRef.current = {};
    setAnalysisDataCache({});
    setData(null);
    setJob(null);
    setJobStatus(null);
    setJobError(null);
    loadClusterInfo();
    loadAnalyses();
  }, [selectedClusterId, loadClusterInfo, loadAnalyses]);

  useEffect(() => {
    if (!selectedClusterId) return;
    loadClusterInfo();
  }, [selectedClusterId, modelAId, loadClusterInfo]);

  // Default sample selection: select all MD samples (type=md_eval).
  useEffect(() => {
    if (!sampleEntries.length) {
      setSelectedSampleIds([]);
      setPlotSampleIds([]);
      return;
    }
    if (selectedSampleIds.length) return;
    const md = sampleEntries.filter((s) => s.type === 'md_eval').map((s) => s.sample_id);
    const initial = md.length ? md : sampleEntries.slice(0, 3).map((s) => s.sample_id);
    setSelectedSampleIds(initial);
    setPlotSampleIds(initial);
  }, [sampleEntries, selectedSampleIds]);

  const residueLabels = useMemo(() => {
    const keys = clusterInfo?.residue_keys || [];
    if (Array.isArray(keys) && keys.length) return keys;
    const n = clusterInfo?.n_residues || 0;
    return Array.from({ length: n }, (_, i) => `res_${i}`);
  }, [clusterInfo]);

  const selectedMeta = useMemo(() => {
    if (!modelAId || !modelBId) return null;
    const dropInvalid = !keepInvalid;
    const baseMatch = (a) =>
      a.model_a_id === modelAId &&
      a.model_b_id === modelBId &&
      (a.md_label_mode || 'assigned').toLowerCase() === mdLabelMode &&
      String(a.ranking_method || 'param_l2').toLowerCase() === 'param_l2';
    const exact = analyses.filter((a) => baseMatch(a) && Boolean(a.drop_invalid) === Boolean(dropInvalid));
    const fallback = exact.length ? exact : analyses.filter((a) => baseMatch(a));
    if (!fallback.length) return null;
    const score = (a) => {
      const nSamples = Number(a?.summary?.n_samples);
      const n = Number.isFinite(nSamples) ? nSamples : 0;
      const t = Date.parse(String(a?.updated_at || a?.created_at || ''));
      const ts = Number.isFinite(t) ? t : 0;
      return [n, ts];
    };
    const sorted = [...fallback].sort((x, y) => {
      const [nx, tx] = score(x);
      const [ny, ty] = score(y);
      if (ny !== nx) return ny - nx;
      return ty - tx;
    });
    return sorted[0] || null;
  }, [analyses, modelAId, modelBId, mdLabelMode, keepInvalid]);

  const loadAnalysisData = useCallback(
    async (analysisId) => {
      if (!analysisId) return null;
      const cacheKey = `delta_commitment:${analysisId}`;
      const cached = analysisDataCacheRef.current;
      if (Object.prototype.hasOwnProperty.call(cached, cacheKey)) return cached[cacheKey];
      const inflight = analysisInFlightRef.current;
      if (inflight[cacheKey]) return inflight[cacheKey];
      const p = fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'delta_commitment', analysisId)
        .then((payload) => {
          analysisDataCacheRef.current = { ...analysisDataCacheRef.current, [cacheKey]: payload };
          setAnalysisDataCache((prev) => ({ ...prev, [cacheKey]: payload }));
          delete analysisInFlightRef.current[cacheKey];
          return payload;
        })
        .catch((err) => {
          delete analysisInFlightRef.current[cacheKey];
          throw err;
        });
      inflight[cacheKey] = p;
      return p;
    },
    [projectId, systemId, selectedClusterId]
  );

  useEffect(() => {
    const run = async () => {
      setData(null);
      setDataError(null);
      if (!selectedMeta?.analysis_id) return;
      setDataLoading(true);
      try {
        const payload = await loadAnalysisData(selectedMeta.analysis_id);
        setData(payload);
      } catch (err) {
        setDataError(err.message || 'Failed to load analysis data.');
      } finally {
        setDataLoading(false);
      }
    };
    run();
  }, [selectedMeta, loadAnalysisData]);

  const handleRun = useCallback(async () => {
    if (!selectedClusterId || !modelAId || !modelBId || modelAId === modelBId) return;
    if (!selectedSampleIds.length) return;
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
        sample_ids: selectedSampleIds,
        md_label_mode: mdLabelMode,
        keep_invalid: keepInvalid,
        // Backend requires >=1; this parameter only affects the stored top_residue_indices convenience field.
        top_k_residues: Math.max(1, Number(topKResidues)),
        top_k_edges: Number(topKEdges),
        ranking_method: 'param_l2',
        energy_bins: Number(energyBins),
      };
      const res = await submitDeltaCommitmentJob(payload);
      setJob(res);
    } catch (err) {
      setJobError(err.message || 'Failed to submit delta commitment job.');
    }
  }, [
    projectId,
    systemId,
    selectedClusterId,
    modelAId,
    modelBId,
    selectedSampleIds,
    mdLabelMode,
    keepInvalid,
    topKResidues,
    topKEdges,
    energyBins,
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
            await loadAnalyses();
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
  }, [job, loadAnalyses]);

  const analysisSampleIds = useMemo(() => {
    const ids = data?.data?.sample_ids;
    return Array.isArray(ids) ? ids.map(String) : [];
  }, [data]);
  const analysisSampleLabels = useMemo(() => {
    const labels = data?.data?.sample_labels;
    if (Array.isArray(labels)) return labels.map(String);
    return analysisSampleIds;
  }, [data, analysisSampleIds]);
  const analysisSampleTypes = useMemo(() => {
    const types = data?.data?.sample_types;
    return Array.isArray(types) ? types.map(String) : [];
  }, [data]);

  const dhTable = useMemo(() => (Array.isArray(data?.data?.dh) ? data.data.dh : null), [data]);
  const pNode = useMemo(() => (Array.isArray(data?.data?.p_node) ? data.data.p_node : null), [data]);
  const kList = useMemo(() => (Array.isArray(data?.data?.K_list) ? data.data.K_list.map((v) => Number(v)) : null), [data]);
  const hasAltCommitmentData = useMemo(() => {
    const okDh = Boolean(dhTable && Array.isArray(dhTable[0]) && dhTable.length > 0);
    const okP = Boolean(pNode && Array.isArray(pNode[0]) && pNode.length > 0);
    return okDh && okP;
  }, [dhTable, pNode]);

  const analysisSampleIndexById = useMemo(() => {
    const m = new Map();
    analysisSampleIds.forEach((sid, i) => m.set(String(sid), i));
    return m;
  }, [analysisSampleIds]);

  // Default reference set for centered mode: all MD samples if present, else the first available sample.
  useEffect(() => {
    if (commitmentMode !== 'centered') return;
    if (referenceSampleIds.length) return;
    if (!analysisSampleIds.length) return;
    const md = analysisSampleIds.filter((sid, idx) => (analysisSampleTypes[idx] || '').toLowerCase().includes('md'));
    if (md.length) setReferenceSampleIds(md);
    else setReferenceSampleIds([analysisSampleIds[0]]);
  }, [commitmentMode, referenceSampleIds.length, analysisSampleIds, analysisSampleTypes]);

  // Default plotting selection: whatever is available in the analysis (or the selected set).
  useEffect(() => {
    if (!analysisSampleIds.length) return;
    if (plotSampleIds.length) return;
    setPlotSampleIds(analysisSampleIds.slice(0, Math.min(6, analysisSampleIds.length)));
  }, [analysisSampleIds, plotSampleIds]);

  const plotRows = useMemo(() => {
    const rows = [];
    for (const sid of plotSampleIds) {
      const idx = analysisSampleIndexById.get(String(sid));
      if (idx == null) continue;
      rows.push(idx);
    }
    return rows;
  }, [plotSampleIds, analysisSampleIndexById]);

  const dResidue = useMemo(() => (Array.isArray(data?.data?.D_residue) ? data.data.D_residue : []), [data]);
  const topResidueOrder = useMemo(() => {
    if (!Array.isArray(dResidue) || !dResidue.length) return [];
    const pairs = dResidue.map((v, i) => [Number(v), i]);
    pairs.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
    return pairs.map((p) => p[1]);
  }, [dResidue]);

  const topResidueIndices = useMemo(
    () => (Array.isArray(data?.data?.top_residue_indices) ? data.data.top_residue_indices : []),
    [data]
  );
  const topEdgeIndices = useMemo(() => (Array.isArray(data?.data?.top_edge_indices) ? data.data.top_edge_indices : []), [data]);
  const edges = useMemo(() => (Array.isArray(data?.data?.edges) ? data.data.edges : []), [data]);
  const dEdge = useMemo(() => (Array.isArray(data?.data?.D_edge) ? data.data.D_edge : null), [data]);
  const singleClusterByResidue = useMemo(() => {
    const n = residueLabels.length;
    const out = new Array(n).fill(false);
    const source =
      Array.isArray(kList) && kList.length
        ? kList
        : Array.isArray(clusterInfo?.cluster_counts)
        ? clusterInfo.cluster_counts
        : [];
    if (!Array.isArray(source) || !source.length) return out;
    const m = Math.min(n, source.length);
    for (let i = 0; i < m; i += 1) {
      const ki = Number(source[i]);
      if (Number.isFinite(ki) && ki <= 1) out[i] = true;
    }
    return out;
  }, [kList, clusterInfo, residueLabels.length]);

  const qResidueAll = useMemo(() => (Array.isArray(data?.data?.q_residue_all) ? data.data.q_residue_all : []), [data]);

  const centeredCalib = useMemo(() => {
    if (commitmentMode !== 'centered') return null;
    if (!dhTable || !pNode) return null;
    if (!referenceSampleIds.length || !analysisSampleIds.length) return null;

    const refIdxs = referenceSampleIds
      .map((sid) => analysisSampleIds.indexOf(String(sid)))
      .filter((i) => i != null && i >= 0);
    if (!refIdxs.length) return null;

    const N = dhTable.length;
    const Kmax = Array.isArray(dhTable[0]) ? dhTable[0].length : 0;
    if (N <= 0 || Kmax <= 0) return null;

    const eps = 1e-9;
    const thresholds = new Array(N);
    const alphas = new Array(N);
    for (let i = 0; i < N; i += 1) {
      const dhRow = (Array.isArray(dhTable[i]) ? dhTable[i] : []).map((v) => Number(v));
      const Ki = kList && Number.isFinite(kList[i]) ? Math.max(0, Math.min(Kmax, Math.floor(kList[i]))) : dhRow.length;
      if (dhRow.length < Ki || Ki <= 0) {
        thresholds[i] = 0;
        alphas[i] = 0.5;
        // eslint-disable-next-line no-continue
        continue;
      }
      const pRef = new Array(Ki).fill(0);
      for (const ridx of refIdxs) {
        const prow = pNode?.[ridx]?.[i];
        if (!Array.isArray(prow) || prow.length < Ki) continue;
        for (let a = 0; a < Ki; a += 1) pRef[a] += Number(prow[a]) || 0;
      }
      const inv = 1 / refIdxs.length;
      for (let a = 0; a < Ki; a += 1) pRef[a] *= inv;

      const order = Array.from({ length: Ki }, (_, a) => a);
      order.sort((a, b) => dhRow[a] - dhRow[b]);
      let cum = 0;
      let t = dhRow[order[order.length - 1]];
      for (const a of order) {
        cum += pRef[a];
        if (cum >= 0.5) {
          t = dhRow[a];
          break;
        }
      }
      thresholds[i] = t;

      let cBefore = 0;
      let cAt = 0;
      for (let a = 0; a < Ki; a += 1) {
        const dv = dhRow[a];
        const pv = pRef[a] || 0;
        if (dv < t - eps) cBefore += pv;
        else if (Math.abs(dv - t) <= eps) cAt += pv;
      }
      if (cAt > 0) {
        const alpha = (0.5 - cBefore) / cAt;
        alphas[i] = Math.max(0, Math.min(1, alpha));
      } else {
        alphas[i] = 0.5;
      }
    }
    return { thresholds, alphas, eps };
  }, [commitmentMode, dhTable, pNode, referenceSampleIds, analysisSampleIds, kList]);

  const showTopResiduesOnly = useMemo(
    () => Number(topKResidues) > 0 && Number(topKResidues) < residueLabels.length,
    [topKResidues, residueLabels.length]
  );
  const residueAxis = useMemo(() => {
    if (!showTopResiduesOnly) {
      return { indices: residueLabels.map((_, i) => i), labels: residueLabels };
    }
    const k = Math.max(1, Math.min(Number(topKResidues), residueLabels.length));
    const src = topResidueOrder.length ? topResidueOrder : (Array.isArray(topResidueIndices) ? topResidueIndices : []);
    const idxs = src.slice(0, k).map((x) => Number(x));
    const labels = idxs.map((i) => residueLabels[i] ?? String(i));
    return { indices: idxs, labels };
  }, [showTopResiduesOnly, residueLabels, topKResidues, topResidueOrder, topResidueIndices]);
  const residueAxisLabels = residueAxis.labels;
  const edgeAxisLabels = useMemo(() => {
    return (Array.isArray(topEdgeIndices) ? topEdgeIndices : []).map((raw) => {
      const eidx = Number(raw);
      const edge = edges[eidx];
      return edgeLabel(edge, residueLabels) || String(raw);
    });
  }, [topEdgeIndices, edges, residueLabels]);

  const qEdge = useMemo(() => (Array.isArray(data?.data?.q_edge) ? data.data.q_edge : []), [data]);
  const centeredEdgeRefIdxs = useMemo(() => {
    if (commitmentMode !== 'centered') return [];
    if (!referenceSampleIds.length || !analysisSampleIds.length) return [];
    return referenceSampleIds
      .map((sid) => analysisSampleIds.indexOf(String(sid)))
      .filter((i) => Number.isInteger(i) && i >= 0);
  }, [commitmentMode, referenceSampleIds, analysisSampleIds]);

  const qResidueHeatmap = useMemo(() => {
    if (!plotRows.length || !Array.isArray(qResidueAll) || !qResidueAll.length) return null;
    const N = residueLabels.length;
    const base = (row, i) => clamp01(Number(qResidueAll?.[row]?.[i]));
    const edgeRowForSmoothing = (rowIdx) => {
      if (!Array.isArray(qEdge) || !Array.isArray(qEdge[rowIdx])) return [];
      const raw = qEdge[rowIdx].map((v) => Number(v));
      if (commitmentMode !== 'centered' || !centeredEdgeRefIdxs.length) return raw;
      const eps = 1e-9;
      const out = new Array(raw.length).fill(NaN);
      for (let col = 0; col < raw.length; col += 1) {
        const v = raw[col];
        if (!Number.isFinite(v)) continue;
        let before = 0;
        let at = 0;
        let n = 0;
        for (const ridx of centeredEdgeRefIdxs) {
          const rv = Number(qEdge?.[ridx]?.[col]);
          if (!Number.isFinite(rv)) continue;
          n += 1;
          if (rv < v - eps) before += 1;
          else if (Math.abs(rv - v) <= eps) at += 1;
        }
        out[col] = n > 0 ? (before + 0.5 * at) / n : v;
      }
      return out;
    };

    // If dh/p_node aren't available (older analyses), fall back to the base q.
    const canAlt = Boolean(dhTable && pNode && Array.isArray(dhTable?.[0]) && Array.isArray(pNode?.[0]?.[0]));
    if (commitmentMode !== 'prob' && !canAlt) {
      const z = plotRows.map((row) => residueAxis.indices.map((i) => base(row, i)));
      const y = plotRows.map((row) => analysisSampleLabels[row] ?? String(analysisSampleIds[row] ?? row));
      return { z, y, x: residueAxisLabels };
    }

    const Kmax = dhTable && Array.isArray(dhTable[0]) ? dhTable[0].length : 0;
    const z = plotRows.map((row) => {
      if (commitmentMode === 'prob') return residueAxis.indices.map((i) => base(row, i));

      if (commitmentMode === 'centered') {
        if (!centeredCalib || !Kmax) return residueAxis.indices.map((i) => base(row, i));
        const { thresholds, alphas, eps } = centeredCalib;
        const outFull = new Array(N);
        for (let i = 0; i < N; i += 1) outFull[i] = base(row, i);
        for (let idx = 0; idx < residueAxis.indices.length; idx += 1) {
          const i = residueAxis.indices[idx];
          const dhRow = dhTable?.[i];
          const prow = pNode?.[row]?.[i];
          const Ki = kList && Number.isFinite(kList[i]) ? Math.max(0, Math.min(Kmax, Math.floor(kList[i]))) : Kmax;
          if (!Array.isArray(dhRow) || !Array.isArray(prow) || dhRow.length < Ki || prow.length < Ki || Ki <= 0) {
            outFull[i] = base(row, i);
            // eslint-disable-next-line no-continue
            continue;
          }
          const t = Number(thresholds[i]);
          const alpha = Number(alphas[i]);
          let accBefore = 0;
          let accAt = 0;
          for (let a = 0; a < Ki; a += 1) {
            const dhv = Number(dhRow[a]);
            const pv = Number(prow[a]) || 0;
            if (dhv < t - eps) accBefore += pv;
            else if (Math.abs(dhv - t) <= eps) accAt += pv;
          }
          outFull[i] = clamp01(accBefore + alpha * accAt);
        }
        if (!edgeSmoothEnabled) return residueAxis.indices.map((i) => outFull[i]);
        const strength = clamp(Number(edgeSmoothStrength), 0, 1);
        if (strength <= 0) return residueAxis.indices.map((i) => outFull[i]);
        if (!Array.isArray(qEdge) || !Array.isArray(qEdge[row]) || !Array.isArray(topEdgeIndices) || !topEdgeIndices.length) {
          return residueAxis.indices.map((i) => outFull[i]);
        }
        const rowQe = edgeRowForSmoothing(row);
        const sumW = new Array(N).fill(0);
        const sumWD = new Array(N).fill(0);
        for (let col = 0; col < topEdgeIndices.length && col < rowQe.length; col += 1) {
          const eidx = Number(topEdgeIndices[col]);
          const e = edges[eidx];
          if (!Array.isArray(e) || e.length < 2) continue;
          const r = Number(e[0]);
          const s = Number(e[1]);
          if (!Number.isInteger(r) || !Number.isInteger(s) || r < 0 || s < 0 || r >= N || s >= N) continue;
          const qv = rowQe[col];
          if (!Number.isFinite(qv)) continue;
          const wRaw = dEdge && Number.isFinite(Number(dEdge[eidx])) ? Math.abs(Number(dEdge[eidx])) : 1.0;
          const w = wRaw > 1e-12 ? wRaw : 1.0;
          const d = clamp01(qv) - 0.5;
          sumW[r] += w;
          sumWD[r] += w * d;
          sumW[s] += w;
          sumWD[s] += w * d;
        }
        const sm = new Array(N);
        for (let i = 0; i < N; i += 1) {
          const di = outFull[i] - 0.5;
          const de = sumW[i] > 0 ? sumWD[i] / sumW[i] : 0;
          sm[i] = clamp01(0.5 + (1 - strength) * di + strength * de);
        }
        return residueAxis.indices.map((i) => sm[i]);
      }

      if (commitmentMode === 'mu_sigmoid') {
        if (!Kmax) return residueAxis.indices.map((i) => base(row, i));
        const mu = new Array(N);
        for (let i = 0; i < N; i += 1) {
          const dhRow = dhTable?.[i];
          const prow = pNode?.[row]?.[i];
          const Ki = kList && Number.isFinite(kList[i]) ? Math.max(0, Math.min(Kmax, Math.floor(kList[i]))) : Kmax;
          if (!Array.isArray(dhRow) || !Array.isArray(prow) || dhRow.length < Ki || prow.length < Ki || Ki <= 0) {
            mu[i] = NaN;
            // eslint-disable-next-line no-continue
            continue;
          }
          let m = 0;
          for (let a = 0; a < Ki; a += 1) m += (Number(prow[a]) || 0) * (Number(dhRow[a]) || 0);
          mu[i] = m;
        }
        const finite = mu.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
        const med = finite.length ? finite[Math.floor(finite.length / 2)] : 0;
        const absDev = finite.map((v) => Math.abs(v - med)).sort((a, b) => a - b);
        const mad = absDev.length ? absDev[Math.floor(absDev.length / 2)] : 0;
        const scale = mad > 1e-9 ? mad : 1.0;
        const outFull = new Array(N);
        for (let i = 0; i < N; i += 1) outFull[i] = clamp01(sigmoid(-(Number(mu[i]) || 0) / scale));
        if (!edgeSmoothEnabled) return residueAxis.indices.map((i) => outFull[i]);
        const strength = clamp(Number(edgeSmoothStrength), 0, 1);
        if (strength <= 0) return residueAxis.indices.map((i) => outFull[i]);
        if (!Array.isArray(qEdge) || !Array.isArray(qEdge[row]) || !Array.isArray(topEdgeIndices) || !topEdgeIndices.length) {
          return residueAxis.indices.map((i) => outFull[i]);
        }
        const rowQe = edgeRowForSmoothing(row);
        const sumW = new Array(N).fill(0);
        const sumWD = new Array(N).fill(0);
        for (let col = 0; col < topEdgeIndices.length && col < rowQe.length; col += 1) {
          const eidx = Number(topEdgeIndices[col]);
          const e = edges[eidx];
          if (!Array.isArray(e) || e.length < 2) continue;
          const r = Number(e[0]);
          const s = Number(e[1]);
          if (!Number.isInteger(r) || !Number.isInteger(s) || r < 0 || s < 0 || r >= N || s >= N) continue;
          const qv = rowQe[col];
          if (!Number.isFinite(qv)) continue;
          const wRaw = dEdge && Number.isFinite(Number(dEdge[eidx])) ? Math.abs(Number(dEdge[eidx])) : 1.0;
          const w = wRaw > 1e-12 ? wRaw : 1.0;
          const d = clamp01(qv) - 0.5;
          sumW[r] += w;
          sumWD[r] += w * d;
          sumW[s] += w;
          sumWD[s] += w * d;
        }
        const sm = new Array(N);
        for (let i = 0; i < N; i += 1) {
          const di = outFull[i] - 0.5;
          const de = sumW[i] > 0 ? sumWD[i] / sumW[i] : 0;
          sm[i] = clamp01(0.5 + (1 - strength) * di + strength * de);
        }
        return residueAxis.indices.map((i) => sm[i]);
      }

      // prob case
      const outFull = new Array(N);
      for (let i = 0; i < N; i += 1) outFull[i] = base(row, i);
      if (!edgeSmoothEnabled) return residueAxis.indices.map((i) => outFull[i]);
      const strength = clamp(Number(edgeSmoothStrength), 0, 1);
      if (strength <= 0) return residueAxis.indices.map((i) => outFull[i]);
      if (!Array.isArray(qEdge) || !Array.isArray(qEdge[row]) || !Array.isArray(topEdgeIndices) || !topEdgeIndices.length) {
        return residueAxis.indices.map((i) => outFull[i]);
      }
      const rowQe = edgeRowForSmoothing(row);
      const sumW = new Array(N).fill(0);
      const sumWD = new Array(N).fill(0);
      for (let col = 0; col < topEdgeIndices.length && col < rowQe.length; col += 1) {
        const eidx = Number(topEdgeIndices[col]);
        const e = edges[eidx];
        if (!Array.isArray(e) || e.length < 2) continue;
        const r = Number(e[0]);
        const s = Number(e[1]);
        if (!Number.isInteger(r) || !Number.isInteger(s) || r < 0 || s < 0 || r >= N || s >= N) continue;
        const qv = rowQe[col];
        if (!Number.isFinite(qv)) continue;
        const wRaw = dEdge && Number.isFinite(Number(dEdge[eidx])) ? Math.abs(Number(dEdge[eidx])) : 1.0;
        const w = wRaw > 1e-12 ? wRaw : 1.0;
        const d = clamp01(qv) - 0.5;
        sumW[r] += w;
        sumWD[r] += w * d;
        sumW[s] += w;
        sumWD[s] += w * d;
      }
      const sm = new Array(N);
      for (let i = 0; i < N; i += 1) {
        const di = outFull[i] - 0.5;
        const de = sumW[i] > 0 ? sumWD[i] / sumW[i] : 0;
        sm[i] = clamp01(0.5 + (1 - strength) * di + strength * de);
      }
      return residueAxis.indices.map((i) => sm[i]);
    });
    const singleMaskCols = residueAxis.indices.map((i) => Boolean(singleClusterByResidue[i]));
    const colKeepMask = singleMaskCols.map((isSingle) => (hideSingleClusterResidues ? !isSingle : true));
    const keptCols = colKeepMask.reduce((acc, keep) => acc + (keep ? 1 : 0), 0);
    if (keptCols <= 0) return null;

    const filteredX = residueAxisLabels.filter((_, col) => colKeepMask[col]);
    const zFiltered = z.map((row) => row.filter((_, col) => colKeepMask[col]));
    const singleFiltered = singleMaskCols.filter((_, col) => colKeepMask[col]);
    const zBase = singleFiltered.some(Boolean)
      ? zFiltered.map((row) => row.map((v, col) => (singleFiltered[col] ? null : v)))
      : zFiltered;
    const zSingle = singleFiltered.some(Boolean)
      ? zFiltered.map((row) => row.map((_, col) => (singleFiltered[col] ? 1 : null)))
      : null;
    const y = plotRows.map((row) => analysisSampleLabels[row] ?? String(analysisSampleIds[row] ?? row));
    return { z: zBase, zSingle, y, x: filteredX, hasSingle: singleFiltered.some(Boolean) };
  }, [
    plotRows,
    qResidueAll,
    residueAxis,
    analysisSampleLabels,
    analysisSampleIds,
    residueAxisLabels,
    singleClusterByResidue,
    hideSingleClusterResidues,
    commitmentMode,
    dhTable,
    pNode,
    centeredCalib,
    residueLabels.length,
    kList,
    edgeSmoothEnabled,
    edgeSmoothStrength,
    qEdge,
    centeredEdgeRefIdxs,
    topEdgeIndices,
    edges,
    dEdge,
  ]);

  const qEdgeHeatmap = useMemo(() => {
    if (!plotRows.length || !Array.isArray(qEdge) || !qEdge.length) return null;
    if (!edgeAxisLabels.length) return null;
    const z = plotRows.map((row) => qEdge[row].map((v) => clamp01(Number(v))));
    if (!z.length || !z[0]?.length) return null;
    const y = plotRows.map((row) => analysisSampleLabels[row] ?? String(analysisSampleIds[row] ?? row));
    return { z, y, x: edgeAxisLabels };
  }, [plotRows, qEdge, analysisSampleLabels, analysisSampleIds, edgeAxisLabels]);

  const energyBinsArr = useMemo(() => (Array.isArray(data?.data?.energy_bins) ? data.data.energy_bins : []), [data]);
  const energyHist = useMemo(() => (Array.isArray(data?.data?.energy_hist) ? data.data.energy_hist : []), [data]);

  const energyPlot = useMemo(() => {
    if (!plotRows.length || !Array.isArray(energyBinsArr) || energyBinsArr.length < 2) return null;
    if (!Array.isArray(energyHist) || !energyHist.length) return null;
    const centers = [];
    for (let i = 0; i < energyBinsArr.length - 1; i += 1) {
      centers.push(0.5 * (Number(energyBinsArr[i]) + Number(energyBinsArr[i + 1])));
    }
    const traces = plotRows.map((row, idx) => ({
      type: 'scatter',
      mode: 'lines',
      name: analysisSampleLabels[row] ?? String(analysisSampleIds[row] ?? row),
      x: centers,
      y: energyHist[row].map((v) => Number(v)),
      line: { color: pickColor(idx), width: 2 },
      opacity: 0.9,
    }));
    return { traces };
  }, [plotRows, energyBinsArr, energyHist, analysisSampleLabels, analysisSampleIds]);

  const missingSamplesForPlot = useMemo(() => {
    const missing = [];
    for (const sid of plotSampleIds) {
      if (!analysisSampleIndexById.has(String(sid))) missing.push(String(sid));
    }
    return missing;
  }, [plotSampleIds, analysisSampleIndexById]);

  if (loadingSystem) return <Loader message="Loading Delta Potts Evaluation..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Delta Potts Evaluation: How To Read It"
        docPath="/docs/delta_potts_commitment_help.md"
        onClose={() => setHelpOpen(false)}
      />

      <div className="flex items-start justify-between gap-3">
        <div>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)}
            className="text-cyan-400 hover:text-cyan-300 text-sm"
          >
            ← Back to system
          </button>
          <h1 className="text-2xl font-semibold text-white">Delta Potts Evaluation</h1>
          <p className="text-sm text-gray-400">
            Select two delta models (A,B), pick samples, and compute per-residue/per-edge commitment (incremental store).
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
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_commitment_3d${samplingSuffix}`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            3D Commitment View
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_js${samplingSuffix}`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Delta JS View
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[380px_1fr] gap-4">
        <aside className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-200">Selection</h2>
              <button
                type="button"
                onClick={async () => {
                  await loadClusterInfo();
                  await loadAnalyses();
                }}
                className="text-xs px-2 py-1 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
              >
                <RefreshCw className="h-3 w-3" />
                Refresh
              </button>
            </div>

            <div className="space-y-1">
              <label className="block text-xs text-gray-400">Cluster</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-3 py-2 text-sm text-gray-100"
              >
                {clusterOptions.map((c) => (
                  <option key={c.cluster_id} value={c.cluster_id}>
                    {c.name || c.cluster_id}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">Model A</label>
                <select
                  value={modelAId}
                  onChange={(e) => setModelAId(e.target.value)}
                  disabled={!deltaModels.length}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 disabled:opacity-60"
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
                <label className="block text-xs text-gray-400">Model B</label>
                <select
                  value={modelBId}
                  onChange={(e) => setModelBId(e.target.value)}
                  disabled={!deltaModels.length}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 disabled:opacity-60"
                >
                  {!deltaModels.length && <option value="">No delta models</option>}
                  {deltaModels.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.name || m.model_id}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">MD labels</label>
                <select
                  value={mdLabelMode}
                  onChange={(e) => setMdLabelMode(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  <option value="assigned">assigned</option>
                  <option value="halo">halo</option>
                </select>
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 text-sm text-gray-200">
                  <input
                    type="checkbox"
                    checked={keepInvalid}
                    onChange={(e) => setKeepInvalid(e.target.checked)}
                    className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                  />
                  Keep invalid
                </label>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">Residues to show (0 = all)</label>
                <input
                  type="number"
                  min="0"
                  value={topKResidues}
                  onChange={(e) => setTopKResidues(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
              </div>
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">Top edges</label>
                <input
                  type="number"
                  min="1"
                  value={topKEdges}
                  onChange={(e) => setTopKEdges(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
              </div>
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">Energy bins</label>
                <input
                  type="number"
                  min="20"
                  value={energyBins}
                  onChange={(e) => setEnergyBins(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">Commitment mode</label>
                <select
                  value={commitmentMode}
                  onChange={(e) => setCommitmentMode(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  <option value="prob">Base: Pr(Δh &lt; 0)</option>
                  <option value="centered" disabled={!hasAltCommitmentData}>
                    Centered: Pr(Δh ≤ median(ref))
                  </option>
                  <option value="mu_sigmoid" disabled={!hasAltCommitmentData}>
                    Mean field (smooth)
                  </option>
                </select>
                <p className="text-[11px] text-gray-500">
                  Centered mode helps when ensembles share discrete marginals (mixtures), so "neutral" residues appear closer to white.
                </p>
                {!hasAltCommitmentData && (
                  <p className="text-[11px] text-yellow-300">
                    Centered/Mean require analysis artifacts generated with the latest backend. Re-run commitment analysis for this (A,B) pair.
                  </p>
                )}
              </div>
              <div className="space-y-1">
                <label className="block text-xs text-gray-400">Reference ensemble(s)</label>
                <select
                  multiple
                  disabled={commitmentMode !== 'centered' || !hasAltCommitmentData}
                  value={referenceSampleIds}
                  onChange={(e) => {
                    const opts = Array.from(e.target.selectedOptions).map((o) => String(o.value));
                    setReferenceSampleIds(opts);
                  }}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 h-24 disabled:opacity-60"
                >
                  {analysisSampleIds.map((sid, idx) => (
                    <option key={`ref:${sid}`} value={sid}>
                      {analysisSampleTypes[idx] ? `${analysisSampleLabels[idx] || sid} (${analysisSampleTypes[idx]})` : analysisSampleLabels[idx] || sid}
                    </option>
                  ))}
                </select>
                <p className="text-[11px] text-gray-500">
                  Uses stored node marginals <code>p_i(a)</code>. If you just ran analysis, reload data to enable centered/mean modes.
                </p>
              </div>
            </div>

            <div className="rounded-md border border-gray-800 bg-gray-950/30 p-2 space-y-2">
              <label className="flex items-center gap-2 text-sm text-gray-200">
                <input
                  type="checkbox"
                  checked={edgeSmoothEnabled}
                  onChange={(e) => setEdgeSmoothEnabled(e.target.checked)}
                  className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                />
                Edge-weighted residue coloring (uses top edges)
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 items-end">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Smoothing strength (0..1)</label>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={edgeSmoothStrength}
                    onChange={(e) => setEdgeSmoothStrength(Number(e.target.value))}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                    disabled={!edgeSmoothEnabled}
                  />
                </div>
                <p className="text-[11px] text-gray-500">
                  Each residue value is blended with the average commitment of its incident top edges (weighted by |ΔJ|).
                </p>
              </div>
              <label className="flex items-center gap-2 text-sm text-gray-200">
                <input
                  type="checkbox"
                  checked={hideSingleClusterResidues}
                  onChange={(e) => setHideSingleClusterResidues(e.target.checked)}
                  className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                />
                Hide single-cluster residues from heatmaps
              </label>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="block text-xs text-gray-400">Samples to compute</label>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setSelectedSampleIds(sampleEntries.filter((s) => s.type === 'md_eval').map((s) => s.sample_id))}
                    className="text-[11px] px-2 py-1 rounded border border-gray-800 text-gray-200 hover:border-gray-600"
                  >
                    MD only
                  </button>
                  <button
                    type="button"
                    onClick={() => setSelectedSampleIds(sampleEntries.map((s) => s.sample_id))}
                    className="text-[11px] px-2 py-1 rounded border border-gray-800 text-gray-200 hover:border-gray-600"
                  >
                    All
                  </button>
                </div>
              </div>
              <div className="max-h-[260px] overflow-auto rounded-md border border-gray-800 bg-gray-950/30">
                {sampleEntries.map((s) => {
                  const sid = s.sample_id;
                  const checked = selectedSampleIds.includes(sid);
                  return (
                    <label key={sid} className="flex items-center justify-between gap-3 px-3 py-2 border-b border-gray-900 text-sm text-gray-200">
                      <span className="min-w-0 truncate">
                        {s.name || sid} <span className="text-[11px] text-gray-500">({s.type || 'sample'})</span>
                      </span>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => {
                          const on = e.target.checked;
                          setSelectedSampleIds((prev) => {
                            if (on) return prev.includes(sid) ? prev : [...prev, sid];
                            return prev.filter((x) => x !== sid);
                          });
                        }}
                        className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                      />
                    </label>
                  );
                })}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleRun}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white text-sm disabled:opacity-60"
                disabled={!selectedClusterId || !modelAId || !modelBId || modelAId === modelBId || !selectedSampleIds.length}
              >
                <Play className="h-4 w-4" />
                Run commitment
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
            {clusterInfoError && <ErrorMessage message={clusterInfoError} />}
            {analysesError && <ErrorMessage message={analysesError} />}
          </div>

          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
            <label className="block text-xs text-gray-400">Samples to plot (must exist in analysis)</label>
            <div className="max-h-[220px] overflow-auto rounded-md border border-gray-800 bg-gray-950/30">
              {sampleEntries.map((s) => {
                const sid = s.sample_id;
                const has = analysisSampleIndexById.has(String(sid));
                const checked = plotSampleIds.includes(sid);
                return (
                  <label key={`plot:${sid}`} className="flex items-center justify-between gap-3 px-3 py-2 border-b border-gray-900 text-sm text-gray-200">
                    <span className="min-w-0 truncate">
                      {s.name || sid}{' '}
                      {!has ? <span className="text-[11px] text-yellow-500">missing</span> : <span className="text-[11px] text-gray-500">ok</span>}
                    </span>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => {
                        const on = e.target.checked;
                        setPlotSampleIds((prev) => {
                          if (on) return prev.includes(sid) ? prev : [...prev, sid];
                          return prev.filter((x) => x !== sid);
                        });
                      }}
                      className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                    />
                  </label>
                );
              })}
            </div>
            {!!missingSamplesForPlot.length && (
              <div className="text-[11px] text-yellow-200">
                Missing analyses for: <span className="font-mono">{missingSamplesForPlot.join(', ')}</span> (run commitment to compute them).
              </div>
            )}
          </div>
        </aside>

        <main className="space-y-4">
          {dataError && <ErrorMessage message={dataError} />}
          {dataLoading && <Loader message="Loading analysis..." />}

          {!dataLoading && !data && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 text-sm text-gray-300">
              No commitment analysis found for this (A,B,params) selection. Click <span className="font-semibold">Run commitment</span>.
            </div>
          )}

          {!!qResidueHeatmap && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
              <h2 className="text-sm font-semibold text-gray-200">Per-Residue Commitment</h2>
              <p className="text-[11px] text-gray-500">
                Heatmap of <code>q_i = Pr(dh_i(X_i) &lt; 0)</code> for top residues (ranking: parameter L2).
              </p>
              <Plot
                data={[
                  {
                    type: 'heatmap',
                    z: qResidueHeatmap.z,
                    x: qResidueHeatmap.x,
                    y: qResidueHeatmap.y,
                    zmin: 0,
                    zmax: 1,
                    colorscale: [
                      [0, '#2563eb'],
                      [0.5, '#ffffff'],
                      [1, '#dc2626'],
                    ],
                    hovertemplate: 'sample=%{y}<br>res=%{x}<br>q=%{z:.3f}<extra></extra>',
                  },
                  ...(qResidueHeatmap.hasSingle && qResidueHeatmap.zSingle
                    ? [
                        {
                          type: 'heatmap',
                          z: qResidueHeatmap.zSingle,
                          x: qResidueHeatmap.x,
                          y: qResidueHeatmap.y,
                          zmin: 0,
                          zmax: 1,
                          colorscale: [
                            [0, '#9ca3af'],
                            [1, '#9ca3af'],
                          ],
                          showscale: false,
                          hovertemplate: 'sample=%{y}<br>res=%{x}<br>single-cluster residue<extra></extra>',
                        },
                      ]
                    : []),
                ]}
                layout={{
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 140, r: 20, t: 10, b: 90 },
                  height: 420,
                  xaxis: { tickangle: -45, automargin: true },
                  yaxis: { automargin: true },
                }}
                config={{ responsive: true, displaylogo: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}
          {!qResidueHeatmap && hideSingleClusterResidues && data && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 text-sm text-gray-300">
              Per-residue heatmap hidden: all currently selected residues are single-cluster and filtered out.
            </div>
          )}

          {!!qEdgeHeatmap && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
              <h2 className="text-sm font-semibold text-gray-200">Per-Edge Commitment</h2>
              <p className="text-[11px] text-gray-500">
                Heatmap of <code>q_ij = Pr(dJ_ij(X_i,X_j) &lt; 0)</code> for top edges (ranking: parameter L2).
              </p>
              <Plot
                data={[
                  {
                    type: 'heatmap',
                    z: qEdgeHeatmap.z,
                    x: qEdgeHeatmap.x,
                    y: qEdgeHeatmap.y,
                    zmin: 0,
                    zmax: 1,
                    colorscale: [
                      [0, '#2563eb'],
                      [0.5, '#ffffff'],
                      [1, '#dc2626'],
                    ],
                    hovertemplate: 'sample=%{y}<br>edge=%{x}<br>q=%{z:.3f}<extra></extra>',
                  },
                ]}
                layout={{
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 140, r: 20, t: 10, b: 120 },
                  height: 420,
                  xaxis: { tickangle: -45, automargin: true },
                  yaxis: { automargin: true },
                }}
                config={{ responsive: true, displaylogo: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {!!energyPlot && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
              <h2 className="text-sm font-semibold text-gray-200">ΔE Distributions</h2>
              <p className="text-[11px] text-gray-500">
                Density estimates from histograms of <code>ΔE = E_A - E_B</code> (shared bins across samples).
              </p>
              <Plot
                data={energyPlot.traces}
                layout={{
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 60, r: 20, t: 10, b: 50 },
                  height: 340,
                  xaxis: { title: 'ΔE', zeroline: false },
                  yaxis: { title: 'density', zeroline: false },
                  legend: { orientation: 'h' },
                }}
                config={{ responsive: true, displaylogo: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
