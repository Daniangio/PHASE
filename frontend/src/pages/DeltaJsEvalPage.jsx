import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { ChevronDown, ChevronRight, CircleHelp, Play, RefreshCw } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import JsRangeFilterBuilder, { passesAnyJsFilter } from '../components/common/JsRangeFilterBuilder';
import FilterSetupManager from '../components/common/FilterSetupManager';
import {
  fetchClusterAnalyses,
  fetchClusterAnalysisData,
  fetchClusterUiSetups,
  saveClusterUiSetup,
  deleteClusterUiSetup,
  fetchPottsClusterInfo,
  fetchSystem,
} from '../api/projects';
import { fetchJobStatus, submitDeltaJsJob } from '../api/jobs';

const JS_MAX = Math.log(2);

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function normJs(x) {
  return clamp01(Number(x) / JS_MAX);
}

function edgeLabel(edge, residueLabels) {
  if (!Array.isArray(edge) || edge.length < 2) return '';
  const r = Number(edge[0]);
  const s = Number(edge[1]);
  const a = residueLabels[r] ?? String(r);
  const b = residueLabels[s] ?? String(s);
  return `${a}-${b}`;
}

function rgba(r, g, b, a = 1) {
  return `rgba(${Math.round(r)},${Math.round(g)},${Math.round(b)},${a})`;
}

function jsABOWeights(dA, dB) {
  const cA = 1 - normJs(dA);
  const cB = 1 - normJs(dB);
  let wA = cA * (1 - cB);
  let wB = cB * (1 - cA);
  let wShared = cA * cB;
  let wOther = (1 - cA) * (1 - cB);
  const s = wA + wB + wShared + wOther;
  if (s <= 0) return { A: 0.25, B: 0.25, shared: 0.25, other: 0.25 };
  wA /= s;
  wB /= s;
  wShared /= s;
  wOther /= s;
  return { A: wA, B: wB, shared: wShared, other: wOther };
}

function jsABOColor(dA, dB, alpha = 1) {
  const w = jsABOWeights(dA, dB);
  const cA = [227, 74, 51]; // red
  const cB = [49, 130, 189]; // blue
  const cShared = [44, 162, 95]; // green
  const cOther = [148, 103, 189]; // purple
  const r = w.A * cA[0] + w.B * cB[0] + w.shared * cShared[0] + w.other * cOther[0];
  const g = w.A * cA[1] + w.B * cB[1] + w.shared * cShared[1] + w.other * cOther[1];
  const b = w.A * cA[2] + w.B * cB[2] + w.shared * cShared[2] + w.other * cOther[2];
  return rgba(r, g, b, alpha);
}

function aboLabel(dA, dB) {
  const w = jsABOWeights(dA, dB);
  const arr = [
    ['A-like', w.A],
    ['B-like', w.B],
    ['Similar to both', w.shared],
    ['Far from both', w.other],
  ].sort((a, b) => b[1] - a[1]);
  return arr[0][0];
}

export default function DeltaJsEvalPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [clusterInfo, setClusterInfo] = useState(null);
  const [clusterInfoError, setClusterInfoError] = useState(null);

  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [keepInvalid, setKeepInvalid] = useState(false);
  const [topKResidues, setTopKResidues] = useState(0);
  const [topKEdges, setTopKEdges] = useState(2000);
  const [hideSingleClusterResidues, setHideSingleClusterResidues] = useState(false);
  const [jsFilters, setJsFilters] = useState([{ aMin: 0, aMax: 1, bMin: 0, bMax: 1 }]);
  const [edgeBlendEnabled, setEdgeBlendEnabled] = useState(false);
  const [edgeBlendStrength, setEdgeBlendStrength] = useState(0.75);

  const [modelAId, setModelAId] = useState('');
  const [modelBId, setModelBId] = useState('');
  const [useModelPair, setUseModelPair] = useState(false);
  const [edgeMode, setEdgeMode] = useState('contact');
  const [contactStateIds, setContactStateIds] = useState([]);
  const [contactPdbs, setContactPdbs] = useState('');
  const [contactCutoff, setContactCutoff] = useState(10.0);
  const [contactAtomMode, setContactAtomMode] = useState('CA');
  const [refSampleIdsA, setRefSampleIdsA] = useState([]);
  const [refSampleIdsB, setRefSampleIdsB] = useState([]);

  const [selectedSampleIds, setSelectedSampleIds] = useState([]);
  const [plotSampleIds, setPlotSampleIds] = useState([]);

  const [analyses, setAnalyses] = useState([]);
  const [analysesError, setAnalysesError] = useState(null);
  const [data, setData] = useState(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState(null);

  const analysisDataCacheRef = useRef({});
  const analysisInFlightRef = useRef({});

  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [filterSetups, setFilterSetups] = useState([]);
  const [filterSetupsError, setFilterSetupsError] = useState(null);
  const [selectedFilterSetupId, setSelectedFilterSetupId] = useState('');
  const [newFilterSetupName, setNewFilterSetupName] = useState('');
  const [runPanelOpen, setRunPanelOpen] = useState(false);

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
  const samplingSuffix = useMemo(() => {
    const params = new URLSearchParams();
    if (selectedClusterId) params.set('cluster_id', selectedClusterId);
    const s = params.toString();
    return s ? `?${s}` : '';
  }, [selectedClusterId]);
  const sampleEntries = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);
  const stateOptions = useMemo(() => {
    const raw = system?.states;
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    if (typeof raw === 'object') return Object.values(raw);
    return [];
  }, [system]);
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
  const modelById = useMemo(() => {
    const map = new Map();
    deltaModels.forEach((m) => map.set(String(m.model_id), m));
    return map;
  }, [deltaModels]);

  useEffect(() => {
    if (!clusterOptions.length) return;
    const qs = new URLSearchParams(location.search || '');
    const requested = String(qs.get('cluster_id') || '').trim();
    if (requested && clusterOptions.some((c) => c.cluster_id === requested)) {
      if (selectedClusterId !== requested) setSelectedClusterId(requested);
      return;
    }
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterId, location.search]);

  useEffect(() => {
    if (!useModelPair) {
      setModelAId('');
      setModelBId('');
      return;
    }
    if (!deltaModels.length) {
      setModelAId('');
      setModelBId('');
      return;
    }
    const ids = new Set(deltaModels.map((m) => m.model_id));
    if (!modelAId || !ids.has(modelAId)) setModelAId(deltaModels[0].model_id);
    if (!modelBId || !ids.has(modelBId)) {
      const other = deltaModels.find((m) => m.model_id !== (modelAId || deltaModels[0].model_id));
      setModelBId((other || deltaModels[0]).model_id);
    }
  }, [deltaModels, modelAId, modelBId, useModelPair]);

  const inferRefSamplesForModel = useCallback(
    (modelId) => {
      const model = modelById.get(String(modelId));
      const stateIds = Array.isArray(model?.params?.state_ids) ? model.params.state_ids.map(String) : [];
      const out = sampleEntries
        .filter((s) => String(s?.type || '') === 'md_eval')
        .filter((s) => (stateIds.length ? stateIds.includes(String(s?.state_id || '')) : true))
        .map((s) => String(s.sample_id));
      return out;
    },
    [modelById, sampleEntries]
  );

  useEffect(() => {
    if (!sampleEntries.length) {
      setSelectedSampleIds([]);
      setPlotSampleIds([]);
      return;
    }
    if (!selectedSampleIds.length) {
      const md = sampleEntries.filter((s) => s.type === 'md_eval').map((s) => s.sample_id);
      const initial = md.length ? md : sampleEntries.slice(0, 3).map((s) => s.sample_id);
      setSelectedSampleIds(initial);
      setPlotSampleIds(initial);
    }
  }, [sampleEntries, selectedSampleIds.length]);

  useEffect(() => {
    if (!useModelPair) return;
    if (!modelAId) return;
    if (!refSampleIdsA.length) setRefSampleIdsA(inferRefSamplesForModel(modelAId));
  }, [useModelPair, modelAId, refSampleIdsA.length, inferRefSamplesForModel]);

  useEffect(() => {
    if (!useModelPair) return;
    if (!modelBId) return;
    if (!refSampleIdsB.length) setRefSampleIdsB(inferRefSamplesForModel(modelBId));
  }, [useModelPair, modelBId, refSampleIdsB.length, inferRefSamplesForModel]);

  const loadClusterInfo = useCallback(async () => {
    if (!selectedClusterId) return;
    setClusterInfoError(null);
    try {
      const info = await fetchPottsClusterInfo(projectId, systemId, selectedClusterId, {
        modelId: useModelPair ? modelAId || undefined : undefined,
      });
      setClusterInfo(info);
    } catch (err) {
      setClusterInfoError(err.message || 'Failed to load cluster info.');
      setClusterInfo(null);
    }
  }, [projectId, systemId, selectedClusterId, modelAId, useModelPair]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesError(null);
    try {
      const res = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_js' });
      setAnalyses(Array.isArray(res?.analyses) ? res.analyses : []);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
    }
  }, [projectId, systemId, selectedClusterId]);

  const loadFilterSetups = useCallback(async () => {
    if (!selectedClusterId) return;
    setFilterSetupsError(null);
    try {
      const res = await fetchClusterUiSetups(projectId, systemId, selectedClusterId, {
        setupType: 'js_range_filters',
        page: 'delta_js',
      });
      const arr = Array.isArray(res?.setups) ? res.setups : [];
      setFilterSetups(arr);
      if (!arr.some((x) => String(x?.setup_id) === String(selectedFilterSetupId))) {
        setSelectedFilterSetupId(arr.length ? String(arr[0].setup_id) : '');
      }
    } catch (err) {
      setFilterSetups([]);
      setFilterSetupsError(err.message || 'Failed to load filter setups.');
    }
  }, [projectId, systemId, selectedClusterId, selectedFilterSetupId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    analysisDataCacheRef.current = {};
    analysisInFlightRef.current = {};
    setData(null);
    setJob(null);
    setJobStatus(null);
    setJobError(null);
    loadClusterInfo();
    loadAnalyses();
    loadFilterSetups();
  }, [selectedClusterId, loadClusterInfo, loadAnalyses, loadFilterSetups]);

  useEffect(() => {
    if (!selectedClusterId) return;
    loadClusterInfo();
  }, [selectedClusterId, modelAId, loadClusterInfo]);

  const selectedMeta = useMemo(() => {
    const dropInvalid = !keepInvalid;
    const expectedEdgeSource = useModelPair ? 'potts_intersection' : edgeMode;
    const base = analyses.filter((a) => {
      if ((a.md_label_mode || 'assigned').toLowerCase() !== mdLabelMode) return false;
      if (Boolean(a.drop_invalid) !== Boolean(dropInvalid)) return false;
      if (useModelPair) return a.model_a_id === modelAId && a.model_b_id === modelBId;
      if (a.model_a_id || a.model_b_id) return false;
      return String(a.edge_source || a.edge_mode || 'cluster') === expectedEdgeSource;
    });
    if (!base.length) return null;
    const scored = [...base].sort((x, y) => {
      const nx = Number(x?.summary?.n_samples || 0);
      const ny = Number(y?.summary?.n_samples || 0);
      if (ny !== nx) return ny - nx;
      const tx = Date.parse(String(x?.updated_at || x?.created_at || ''));
      const ty = Date.parse(String(y?.updated_at || y?.created_at || ''));
      return (Number.isFinite(ty) ? ty : 0) - (Number.isFinite(tx) ? tx : 0);
    });
    return scored[0] || null;
  }, [analyses, modelAId, modelBId, mdLabelMode, keepInvalid, useModelPair, edgeMode]);

  const loadAnalysisData = useCallback(
    async (analysisId) => {
      if (!analysisId) return null;
      const cacheKey = `delta_js:${analysisId}`;
      if (Object.prototype.hasOwnProperty.call(analysisDataCacheRef.current, cacheKey)) {
        return analysisDataCacheRef.current[cacheKey];
      }
      if (analysisInFlightRef.current[cacheKey]) return analysisInFlightRef.current[cacheKey];
      const p = fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'delta_js', analysisId)
        .then((payload) => {
          analysisDataCacheRef.current = { ...analysisDataCacheRef.current, [cacheKey]: payload };
          delete analysisInFlightRef.current[cacheKey];
          return payload;
        })
        .catch((err) => {
          delete analysisInFlightRef.current[cacheKey];
          throw err;
        });
      analysisInFlightRef.current[cacheKey] = p;
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
        setData(await loadAnalysisData(selectedMeta.analysis_id));
      } catch (err) {
        setDataError(err.message || 'Failed to load analysis data.');
      } finally {
        setDataLoading(false);
      }
    };
    run();
  }, [selectedMeta, loadAnalysisData]);

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

  const handleRun = useCallback(async () => {
    setJobError(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        sample_ids: selectedSampleIds,
        reference_sample_ids_a: refSampleIdsA,
        reference_sample_ids_b: refSampleIdsB,
        md_label_mode: mdLabelMode,
        keep_invalid: keepInvalid,
        top_k_residues: Number(topKResidues) > 0 ? Number(topKResidues) : undefined,
        top_k_edges: Number(topKEdges),
      };
      if (useModelPair) {
        payload.model_a_id = modelAId;
        payload.model_b_id = modelBId;
      } else {
        payload.edge_mode = edgeMode;
        if (edgeMode === 'contact') {
          if (contactStateIds.length) payload.contact_state_ids = contactStateIds;
          const extraPdbs = String(contactPdbs || '')
            .split(',')
            .map((x) => x.trim())
            .filter(Boolean);
          if (extraPdbs.length) payload.contact_pdbs = extraPdbs;
          payload.contact_cutoff = Number(contactCutoff);
          payload.contact_atom_mode = String(contactAtomMode || 'CA').toUpperCase();
        }
      }
      const res = await submitDeltaJsJob(payload);
      setJob(res);
      setJobStatus(null);
    } catch (err) {
      setJobError(err.message || 'Failed to submit delta JS job.');
    }
  }, [
    projectId,
    systemId,
    selectedClusterId,
    useModelPair,
    modelAId,
    modelBId,
    edgeMode,
    contactStateIds,
    contactPdbs,
    contactCutoff,
    contactAtomMode,
    selectedSampleIds,
    refSampleIdsA,
    refSampleIdsB,
    mdLabelMode,
    keepInvalid,
    topKResidues,
    topKEdges,
  ]);

  const handleSaveFilterSetup = useCallback(async () => {
    if (!selectedClusterId) return;
    const name = String(newFilterSetupName || '').trim();
    if (!name) {
      setFilterSetupsError('Provide a setup name.');
      return;
    }
    setFilterSetupsError(null);
    try {
      const saved = await saveClusterUiSetup(projectId, systemId, selectedClusterId, {
        name,
        setup_type: 'js_range_filters',
        page: 'delta_js',
        payload: { rules: jsFilters },
      });
      setNewFilterSetupName('');
      await loadFilterSetups();
      if (saved?.setup_id) setSelectedFilterSetupId(String(saved.setup_id));
    } catch (err) {
      setFilterSetupsError(err.message || 'Failed to save filter setup.');
    }
  }, [
    projectId,
    systemId,
    selectedClusterId,
    newFilterSetupName,
    jsFilters,
    loadFilterSetups,
  ]);

  const handleLoadFilterSetup = useCallback(() => {
    if (!selectedFilterSetupId) return;
    const entry = filterSetups.find((x) => String(x?.setup_id) === String(selectedFilterSetupId));
    const rules = entry?.payload?.rules;
    if (Array.isArray(rules) && rules.length) {
      setJsFilters(rules);
      setFilterSetupsError(null);
      return;
    }
    setFilterSetupsError('Selected setup has no valid rules payload.');
  }, [selectedFilterSetupId, filterSetups]);

  const handleDeleteFilterSetup = useCallback(async () => {
    if (!selectedClusterId || !selectedFilterSetupId) return;
    setFilterSetupsError(null);
    try {
      await deleteClusterUiSetup(projectId, systemId, selectedClusterId, selectedFilterSetupId);
      const removedId = selectedFilterSetupId;
      await loadFilterSetups();
      if (String(selectedFilterSetupId) === String(removedId)) setSelectedFilterSetupId('');
    } catch (err) {
      setFilterSetupsError(err.message || 'Failed to delete filter setup.');
    }
  }, [projectId, systemId, selectedClusterId, selectedFilterSetupId, loadFilterSetups]);

  const residueLabels = useMemo(() => {
    const keys = clusterInfo?.residue_keys || [];
    if (Array.isArray(keys) && keys.length) return keys;
    const n = clusterInfo?.n_residues || 0;
    return Array.from({ length: n }, (_, i) => `res_${i}`);
  }, [clusterInfo]);

  const singleClusterByResidue = useMemo(() => {
    const n = residueLabels.length;
    const out = new Array(n).fill(false);
    const source = Array.isArray(data?.data?.K_list)
      ? data.data.K_list
      : Array.isArray(clusterInfo?.cluster_counts)
      ? clusterInfo.cluster_counts
      : [];
    const m = Math.min(n, source.length);
    for (let i = 0; i < m; i += 1) {
      const ki = Number(source[i]);
      if (Number.isFinite(ki) && ki <= 1) out[i] = true;
    }
    return out;
  }, [data, clusterInfo, residueLabels.length]);

  const analysisSampleIds = useMemo(() => {
    const ids = data?.data?.sample_ids;
    return Array.isArray(ids) ? ids.map(String) : [];
  }, [data]);
  const analysisSampleLabels = useMemo(() => {
    const labels = data?.data?.sample_labels;
    if (Array.isArray(labels)) return labels.map(String);
    return analysisSampleIds;
  }, [data, analysisSampleIds]);
  const analysisSampleIndexById = useMemo(() => {
    const map = new Map();
    analysisSampleIds.forEach((sid, idx) => map.set(String(sid), idx));
    return map;
  }, [analysisSampleIds]);

  useEffect(() => {
    if (!analysisSampleIds.length) return;
    if (plotSampleIds.length) return;
    setPlotSampleIds(analysisSampleIds.slice(0, Math.min(6, analysisSampleIds.length)));
  }, [analysisSampleIds, plotSampleIds.length]);

  const plotRows = useMemo(() => {
    const rows = [];
    for (const sid of plotSampleIds) {
      const idx = analysisSampleIndexById.get(String(sid));
      if (idx == null) continue;
      rows.push(idx);
    }
    return rows;
  }, [plotSampleIds, analysisSampleIndexById]);

  const topResidueOrder = useMemo(() => {
    const src = Array.isArray(data?.data?.D_residue) ? data.data.D_residue : [];
    const pairs = src.map((v, i) => [Number(v), i]);
    pairs.sort((a, b) => Math.abs(b[0]) - Math.abs(a[0]));
    return pairs.map((p) => p[1]);
  }, [data]);

  const residueAxis = useMemo(() => {
    const total = residueLabels.length;
    const k = Number(topKResidues);
    if (!Number.isFinite(k) || k <= 0 || k >= total) {
      const idx = residueLabels.map((_, i) => i);
      return { indices: idx, labels: residueLabels };
    }
    const idx = topResidueOrder.slice(0, Math.max(1, Math.min(total, Math.floor(k))));
    return { indices: idx, labels: idx.map((i) => residueLabels[i] || String(i)) };
  }, [topKResidues, residueLabels, topResidueOrder]);

  const nodeMatrix = useMemo(() => {
    const a = Array.isArray(data?.data?.js_node_a) ? data.data.js_node_a : [];
    const b = Array.isArray(data?.data?.js_node_b) ? data.data.js_node_b : [];
    if (!a.length || !b.length || !plotRows.length) return null;

    const keepCol = residueAxis.indices.map((i) => {
      if (hideSingleClusterResidues && singleClusterByResidue[i]) return false;
      for (const row of plotRows) {
        const dA = Number(a?.[row]?.[i]);
        const dB = Number(b?.[row]?.[i]);
        if (passesAnyJsFilter(dA, dB, jsFilters)) return true;
      }
      return false;
    });
    const kept = keepCol.reduce((acc, x) => acc + (x ? 1 : 0), 0);
    if (kept <= 0) return null;

    const x = [];
    const y = [];
    const color = [];
    const custom = [];
    for (const row of plotRows) {
      for (let c = 0; c < residueAxis.indices.length; c += 1) {
        if (!keepCol[c]) continue;
        const ridx = residueAxis.indices[c];
        const dA = Number(a?.[row]?.[ridx]);
        const dB = Number(b?.[row]?.[ridx]);
        x.push(residueAxis.labels[c]);
        y.push(analysisSampleLabels[row] || String(analysisSampleIds[row] || row));
        color.push(singleClusterByResidue[ridx] ? 'rgba(156,163,175,1)' : jsABOColor(dA, dB));
        custom.push([dA, dB, aboLabel(dA, dB), singleClusterByResidue[ridx] ? 1 : 0]);
      }
    }
    return { x, y, color, custom };
  }, [
    data,
    plotRows,
    residueAxis,
    analysisSampleLabels,
    analysisSampleIds,
    hideSingleClusterResidues,
    singleClusterByResidue,
    jsFilters,
  ]);

  const edgeMatrix = useMemo(() => {
    const a = Array.isArray(data?.data?.js_edge_a) ? data.data.js_edge_a : [];
    const b = Array.isArray(data?.data?.js_edge_b) ? data.data.js_edge_b : [];
    const edges = Array.isArray(data?.data?.edges) ? data.data.edges : [];
    const top = Array.isArray(data?.data?.top_edge_indices) ? data.data.top_edge_indices : [];
    if (!a.length || !b.length || !plotRows.length || !top.length) return null;
    const keepCol = top.map((_, c) => {
      for (const row of plotRows) {
        const dA = Number(a?.[row]?.[c]);
        const dB = Number(b?.[row]?.[c]);
        if (passesAnyJsFilter(dA, dB, jsFilters)) return true;
      }
      return false;
    });
    if (!keepCol.some(Boolean)) return null;

    const xLabels = top.map((raw) => edgeLabel(edges[Number(raw)], residueLabels) || String(raw));
    const x = [];
    const y = [];
    const color = [];
    const custom = [];
    for (const row of plotRows) {
      for (let c = 0; c < xLabels.length; c += 1) {
        if (!keepCol[c]) continue;
        const dA = Number(a?.[row]?.[c]);
        const dB = Number(b?.[row]?.[c]);
        x.push(xLabels[c]);
        y.push(analysisSampleLabels[row] || String(analysisSampleIds[row] || row));
        color.push(jsABOColor(dA, dB));
        custom.push([dA, dB, aboLabel(dA, dB)]);
      }
    }
    return { x, y, color, custom };
  }, [data, plotRows, analysisSampleLabels, analysisSampleIds, residueLabels, jsFilters]);

  const blendedNodeMatrix = useMemo(() => {
    if (!edgeBlendEnabled) return null;
    const nodeA = Array.isArray(data?.data?.js_node_a) ? data.data.js_node_a : [];
    const nodeB = Array.isArray(data?.data?.js_node_b) ? data.data.js_node_b : [];
    const edgeA = Array.isArray(data?.data?.js_edge_a) ? data.data.js_edge_a : [];
    const edgeB = Array.isArray(data?.data?.js_edge_b) ? data.data.js_edge_b : [];
    const topEdgeIndices = Array.isArray(data?.data?.top_edge_indices) ? data.data.top_edge_indices : [];
    const edgesAll = Array.isArray(data?.data?.edges) ? data.data.edges : [];
    const dEdge = Array.isArray(data?.data?.D_edge) ? data.data.D_edge : [];
    if (!nodeA.length || !nodeB.length || !plotRows.length) return null;
    const N = residueAxis.indices.length;
    if (!N) return null;
    const strength = clamp01(Number(edgeBlendStrength));
    const rowA = [];
    const rowB = [];
    for (const row of plotRows) {
      const srcA = Array.isArray(nodeA[row]) ? nodeA[row] : [];
      const srcB = Array.isArray(nodeB[row]) ? nodeB[row] : [];
      const dA = residueAxis.indices.map((ridx) => Number(srcA[ridx]));
      const dB = residueAxis.indices.map((ridx) => Number(srcB[ridx]));
      const idxInAxis = new Map();
      for (let i = 0; i < residueAxis.indices.length; i += 1) idxInAxis.set(Number(residueAxis.indices[i]), i);

      if (strength > 0 && Array.isArray(edgeA[row]) && Array.isArray(edgeB[row]) && topEdgeIndices.length) {
        const sumW = new Array(N).fill(0);
        const sumWA = new Array(N).fill(0);
        const sumWB = new Array(N).fill(0);
        for (let col = 0; col < topEdgeIndices.length; col += 1) {
          const eidx = Number(topEdgeIndices[col]);
          const e = edgesAll[eidx];
          if (!Array.isArray(e) || e.length < 2) continue;
          const r = idxInAxis.get(Number(e[0]));
          const s = idxInAxis.get(Number(e[1]));
          if (r == null || s == null) continue;
          const dAe = Number(edgeA[row][col]);
          const dBe = Number(edgeB[row][col]);
          if (!Number.isFinite(dAe) || !Number.isFinite(dBe)) continue;
          const wr = Number(dEdge[eidx]);
          const w = Number.isFinite(wr) && Math.abs(wr) > 1e-12 ? Math.abs(wr) : 1.0;
          sumW[r] += w;
          sumWA[r] += w * dAe;
          sumWB[r] += w * dBe;
          sumW[s] += w;
          sumWA[s] += w * dAe;
          sumWB[s] += w * dBe;
        }
        for (let i = 0; i < N; i += 1) {
          const baseA = dA[i];
          const baseB = dB[i];
          const eA = sumW[i] > 0 ? sumWA[i] / sumW[i] : baseA;
          const eB = sumW[i] > 0 ? sumWB[i] / sumW[i] : baseB;
          dA[i] = (1 - strength) * baseA + strength * eA;
          dB[i] = (1 - strength) * baseB + strength * eB;
        }
      }
      rowA.push(dA);
      rowB.push(dB);
    }

    const keepCol = residueAxis.indices.map((ridx, c) => {
      if (hideSingleClusterResidues && singleClusterByResidue[ridx]) return false;
      for (let rr = 0; rr < plotRows.length; rr += 1) {
        const dA = Number(rowA[rr][c]);
        const dB = Number(rowB[rr][c]);
        if (passesAnyJsFilter(dA, dB, jsFilters)) return true;
      }
      return false;
    });
    if (!keepCol.some(Boolean)) return null;

    const x = [];
    const y = [];
    const color = [];
    const custom = [];
    for (let rr = 0; rr < plotRows.length; rr += 1) {
      const row = plotRows[rr];
      for (let c = 0; c < residueAxis.indices.length; c += 1) {
        if (!keepCol[c]) continue;
        const ridx = residueAxis.indices[c];
        const dA = Number(rowA[rr][c]);
        const dB = Number(rowB[rr][c]);
        x.push(residueAxis.labels[c]);
        y.push(analysisSampleLabels[row] || String(analysisSampleIds[row] || row));
        color.push(singleClusterByResidue[ridx] ? 'rgba(156,163,175,1)' : jsABOColor(dA, dB));
        custom.push([dA, dB, aboLabel(dA, dB), singleClusterByResidue[ridx] ? 1 : 0]);
      }
    }
    return { x, y, color, custom };
  }, [
    edgeBlendEnabled,
    edgeBlendStrength,
    data,
    plotRows,
    residueAxis,
    analysisSampleLabels,
    analysisSampleIds,
    hideSingleClusterResidues,
    singleClusterByResidue,
    jsFilters,
  ]);

  const contactPdbList = useMemo(
    () =>
      String(contactPdbs || '')
        .split(',')
        .map((x) => x.trim())
        .filter(Boolean),
    [contactPdbs]
  );
  const edgeConfigValid = useMemo(() => {
    if (useModelPair) return true;
    if (!edgeMode) return false;
    if (edgeMode !== 'contact') return true;
    return (contactStateIds.length > 0 || contactPdbList.length > 0) && Number(contactCutoff) > 0;
  }, [useModelPair, edgeMode, contactStateIds.length, contactPdbList.length, contactCutoff]);

  if (loadingSystem) return <Loader message="Loading delta JS analysis..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Delta JS (A/B/Other): Help"
        docPath="/docs/delta_js_help.md"
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
          <h1 className="text-2xl font-semibold text-white">Delta JS (A/B/Other)</h1>
          <p className="text-sm text-gray-400">
            Alternative to commitment: per-residue/per-edge JS distance to A and B references with 4-way color logic.
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
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_js_3d${samplingSuffix}`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            3D JS View
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
            <div>
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
            <label className="flex items-center gap-2 text-sm text-gray-200">
              <input
                type="checkbox"
                checked={useModelPair}
                onChange={(e) => setUseModelPair(e.target.checked)}
                className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
              />
              Use Potts model pair (optional)
            </label>
            {useModelPair && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-400">Model A</label>
                  <select
                    value={modelAId}
                    onChange={(e) => {
                      const v = e.target.value;
                      setModelAId(v);
                      setRefSampleIdsA(inferRefSamplesForModel(v));
                    }}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                  >
                    {deltaModels.map((m) => (
                      <option key={m.model_id} value={m.model_id}>
                        {m.name || m.model_id}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400">Model B</label>
                  <select
                    value={modelBId}
                    onChange={(e) => {
                      const v = e.target.value;
                      setModelBId(v);
                      setRefSampleIdsB(inferRefSamplesForModel(v));
                    }}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                  >
                    {deltaModels.map((m) => (
                      <option key={m.model_id} value={m.model_id}>
                        {m.name || m.model_id}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            )}
            {useModelPair && !deltaModels.length && (
              <div className="text-[11px] text-yellow-400">No delta models available on this cluster.</div>
            )}
            {!useModelPair && (
              <div className="rounded-md border border-gray-800 bg-gray-950/30 p-2 space-y-2">
                <div>
                  <label className="block text-xs text-gray-400">Edge mode (model-free)</label>
                  <select
                    value={edgeMode}
                    onChange={(e) => setEdgeMode(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                  >
                    <option value="contact">contact (same as Potts contact build)</option>
                    <option value="all_vs_all">all_vs_all</option>
                    <option value="cluster">cluster (use cluster.npz edges)</option>
                  </select>
                </div>
                {edgeMode === 'contact' && (
                  <>
                    <div>
                      <label className="block text-xs text-gray-400">Contact states (PDB source)</label>
                      <select
                        multiple
                        value={contactStateIds}
                        onChange={(e) =>
                          setContactStateIds(Array.from(e.target.selectedOptions).map((o) => String(o.value)))
                        }
                        className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 h-24"
                      >
                        {stateOptions.map((st) => {
                          const sid = st?.state_id || st?.id || '';
                          if (!sid) return null;
                          return (
                            <option key={`edge-state:${sid}`} value={String(sid)}>
                              {st?.name || sid}
                            </option>
                          );
                        })}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-gray-400">Extra contact PDB paths (comma separated)</label>
                      <input
                        type="text"
                        value={contactPdbs}
                        onChange={(e) => setContactPdbs(e.target.value)}
                        placeholder="structures/a.pdb,structures/b.pdb"
                        className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="block text-xs text-gray-400">Contact cutoff (A)</label>
                        <input
                          type="number"
                          min="0.1"
                          step="0.1"
                          value={contactCutoff}
                          onChange={(e) => setContactCutoff(e.target.value)}
                          className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-gray-400">Atom mode</label>
                        <select
                          value={contactAtomMode}
                          onChange={(e) => setContactAtomMode(String(e.target.value || 'CA').toUpperCase())}
                          className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                        >
                          <option value="CA">CA</option>
                          <option value="CM">CM</option>
                        </select>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
            <div className="grid grid-cols-2 gap-2">
              <div>
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
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400">Top residues</label>
                <input
                  type="number"
                  min="0"
                  value={topKResidues}
                  onChange={(e) => setTopKResidues(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400">Top edges</label>
                <input
                  type="number"
                  min="1"
                  value={topKEdges}
                  onChange={(e) => setTopKEdges(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
              </div>
            </div>
            <div className="grid grid-cols-1 gap-2">
              <label className="block text-xs text-gray-400">Reference A samples</label>
              <select
                multiple
                value={refSampleIdsA}
                onChange={(e) => setRefSampleIdsA(Array.from(e.target.selectedOptions).map((o) => String(o.value)))}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 h-24"
              >
                {sampleEntries.map((s) => (
                  <option key={`ra:${s.sample_id}`} value={String(s.sample_id)}>
                    {s.name || s.sample_id} ({s.type || 'sample'})
                  </option>
                ))}
              </select>
              <label className="block text-xs text-gray-400">Reference B samples</label>
              <select
                multiple
                value={refSampleIdsB}
                onChange={(e) => setRefSampleIdsB(Array.from(e.target.selectedOptions).map((o) => String(o.value)))}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 h-24"
              >
                {sampleEntries.map((s) => (
                  <option key={`rb:${s.sample_id}`} value={String(s.sample_id)}>
                    {s.name || s.sample_id} ({s.type || 'sample'})
                  </option>
                ))}
              </select>
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-200">
              <input
                type="checkbox"
                checked={hideSingleClusterResidues}
                onChange={(e) => setHideSingleClusterResidues(e.target.checked)}
                className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
              />
              Hide single-cluster residues from node heatmap
            </label>
            <JsRangeFilterBuilder rules={jsFilters} onChange={setJsFilters} />
            <FilterSetupManager
              setups={filterSetups}
              selectedSetupId={selectedFilterSetupId}
              onSelectedSetupIdChange={setSelectedFilterSetupId}
              newSetupName={newFilterSetupName}
              onNewSetupNameChange={setNewFilterSetupName}
              onLoad={handleLoadFilterSetup}
              onSave={handleSaveFilterSetup}
              onDelete={handleDeleteFilterSetup}
              error={filterSetupsError}
            />
            <div className="rounded-md border border-gray-800 bg-gray-950/30 p-2 space-y-2">
              <button
                type="button"
                onClick={() => setRunPanelOpen((v) => !v)}
                className="w-full inline-flex items-center justify-between text-sm text-gray-200"
              >
                <span>Run analysis</span>
                {runPanelOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              </button>
              {runPanelOpen && (
                <div className="space-y-2">
                  <label className="block text-xs text-gray-400">Samples to compute</label>
                  <div className="max-h-[220px] overflow-auto rounded-md border border-gray-800 bg-gray-950/30">
                    {sampleEntries.map((s) => {
                      const sid = String(s.sample_id);
                      const checked = selectedSampleIds.includes(sid);
                      return (
                        <label key={`sel:${sid}`} className="flex items-center justify-between gap-3 px-3 py-2 border-b border-gray-900 text-sm text-gray-200">
                          <span className="min-w-0 truncate">
                            {s.name || sid} <span className="text-[11px] text-gray-500">({s.type || 'sample'})</span>
                          </span>
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={(e) => {
                              const on = e.target.checked;
                              setSelectedSampleIds((prev) => (on ? (prev.includes(sid) ? prev : [...prev, sid]) : prev.filter((x) => x !== sid)));
                            }}
                            className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                          />
                        </label>
                      );
                    })}
                  </div>
                  <button
                    type="button"
                    onClick={handleRun}
                    disabled={
                      !selectedClusterId ||
                      !selectedSampleIds.length ||
                      !refSampleIdsA.length ||
                      !refSampleIdsB.length ||
                      !edgeConfigValid ||
                      (useModelPair && (!modelAId || !modelBId || modelAId === modelBId))
                    }
                    className="inline-flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white text-sm disabled:opacity-60"
                  >
                    <Play className="h-4 w-4" />
                    Run JS analysis
                  </button>
                  {job?.job_id && (
                    <div className="text-[11px] text-gray-300">
                      Job: <span className="text-gray-200">{job.job_id}</span>
                      {jobStatus?.meta?.status ? ` · ${jobStatus.meta.status}` : ''}
                      {typeof jobStatus?.meta?.progress === 'number' ? ` · ${jobStatus.meta.progress}%` : ''}
                    </div>
                  )}
                  {jobError && <ErrorMessage message={jobError} />}
                </div>
              )}
            </div>
            {clusterInfoError && <ErrorMessage message={clusterInfoError} />}
            {analysesError && <ErrorMessage message={analysesError} />}
          </div>

          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
            <label className="block text-xs text-gray-400">Samples to plot (must exist in analysis)</label>
            <div className="max-h-[220px] overflow-auto rounded-md border border-gray-800 bg-gray-950/30">
              {sampleEntries.map((s) => {
                const sid = String(s.sample_id);
                const has = analysisSampleIndexById.has(sid);
                const checked = plotSampleIds.includes(sid);
                return (
                  <label key={`plot:${sid}`} className="flex items-center justify-between gap-3 px-3 py-2 border-b border-gray-900 text-sm text-gray-200">
                    <span className="min-w-0 truncate">
                      {s.name || sid} {!has ? <span className="text-[11px] text-yellow-500">missing</span> : <span className="text-[11px] text-gray-500">ok</span>}
                    </span>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={(e) => {
                        const on = e.target.checked;
                        setPlotSampleIds((prev) => (on ? (prev.includes(sid) ? prev : [...prev, sid]) : prev.filter((x) => x !== sid)));
                      }}
                      className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                    />
                  </label>
                );
              })}
            </div>
          </div>
          <div className="rounded-md border border-gray-800 bg-gray-950/30 p-3 space-y-2">
            <label className="flex items-center gap-2 text-sm text-gray-200">
              <input
                type="checkbox"
                checked={edgeBlendEnabled}
                onChange={(e) => setEdgeBlendEnabled(e.target.checked)}
                className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
              />
              Edge-weighted node blending
            </label>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Smoothing strength (0..1)</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={edgeBlendStrength}
                onChange={(e) => setEdgeBlendStrength(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                disabled={!edgeBlendEnabled}
              />
            </div>
          </div>
        </aside>

        <main className="space-y-4">
          {dataError && <ErrorMessage message={dataError} />}
          {dataLoading && <Loader message="Loading analysis..." />}
          {!dataLoading && !data && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 text-sm text-gray-300">
              No delta JS analysis found for this selection. Click <span className="font-semibold">Run JS analysis</span>.
            </div>
          )}

          {nodeMatrix && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
              <h2 className="text-sm font-semibold text-gray-200">Per-Residue JS A/B/Other</h2>
              <p className="text-[11px] text-gray-500">
                Per-residue values (not node+edge mixed). Color code: red=A-like, blue=B-like, green=similar to both, purple=far from both.
              </p>
              <Plot
                data={[
                  {
                    type: 'scattergl',
                    mode: 'markers',
                    x: nodeMatrix.x,
                    y: nodeMatrix.y,
                    customdata: nodeMatrix.custom,
                    marker: {
                      symbol: 'square',
                      size: 14,
                      color: nodeMatrix.color,
                      line: { width: 0 },
                    },
                    hovertemplate:
                      'sample=%{y}<br>res=%{x}<br>JS(A)=%{customdata[0]:.3f}<br>JS(B)=%{customdata[1]:.3f}<br>class=%{customdata[2]}<extra></extra>',
                  },
                ]}
                layout={{
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 140, r: 20, t: 10, b: 90 },
                  height: 420,
                  xaxis: { tickangle: -45, automargin: true, type: 'category' },
                  yaxis: { automargin: true, type: 'category' },
                }}
                config={{ responsive: true, displaylogo: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {edgeMatrix && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
              <h2 className="text-sm font-semibold text-gray-200">Per-Edge JS A/B/Other</h2>
              <p className="text-[11px] text-gray-500">Per-edge values on selected top edges.</p>
              <Plot
                data={[
                  {
                    type: 'scattergl',
                    mode: 'markers',
                    x: edgeMatrix.x,
                    y: edgeMatrix.y,
                    customdata: edgeMatrix.custom,
                    marker: {
                      symbol: 'square',
                      size: 12,
                      color: edgeMatrix.color,
                      line: { width: 0 },
                    },
                    hovertemplate:
                      'sample=%{y}<br>edge=%{x}<br>JS(A)=%{customdata[0]:.3f}<br>JS(B)=%{customdata[1]:.3f}<br>class=%{customdata[2]}<extra></extra>',
                  },
                ]}
                layout={{
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 140, r: 20, t: 10, b: 120 },
                  height: 420,
                  xaxis: { tickangle: -45, automargin: true, type: 'category' },
                  yaxis: { automargin: true, type: 'category' },
                }}
                config={{ responsive: true, displaylogo: false }}
                style={{ width: '100%' }}
              />
            </div>
          )}

          {blendedNodeMatrix && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
              <h2 className="text-sm font-semibold text-gray-200">Per-Residue JS A/B/Other (Edge-Weighted Blend)</h2>
              <p className="text-[11px] text-gray-500">
                Node JS blended with incident-edge JS using smoothing strength α and edge weights from discriminative edge JS.
              </p>
              <Plot
                data={[
                  {
                    type: 'scattergl',
                    mode: 'markers',
                    x: blendedNodeMatrix.x,
                    y: blendedNodeMatrix.y,
                    customdata: blendedNodeMatrix.custom,
                    marker: {
                      symbol: 'square',
                      size: 14,
                      color: blendedNodeMatrix.color,
                      line: { width: 0 },
                    },
                    hovertemplate:
                      'sample=%{y}<br>res=%{x}<br>JS(A)=%{customdata[0]:.3f}<br>JS(B)=%{customdata[1]:.3f}<br>class=%{customdata[2]}<extra></extra>',
                  },
                ]}
                layout={{
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  margin: { l: 140, r: 20, t: 10, b: 90 },
                  height: 420,
                  xaxis: { tickangle: -45, automargin: true, type: 'category' },
                  yaxis: { automargin: true, type: 'category' },
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
