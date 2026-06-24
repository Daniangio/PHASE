import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, RefreshCw } from 'lucide-react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Script } from 'molstar/lib/mol-script/script';
import { StructureElement, StructureSelection, StructureProperties } from 'molstar/lib/mol-model/structure';
import { clearStructureOverpaint, setStructureOverpaint } from 'molstar/lib/mol-plugin-state/helpers/structure-overpaint';
import 'molstar/build/viewer/molstar.css';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import JsRangeFilterBuilder, {
  jsRulesFromPayload,
  jsRulesToRaw,
  normalizeJsRules,
  passesAnyJsFilter,
} from '../components/common/JsRangeFilterBuilder';
import FilterSetupManager from '../components/common/FilterSetupManager';
import { formatDeltaJsAnalysisDetails, formatDeltaJsAnalysisName, makeSampleNameById } from '../utils/deltaJsAnalysisLabels';
import {
  fetchClusterAnalyses,
  fetchClusterAnalysisData,
  deleteClusterAnalysis,
  fetchClusterUiSetups,
  saveClusterUiSetup,
  deleteClusterUiSetup,
  fetchPottsClusterInfo,
  fetchSystem,
} from '../api/projects';

const JS_MAX = Math.log(2);

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function clamp(x, lo, hi) {
  if (!Number.isFinite(x)) return lo;
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

function normJs(x) {
  return clamp01(Number(x) / JS_MAX);
}

function jsForDisplay(value, normalized) {
  const v = Number(value);
  if (!Number.isFinite(v)) return NaN;
  return normalized ? v / JS_MAX : v;
}

function passesFiltersByMode(dA, dB, rules, normalizedMode) {
  if (normalizedMode) return passesAnyJsFilter(Number(dA) / JS_MAX, Number(dB) / JS_MAX, rules);
  const scaledRules = (Array.isArray(rules) ? rules : []).map((r) => ({
    aMin: Number(r?.aMin) * JS_MAX,
    aMax: Number(r?.aMax) * JS_MAX,
    bMin: Number(r?.bMin) * JS_MAX,
    bMax: Number(r?.bMax) * JS_MAX,
  }));
  return passesAnyJsFilter(Number(dA), Number(dB), scaledRules);
}

function makeFilterPayload(rules) {
  const normalized = normalizeJsRules(rules);
  return {
    rules: normalized,
    rules_normalized: normalized,
    rules_raw: jsRulesToRaw(normalized, JS_MAX),
    units: {
      storage: 'normalized_js',
      raw: 'nats',
      raw_max: JS_MAX,
      normalized_max: 1,
    },
  };
}

function rgbToHex(r, g, b) {
  const to = (v) => {
    const x = Math.max(0, Math.min(255, Math.round(v)));
    return x.toString(16).padStart(2, '0');
  };
  return `#${to(r)}${to(g)}${to(b)}`;
}

function hexToInt(colorHex) {
  const s = String(colorHex || '').trim().replace('#', '');
  const v = parseInt(s, 16);
  return Number.isFinite(v) ? v : 0xffffff;
}

function parseResidueId(label) {
  if (label == null) return null;
  const m = String(label).match(/-?\d+/);
  if (!m) return null;
  const v = Number(m[0]);
  return Number.isFinite(v) ? v : null;
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

function jsABOColor(dA, dB) {
  const w = jsABOWeights(dA, dB);
  const cA = [227, 74, 51];
  const cB = [49, 130, 189];
  const cShared = [44, 162, 95];
  const cOther = [148, 103, 189];
  const r = w.A * cA[0] + w.B * cB[0] + w.shared * cShared[0] + w.other * cOther[0];
  const g = w.A * cA[1] + w.B * cB[1] + w.shared * cShared[1] + w.other * cOther[1];
  const b = w.A * cA[2] + w.B * cB[2] + w.shared * cShared[2] + w.other * cOther[2];
  return rgbToHex(r, g, b);
}

function jsABOTag(dA, dB) {
  const w = jsABOWeights(dA, dB);
  const ranked = [
    ['A-like', w.A],
    ['B-like', w.B],
    ['similar to both', w.shared],
    ['far from both', w.other],
  ].sort((a, b) => b[1] - a[1]);
  return ranked[0][0];
}


export default function DeltaJs3DPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  const baseComponentRef = useRef(null);
  const lociLabelProviderRef = useRef(null);
  const hoverRef = useRef({
    residueIdMode: 'auth',
    sampleLabel: '',
    authToInfo: new Map(),
    labelSeqToInfo: [],
  });

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [clusterInfo, setClusterInfo] = useState(null);
  const [clusterInfoError, setClusterInfoError] = useState(null);
  const [loadedStructureStateId, setLoadedStructureStateId] = useState('');

  const [analyses, setAnalyses] = useState([]);
  const [analysesError, setAnalysesError] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisDataError, setAnalysisDataError] = useState(null);
  const [analysisDataLoading, setAnalysisDataLoading] = useState(false);

  const [viewerError, setViewerError] = useState(null);
  const [viewerStatus, setViewerStatus] = useState('initializing');
  const [structureLoading, setStructureLoading] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [filterSetups, setFilterSetups] = useState([]);
  const [filterSetupsError, setFilterSetupsError] = useState(null);
  const [selectedFilterSetupId, setSelectedFilterSetupId] = useState('');
  const [newFilterSetupName, setNewFilterSetupName] = useState('');
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [deleteAnalysisBusy, setDeleteAnalysisBusy] = useState('');
  const [deleteAnalysisError, setDeleteAnalysisError] = useState(null);

  const [rowIndex, setRowIndex] = useState(0);
  const [selectedResidueIndex, setSelectedResidueIndex] = useState(-1);
  const [residueIdMode, setResidueIdMode] = useState('auth');
  const [edgeSmoothEnabled, setEdgeSmoothEnabled] = useState(false);
  const [edgeSmoothStrength, setEdgeSmoothStrength] = useState(0.75);
  const [hideSingleCluster, setHideSingleCluster] = useState(true);
  const [jsFilters, setJsFilters] = useState([{ aMin: 0, aMax: 1, bMin: 0, bMax: 1 }]);
  const [displayNormalizedJs, setDisplayNormalizedJs] = useState(true);
  const [orderJsA, setOrderJsA] = useState('lowest');
  const [orderJsB, setOrderJsB] = useState('highest');

  useEffect(() => {
    const run = async () => {
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
    run();
  }, [projectId, systemId]);

  const clusterOptions = useMemo(
    () => (system?.metastable_clusters || []).filter((run) => run.path && run.status !== 'failed'),
    [system]
  );
  const selectedCluster = useMemo(
    () => clusterOptions.find((c) => String(c.cluster_id) === String(selectedClusterId)) || null,
    [clusterOptions, selectedClusterId]
  );
  const sampleEntries = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);
  const sampleNameById = useMemo(() => makeSampleNameById(sampleEntries), [sampleEntries]);
  const stateOptions = useMemo(() => {
    const raw = system?.states;
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    if (typeof raw === 'object') return Object.values(raw);
    return [];
  }, [system]);
  const loadedStateResidShift = useMemo(() => {
    if (!loadedStructureStateId) return 0;
    const st =
      stateOptions.find((s) => String(s?.state_id || '') === String(loadedStructureStateId)) ||
      stateOptions.find((s) => String(s?.id || '') === String(loadedStructureStateId)) ||
      null;
    const raw = Number(st?.resid_shift);
    return Number.isFinite(raw) ? Math.trunc(raw) : 0;
  }, [stateOptions, loadedStructureStateId]);

  const residueLabels = useMemo(() => {
    const keys = clusterInfo?.residue_keys || [];
    if (Array.isArray(keys) && keys.length) return keys;
    const n = clusterInfo?.n_residues || 0;
    return Array.from({ length: n }, (_, i) => `res_${i}`);
  }, [clusterInfo]);

  const singleClusterByResidue = useMemo(() => {
    const n = residueLabels.length;
    const out = new Array(n).fill(false);
    const source = Array.isArray(analysisData?.data?.K_list)
      ? analysisData.data.K_list
      : Array.isArray(clusterInfo?.cluster_counts)
      ? clusterInfo.cluster_counts
      : [];
    const m = Math.min(n, source.length);
    for (let i = 0; i < m; i += 1) {
      const ki = Number(source[i]);
      if (Number.isFinite(ki) && ki <= 1) out[i] = true;
    }
    return out;
  }, [analysisData, clusterInfo, residueLabels.length]);

  const matchingAnalyses = useMemo(() => {
    if (!analyses.length) return [];
    const sorted = [...analyses].sort((x, y) => {
      const nx = Number(x?.summary?.n_samples || 0);
      const ny = Number(y?.summary?.n_samples || 0);
      if (ny !== nx) return ny - nx;
      const tx = Date.parse(String(x?.updated_at || x?.created_at || ''));
      const ty = Date.parse(String(y?.updated_at || y?.created_at || ''));
      return (Number.isFinite(ty) ? ty : 0) - (Number.isFinite(tx) ? tx : 0);
    });
    return sorted;
  }, [analyses]);

  useEffect(() => {
    if (!matchingAnalyses.length) {
      setSelectedAnalysisId('');
      return;
    }
    const has = matchingAnalyses.some((a) => String(a.analysis_id) === String(selectedAnalysisId));
    if (!has) setSelectedAnalysisId(String(matchingAnalyses[0].analysis_id));
  }, [matchingAnalyses, selectedAnalysisId]);

  const selectedMeta = useMemo(() => {
    if (!matchingAnalyses.length) return null;
    const picked = matchingAnalyses.find((a) => String(a.analysis_id) === String(selectedAnalysisId));
    return picked || matchingAnalyses[0] || null;
  }, [matchingAnalyses, selectedAnalysisId]);

  const loadClusterInfo = useCallback(async () => {
    if (!selectedClusterId) return;
    setClusterInfoError(null);
    try {
      setClusterInfo(await fetchPottsClusterInfo(projectId, systemId, selectedClusterId));
    } catch (err) {
      setClusterInfoError(err.message || 'Failed to load cluster info.');
      setClusterInfo(null);
    }
  }, [projectId, systemId, selectedClusterId]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_js' });
      setAnalyses(Array.isArray(data?.analyses) ? data.analyses : []);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
    }
  }, [projectId, systemId, selectedClusterId]);


  const handleDeleteAnalysis = useCallback(async (analysisId) => {
    const aid = String(analysisId || '').trim();
    if (!selectedClusterId || !aid) return;
    if (!window.confirm('Delete this Delta JS analysis? This removes the stored analysis folder.')) return;
    setDeleteAnalysisError(null);
    setDeleteAnalysisBusy(aid);
    try {
      await deleteClusterAnalysis(projectId, systemId, selectedClusterId, 'delta_js', aid);
      if (String(selectedAnalysisId) === aid) {
        setSelectedAnalysisId('');
      }
      setAnalysisData(null);
      await loadAnalyses();
    } catch (err) {
      setDeleteAnalysisError(err.message || 'Failed to delete Delta JS analysis.');
    } finally {
      setDeleteAnalysisBusy('');
    }
  }, [projectId, systemId, selectedClusterId, selectedAnalysisId, loadAnalyses]);

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
      setSelectedFilterSetupId((prev) =>
        arr.some((x) => String(x?.setup_id) === String(prev)) ? prev : (arr.length ? String(arr[0].setup_id) : '')
      );
    } catch (err) {
      setFilterSetups([]);
      setFilterSetupsError(err.message || 'Failed to load filter setups.');
    }
  }, [projectId, systemId, selectedClusterId]);

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
    if (!selectedClusterId) return;
    setAnalysisData(null);
    loadClusterInfo();
    loadAnalyses();
    loadFilterSetups();
  }, [selectedClusterId, loadClusterInfo, loadAnalyses, loadFilterSetups]);

  useEffect(() => {
    const run = async () => {
      setAnalysisDataError(null);
      setAnalysisData(null);
      if (!selectedMeta?.analysis_id) return;
      setAnalysisDataLoading(true);
      try {
        setAnalysisData(
          await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'delta_js', selectedMeta.analysis_id)
        );
      } catch (err) {
        setAnalysisDataError(err.message || 'Failed to load analysis.');
      } finally {
        setAnalysisDataLoading(false);
      }
    };
    run();
  }, [projectId, systemId, selectedClusterId, selectedMeta]);

  const sampleIds = useMemo(
    () => (Array.isArray(analysisData?.data?.sample_ids) ? analysisData.data.sample_ids.map(String) : []),
    [analysisData]
  );
  const sampleLabels = useMemo(() => {
    const labels = Array.isArray(analysisData?.data?.sample_labels) ? analysisData.data.sample_labels.map(String) : [];
    return sampleIds.map((sid, i) => labels[i] || sid);
  }, [analysisData, sampleIds]);

  useEffect(() => {
    setRowIndex((prev) => {
      const i = Number(prev);
      if (Number.isInteger(i) && i >= 0 && i < sampleIds.length) return i;
      return 0;
    });
  }, [sampleIds.length]);

  useEffect(() => {
    setSelectedResidueIndex((prev) => {
      const i = Number(prev);
      if (Number.isInteger(i) && i >= 0 && i < residueLabels.length) return i;
      return -1;
    });
  }, [residueLabels.length]);

  const jsNodeA = useMemo(
    () => (Array.isArray(analysisData?.data?.js_node_a) ? analysisData.data.js_node_a : null),
    [analysisData]
  );
  const jsNodeB = useMemo(
    () => (Array.isArray(analysisData?.data?.js_node_b) ? analysisData.data.js_node_b : null),
    [analysisData]
  );
  const jsEdgeA = useMemo(
    () => (Array.isArray(analysisData?.data?.js_edge_a) ? analysisData.data.js_edge_a : null),
    [analysisData]
  );
  const jsEdgeB = useMemo(
    () => (Array.isArray(analysisData?.data?.js_edge_b) ? analysisData.data.js_edge_b : null),
    [analysisData]
  );
  const topEdgeIndices = useMemo(
    () => (Array.isArray(analysisData?.data?.top_edge_indices) ? analysisData.data.top_edge_indices : []),
    [analysisData]
  );
  const edgesAll = useMemo(
    () => (Array.isArray(analysisData?.data?.edges) ? analysisData.data.edges : []),
    [analysisData]
  );
  const DEdge = useMemo(
    () => (Array.isArray(analysisData?.data?.D_edge) ? analysisData.data.D_edge : []),
    [analysisData]
  );

  const rowDistances = useMemo(() => {
    const N = residueLabels.length;
    const out = [];
    if (!jsNodeA || !jsNodeB || !Array.isArray(jsNodeA[rowIndex]) || !Array.isArray(jsNodeB[rowIndex])) return out;
    const nodeA = jsNodeA[rowIndex].map((v) => Number(v));
    const nodeB = jsNodeB[rowIndex].map((v) => Number(v));
    const dA = nodeA.slice(0, N);
    const dB = nodeB.slice(0, N);

    if (!edgeSmoothEnabled || !Array.isArray(jsEdgeA?.[rowIndex]) || !Array.isArray(jsEdgeB?.[rowIndex])) {
      for (let i = 0; i < N; i += 1) out.push([dA[i], dB[i]]);
      return out;
    }
    const strength = clamp(Number(edgeSmoothStrength), 0, 1);
    if (strength <= 0) {
      for (let i = 0; i < N; i += 1) out.push([dA[i], dB[i]]);
      return out;
    }
    const sumW = new Array(N).fill(0);
    const sumWA = new Array(N).fill(0);
    const sumWB = new Array(N).fill(0);
    for (let col = 0; col < topEdgeIndices.length; col += 1) {
      const eidx = Number(topEdgeIndices[col]);
      const e = edgesAll[eidx];
      if (!Array.isArray(e) || e.length < 2) continue;
      const r = Number(e[0]);
      const s = Number(e[1]);
      if (!Number.isInteger(r) || !Number.isInteger(s) || r < 0 || s < 0 || r >= N || s >= N) continue;
      const dAe = Number(jsEdgeA[rowIndex][col]);
      const dBe = Number(jsEdgeB[rowIndex][col]);
      if (!Number.isFinite(dAe) || !Number.isFinite(dBe)) continue;
      const wRaw = Number.isFinite(Number(DEdge[eidx])) ? Math.abs(Number(DEdge[eidx])) : 1.0;
      const w = wRaw > 1e-12 ? wRaw : 1.0;
      sumW[r] += w;
      sumWA[r] += w * dAe;
      sumWB[r] += w * dBe;
      sumW[s] += w;
      sumWA[s] += w * dAe;
      sumWB[s] += w * dBe;
    }
    for (let i = 0; i < N; i += 1) {
      const ea = sumW[i] > 0 ? sumWA[i] / sumW[i] : dA[i];
      const eb = sumW[i] > 0 ? sumWB[i] / sumW[i] : dB[i];
      out.push([(1 - strength) * dA[i] + strength * ea, (1 - strength) * dB[i] + strength * eb]);
    }
    return out;
  }, [jsNodeA, jsNodeB, jsEdgeA, jsEdgeB, rowIndex, residueLabels.length, edgeSmoothEnabled, edgeSmoothStrength, topEdgeIndices, edgesAll, DEdge]);

  const residueIndexByAuth = useMemo(() => {
    const map = new Map();
    for (let i = 0; i < residueLabels.length; i += 1) {
      const canonicalAuth = parseResidueId(residueLabels[i]);
      if (!Number.isFinite(canonicalAuth)) continue;
      const pdbAuth = Number(canonicalAuth) - Number(loadedStateResidShift || 0);
      if (Number.isFinite(pdbAuth)) map.set(Number(pdbAuth), i);
    }
    return map;
  }, [residueLabels, loadedStateResidShift]);

  useEffect(() => {
    const authToInfo = new Map();
    const labelSeqToInfo = [];
    const n = residueLabels.length;
    for (let i = 0; i < n; i += 1) {
      const pair = Array.isArray(rowDistances[i]) ? rowDistances[i] : [];
      const dA = Number(pair[0]);
      const dB = Number(pair[1]);
      const info = {
        dA,
        dB,
        dADisplay: jsForDisplay(dA, displayNormalizedJs),
        dBDisplay: jsForDisplay(dB, displayNormalizedJs),
        tag: jsABOTag(dA, dB),
      };
      labelSeqToInfo[i] = info;
      const canonicalAuth = parseResidueId(residueLabels[i]);
      if (!Number.isFinite(canonicalAuth)) continue;
      const pdbAuth = Number(canonicalAuth) - Number(loadedStateResidShift || 0);
      if (Number.isFinite(pdbAuth)) authToInfo.set(Number(pdbAuth), info);
    }
    hoverRef.current = {
      residueIdMode,
      sampleLabel: sampleLabels?.[rowIndex] ? String(sampleLabels[rowIndex]) : '',
      authToInfo,
      labelSeqToInfo,
    };
  }, [rowDistances, residueLabels, residueIdMode, sampleLabels, rowIndex, loadedStateResidShift, displayNormalizedJs]);

  useEffect(() => {
    const plugin = pluginRef.current;
    if (viewerStatus !== 'ready' || !plugin) return;
    if (lociLabelProviderRef.current) return;

    const provider = {
      priority: 200,
      label: (loci) => {
        if (!StructureElement.Loci.is(loci)) return undefined;
        const loc = StructureElement.Loci.getFirstLocation(loci);
        if (!loc) return undefined;

        const h = hoverRef.current;
        const auth = Number(StructureProperties.residue.auth_seq_id(loc));
        const labelSeq = Number(StructureProperties.residue.label_seq_id(loc)); // 1-based
        let info = null;
        if (h.residueIdMode === 'auth') {
          info = h.authToInfo.get(auth);
        } else {
          const idx = Number.isFinite(labelSeq) ? Math.floor(labelSeq) - 1 : -1;
          if (idx >= 0 && idx < h.labelSeqToInfo.length) info = h.labelSeqToInfo[idx];
        }
        if (!info) return undefined;
        const dA = Number(info.dA);
        const dB = Number(info.dB);
        if (!Number.isFinite(dA) || !Number.isFinite(dB)) return 'JS(A/B): n/a';
        const sample = h.sampleLabel ? ` · ${h.sampleLabel}` : '';
        return `JS(A): ${Number(info.dADisplay).toFixed(3)} · JS(B): ${Number(info.dBDisplay).toFixed(3)} · ${info.tag}${sample}`;
      },
      group: (label) => `phase-delta-js:${label}`,
    };

    plugin.managers.lociLabels.addProvider(provider);
    lociLabelProviderRef.current = provider;
    return () => {
      try {
        plugin.managers.lociLabels.removeProvider(provider);
      } catch {
        // no-op
      }
      if (lociLabelProviderRef.current === provider) lociLabelProviderRef.current = null;
    };
  }, [viewerStatus]);

  useEffect(() => {
    const plugin = pluginRef.current;
    if (viewerStatus !== 'ready' || !plugin) return undefined;
    const sub = plugin.behaviors.interaction.click.subscribe((evt) => {
      const loci = evt?.current?.loci;
      if (!StructureElement.Loci.is(loci)) return;
      const loc = StructureElement.Loci.getFirstLocation(loci);
      if (!loc) return;
      const auth = Number(StructureProperties.residue.auth_seq_id(loc));
      const labelSeq = Number(StructureProperties.residue.label_seq_id(loc)); // 1-based
      let idx = -1;
      if (residueIdMode === 'auth') {
        if (Number.isFinite(auth) && residueIndexByAuth.has(auth)) idx = Number(residueIndexByAuth.get(auth));
        else if (Number.isFinite(labelSeq)) idx = Math.floor(labelSeq) - 1;
      } else {
        if (Number.isFinite(labelSeq)) idx = Math.floor(labelSeq) - 1;
        else if (Number.isFinite(auth) && residueIndexByAuth.has(auth)) idx = Number(residueIndexByAuth.get(auth));
      }
      if (Number.isInteger(idx) && idx >= 0 && idx < residueLabels.length) setSelectedResidueIndex(idx);
    });
    return () => {
      try {
        sub?.unsubscribe?.();
      } catch {
        // no-op
      }
    };
  }, [viewerStatus, residueIdMode, residueIndexByAuth, residueLabels.length]);

  const selectedResidueInfo = useMemo(() => {
    const idx = Number(selectedResidueIndex);
    if (!Number.isInteger(idx) || idx < 0 || idx >= residueLabels.length) return null;
    const pair = Array.isArray(rowDistances[idx]) ? rowDistances[idx] : [];
    const dA = Number(pair[0]);
    const dB = Number(pair[1]);
    if (!Number.isFinite(dA) || !Number.isFinite(dB)) return null;
    const inRange = passesFiltersByMode(dA, dB, jsFilters, displayNormalizedJs);
    const tag = jsABOTag(dA, dB);
    return {
      residueIndex: idx,
      residueLabel: residueLabels[idx] || String(idx),
      dA,
      dB,
      dADisplay: jsForDisplay(dA, displayNormalizedJs),
      dBDisplay: jsForDisplay(dB, displayNormalizedJs),
      inRange,
      tag,
      color: inRange ? jsABOColor(dA, dB) : '#9ca3af',
    };
  }, [selectedResidueIndex, residueLabels, rowDistances, jsFilters, displayNormalizedJs]);

  const rankedFilteredResidues = useMemo(() => {
    const out = [];
    const wantALow = orderJsA === 'lowest';
    const wantBLow = orderJsB === 'lowest';
    for (let i = 0; i < residueLabels.length && i < rowDistances.length; i += 1) {
      const pair = Array.isArray(rowDistances[i]) ? rowDistances[i] : [];
      const dAraw = Number(pair[0]);
      const dBraw = Number(pair[1]);
      if (!Number.isFinite(dAraw) || !Number.isFinite(dBraw)) continue;
      if (!passesFiltersByMode(dAraw, dBraw, jsFilters, displayNormalizedJs)) continue;
      if (hideSingleCluster && singleClusterByResidue[i]) continue;
      const dA = normJs(dAraw);
      const dB = normJs(dBraw);
      const termA = wantALow ? dA : 1 - dA;
      const termB = wantBLow ? dB : 1 - dB;
      const score = termA + termB;
      out.push({
        residueIndex: i,
        residueLabel: residueLabels[i] || String(i),
        jsA: jsForDisplay(dAraw, displayNormalizedJs),
        jsB: jsForDisplay(dBraw, displayNormalizedJs),
        score,
      });
    }
    out.sort((a, b) => a.score - b.score);
    return out;
  }, [rowDistances, residueLabels, jsFilters, hideSingleCluster, singleClusterByResidue, orderJsA, orderJsB, displayNormalizedJs]);

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
        payload: makeFilterPayload(jsFilters),
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
    const rules = jsRulesFromPayload(entry?.payload, JS_MAX);
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

  const coloringPayload = useMemo(() => {
    if (!rowDistances.length) return null;
    const residueIdsAuth = [];
    const residueIdsLabel = [];
    const colorsAuth = [];
    const colorsLabel = [];
    for (let i = 0; i < residueLabels.length && i < rowDistances.length; i += 1) {
      const [dA, dB] = rowDistances[i];
      const label = residueLabels[i];
      const canonicalAuth = parseResidueId(label);
      const auth = Number.isFinite(canonicalAuth)
        ? Number(canonicalAuth) - Number(loadedStateResidShift || 0)
        : null;
      const inRange = passesFiltersByMode(dA, dB, jsFilters, displayNormalizedJs);
      const color =
        !inRange || (hideSingleCluster && singleClusterByResidue[i]) ? '#9ca3af' : jsABOColor(dA, dB);
      if (Number.isFinite(auth)) {
        residueIdsAuth.push(auth);
        colorsAuth.push(color);
      }
      residueIdsLabel.push(i + 1);
      colorsLabel.push(color);
    }
    return { residueIdsAuth, colorsAuth, residueIdsLabel, colorsLabel };
  }, [rowDistances, residueLabels, hideSingleCluster, singleClusterByResidue, jsFilters, loadedStateResidShift, displayNormalizedJs]);

  const getBaseComponentWrapper = useCallback(() => {
    const plugin = pluginRef.current;
    const baseRef = baseComponentRef.current;
    if (!plugin || !baseRef) return null;
    const root = plugin.managers.structure.hierarchy.current.structures[0];
    const comps = root?.components;
    if (!Array.isArray(comps)) return null;
    const found = comps.find((c) => c?.cell?.transform?.ref === baseRef) || null;
    if (!found || !Array.isArray(found.representations)) return null;
    return found;
  }, []);

  const clearOverpaint = useCallback(async () => {
    const plugin = pluginRef.current;
    const base = getBaseComponentWrapper();
    if (!plugin || !base) return;
    try {
      await clearStructureOverpaint(plugin, [base], ['cartoon']);
    } catch (err) {
      // no-op
    }
  }, [getBaseComponentWrapper]);

  const ensureBaseComponent = useCallback(async () => {
    const plugin = pluginRef.current;
    if (!plugin) return null;
    const structureCell = plugin.managers.structure.hierarchy.current.structures[0]?.cell;
    if (!structureCell) return null;
    try {
      const roots = plugin.managers.structure.hierarchy.current.structures;
      if (roots?.length) await plugin.managers.structure.component.clear(roots);
    } catch {
      // no-op
    }
    const allExpr = MS.struct.generator.all();
    const baseComponent = await plugin.builders.structure.tryCreateComponentFromExpression(structureCell, allExpr, 'phase-js-base');
    if (!baseComponent) return null;
    await plugin.builders.structure.representation.addRepresentation(baseComponent, {
      type: 'cartoon',
      color: 'uniform',
      colorParams: { value: hexToInt('#9ca3af') },
      transparency: { name: 'uniform', params: { value: 0.0 } },
    });
    baseComponentRef.current = baseComponent.ref;
    return getBaseComponentWrapper();
  }, [getBaseComponentWrapper]);

  const loadStructure = useCallback(async (stateIdOverride) => {
    const plugin = pluginRef.current;
    if (!plugin) return;
    setStructureLoading(true);
    setViewerError(null);
    try {
      baseComponentRef.current = null;
      await plugin.clear();
      await plugin.dataTransaction(async () => {
        const sid = String(stateIdOverride || loadedStructureStateId || '').trim();
        if (!sid) throw new Error('Select a structure to load.');
        const url = `/api/v1/projects/${projectId}/systems/${systemId}/structures/${encodeURIComponent(sid)}`;
        const data = await plugin.builders.data.download({ url: Asset.Url(url), label: sid }, { state: { isGhost: true } });
        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
      });
      await ensureBaseComponent();
      await clearOverpaint();
    } catch (err) {
      setViewerError(err.message || 'Failed to load structure.');
    } finally {
      setStructureLoading(false);
    }
  }, [projectId, systemId, loadedStructureStateId, ensureBaseComponent, clearOverpaint]);

  useEffect(() => {
    let disposed = false;
    let rafId;
    const init = async () => {
      if (disposed) return;
      if (!containerRef.current) {
        rafId = requestAnimationFrame(init);
        return;
      }
      if (pluginRef.current) return;
      setViewerStatus('initializing');
      setViewerError(null);
      try {
        const plugin = await createPluginUI({ target: containerRef.current, render: renderReact18 });
        if (disposed) {
          plugin.dispose?.();
          return;
        }
        pluginRef.current = plugin;
        setViewerStatus('ready');
      } catch {
        setViewerStatus('error');
        setViewerError('3D viewer initialization failed.');
      }
    };
    init();
    return () => {
      disposed = true;
      if (rafId) cancelAnimationFrame(rafId);
      if (pluginRef.current) {
        try {
          pluginRef.current.dispose?.();
        } catch {
          // no-op
        }
      }
      pluginRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (viewerStatus !== 'ready') return;
    if (!stateOptions.length) return;
    if (!loadedStructureStateId) {
      const sid = stateOptions[0]?.state_id;
      if (sid) setLoadedStructureStateId(sid);
    }
  }, [viewerStatus, stateOptions, loadedStructureStateId]);

  useEffect(() => {
    if (viewerStatus !== 'ready') return;
    if (!loadedStructureStateId) return;
    loadStructure(loadedStructureStateId);
  }, [viewerStatus, loadedStructureStateId, loadStructure]);

  const applyColoring = useCallback(async () => {
    const plugin = pluginRef.current;
    const base = getBaseComponentWrapper();
    if (!plugin || !base || !coloringPayload) return;
    await clearOverpaint();

    const useAuth = residueIdMode === 'auth';
    const residueIds = useAuth ? coloringPayload.residueIdsAuth : coloringPayload.residueIdsLabel;
    const colors = useAuth ? coloringPayload.colorsAuth : coloringPayload.colorsLabel;
    if (!residueIds.length || residueIds.length !== colors.length) return;

    const groups = new Map();
    for (let i = 0; i < residueIds.length; i += 1) {
      const rid = Number(residueIds[i]);
      const c = String(colors[i] || '').toLowerCase();
      if (!Number.isFinite(rid) || !c) continue;
      if (!groups.has(c)) groups.set(c, []);
      groups.get(c).push(rid);
    }

    const propFn = useAuth
      ? MS.struct.atomProperty.macromolecular.auth_seq_id()
      : MS.struct.atomProperty.macromolecular.label_seq_id();
    const dataRoot = plugin.managers.structure.hierarchy.current.structures[0]?.cell?.obj?.data;
    if (!dataRoot) return;

    for (const [hex, ids] of groups.entries()) {
      if (!ids.length) continue;
      const residueTests =
        ids.length === 1
          ? MS.core.rel.eq([propFn, ids[0]])
          : MS.core.set.has([MS.set(...ids), propFn]);
      const expression = MS.struct.generator.atomGroups({ 'residue-test': residueTests });
      const lociGetter = () => {
        const query = Script.getStructureSelection(expression, dataRoot);
        return StructureSelection.toLociWithSourceUnits(query);
      };
      try {
        // eslint-disable-next-line no-await-in-loop
        await setStructureOverpaint(plugin, [base], hexToInt(hex), lociGetter, ['cartoon']);
      } catch {
        // no-op
      }
    }
  }, [coloringPayload, residueIdMode, getBaseComponentWrapper, clearOverpaint]);

  useEffect(() => {
    applyColoring();
  }, [applyColoring]);

  if (loadingSystem) return <Loader message="Loading Delta JS 3D..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Delta JS (3D): Help"
        docPath="/docs/delta_js_help.md"
        onClose={() => setHelpOpen(false)}
      />
      <div className="flex items-start justify-between gap-3">
        <div>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_js`)}
            className="text-cyan-400 hover:text-cyan-300 text-sm"
          >
            ← Back to Delta JS Evaluation
          </button>
          <h1 className="text-2xl font-semibold text-white">Delta JS (A/B/Other) 3D</h1>
          <p className="text-sm text-gray-400">
            Residue coloring from JS distance to A and B references: red=A-like, blue=B-like, green=similar to both, purple=far from both.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_js_table${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Multi-traj table
          </button>
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
          >
            <CircleHelp className="h-4 w-4" />
            Help
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-4">
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
              <label className="block text-xs text-gray-400 mb-1">Cluster</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {clusterOptions.map((c) => (
                  <option key={c.cluster_id} value={c.cluster_id}>
                    {c.name || c.cluster_id}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Analysis run</label>
              {!matchingAnalyses.length && (
                <div className="rounded-md border border-gray-800 bg-gray-950/40 px-2 py-2 text-sm text-gray-500">
                  No analyses in this cluster.
                </div>
              )}
              {!!matchingAnalyses.length && (
                <div className="max-h-52 overflow-auto rounded-md border border-gray-800 bg-gray-950/30">
                  {matchingAnalyses.map((a) => {
                    const aid = String(a.analysis_id);
                    const active = aid === String(selectedAnalysisId);
                    return (
                      <div
                        key={aid}
                        role="button"
                        tabIndex={0}
                        onClick={() => setSelectedAnalysisId(aid)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') setSelectedAnalysisId(aid);
                        }}
                        className={`w-full text-left px-3 py-2 border-b border-gray-900 hover:bg-gray-800/40 cursor-pointer ${
                          active ? 'bg-cyan-950/40 border-l-2 border-l-cyan-500' : ''
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="text-xs text-gray-100">{formatDeltaJsAnalysisName(a, sampleNameById)}</div>
                            <div className="text-[11px] text-gray-400">{formatDeltaJsAnalysisDetails(a)}</div>
                            <div className="text-[10px] text-gray-600">{aid}</div>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteAnalysis(aid);
                            }}
                            disabled={deleteAnalysisBusy === aid}
                            className="shrink-0 text-[11px] px-2 py-1 rounded border border-red-900/60 text-red-300 hover:bg-red-950/40 disabled:opacity-50"
                            title="Delete Delta JS analysis"
                          >
                            {deleteAnalysisBusy === aid ? 'Deleting...' : 'Delete'}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            {deleteAnalysisError && <ErrorMessage message={deleteAnalysisError} />}
            <div>
              <label className="block text-xs text-gray-400 mb-1">Color by sample</label>
              <select
                value={String(rowIndex)}
                onChange={(e) => setRowIndex(Number(e.target.value))}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {sampleLabels.map((label, idx) => (
                  <option key={`sid:${sampleIds[idx] || idx}`} value={String(idx)}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Load structure (PDB)</label>
              <div className="flex flex-wrap gap-2">
                {stateOptions.map((st) => {
                  const sid = st?.state_id || st?.id || '';
                  if (!sid) return null;
                  const active = sid === loadedStructureStateId;
                  return (
                    <button
                      key={sid}
                      type="button"
                      onClick={() => setLoadedStructureStateId(sid)}
                      className={`text-xs px-3 py-2 rounded-md border ${active ? 'border-cyan-500 text-cyan-200' : 'border-gray-700 text-gray-200 hover:border-gray-500'}`}
                    >
                      {st?.name || sid}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Residue mapping</label>
                <select
                  value={residueIdMode}
                  onChange={(e) => setResidueIdMode(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  <option value="auth">PDB numbering (auth_seq_id)</option>
                  <option value="label">Sequential (label_seq_id)</option>
                </select>
              </div>
              <div className="flex items-end">
                <label className="flex items-center gap-2 text-sm text-gray-200">
                  <input
                    type="checkbox"
                    checked={hideSingleCluster}
                    onChange={(e) => setHideSingleCluster(e.target.checked)}
                    className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                  />
                  Gray single-cluster residues
                </label>
              </div>
            </div>
            <JsRangeFilterBuilder
              rules={jsFilters}
              onChange={setJsFilters}
              displayMax={displayNormalizedJs ? 1 : JS_MAX}
              unitLabel={displayNormalizedJs ? 'normalized JS [0,1]' : 'raw JS [0, ln(2)]'}
            />
            <div className="rounded-md border border-gray-800 bg-gray-950/30 p-2 space-y-2">
              <label className="flex items-center gap-2 text-sm text-gray-200">
                <input
                  type="checkbox"
                  checked={displayNormalizedJs}
                  onChange={(e) => setDisplayNormalizedJs(e.target.checked)}
                  className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                />
                Show normalized JS in [0,1] (default)
              </label>
              <p className="text-[11px] text-gray-500">
                Filters are saved as normalized JS and mirrored as raw JS. Toggle off to view/edit the same ranges in raw nats [0, ln(2)].
              </p>
            </div>
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
              <label className="flex items-center gap-2 text-sm text-gray-200">
                <input
                  type="checkbox"
                  checked={edgeSmoothEnabled}
                  onChange={(e) => setEdgeSmoothEnabled(e.target.checked)}
                  className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                />
                Edge-weighted node blending
              </label>
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
            </div>
            <div className="rounded-md border border-gray-800 bg-gray-950/30 p-2 space-y-2">
              <div className="text-xs text-gray-300 font-medium">Residue ranking rule</div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">JS(A)</label>
                  <select
                    value={orderJsA}
                    onChange={(e) => setOrderJsA(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                  >
                    <option value="lowest">lowest</option>
                    <option value="highest">highest</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">JS(B)</label>
                  <select
                    value={orderJsB}
                    onChange={(e) => setOrderJsB(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                  >
                    <option value="lowest">lowest</option>
                    <option value="highest">highest</option>
                  </select>
                </div>
              </div>
              <div className="text-[11px] text-gray-500">
                Score = term(A) + term(B), where lowest uses JS and highest uses (1 - JS), on normalized JS.
              </div>
            </div>
            <div className="rounded-md border border-gray-800 bg-gray-950/30 p-2 space-y-1">
              <div className="text-xs text-gray-400">Selected residue (click on 3D)</div>
              {selectedResidueInfo ? (
                <>
                  <div className="text-xs text-gray-200">
                    {selectedResidueInfo.residueLabel}
                    <span className="ml-2 inline-block align-middle w-3 h-3 rounded-sm border border-gray-700" style={{ backgroundColor: selectedResidueInfo.color }} />
                  </div>
                  <div className="text-[11px] text-gray-300">JS(A): {selectedResidueInfo.dADisplay.toFixed(3)}</div>
                  <div className="text-[11px] text-gray-300">JS(B): {selectedResidueInfo.dBDisplay.toFixed(3)}</div>
                  <div className="text-[11px] text-gray-400">Tag: {selectedResidueInfo.tag}</div>
                  {!selectedResidueInfo.inRange && <div className="text-[11px] text-gray-500">Filtered out (gray).</div>}
                </>
              ) : (
                <div className="text-[11px] text-gray-500">No residue selected.</div>
              )}
            </div>
            {clusterInfoError && <ErrorMessage message={clusterInfoError} />}
            {analysesError && <ErrorMessage message={analysesError} />}
            {analysisDataError && <ErrorMessage message={analysisDataError} />}
            {!selectedMeta && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200">
                No delta_js analysis found in this cluster.
              </div>
            )}
          </div>
        </aside>

        <main className="space-y-3">
          {(viewerError || viewerStatus === 'error') && <ErrorMessage message={viewerError || 'Viewer error'} />}
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-200">3D Viewer</h2>
              <button
                type="button"
                onClick={() => loadStructure()}
                className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Reload structure
              </button>
            </div>
            <div className="mt-3 h-[70vh] min-h-[520px] rounded-md border border-gray-800 bg-black/20 overflow-hidden relative">
              {viewerStatus !== 'ready' && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/60">
                  <Loader message="Initializing viewer..." />
                </div>
              )}
              {viewerStatus === 'ready' && structureLoading && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50">
                  <Loader message="Loading structure..." />
                </div>
              )}
              <div ref={containerRef} className="w-full h-full relative" />
            </div>
            {analysisDataLoading && <p className="mt-2 text-sm text-gray-400">Loading analysis…</p>}
          </div>
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-200">Filtered Residues Ranking</h2>
              <div className="text-xs text-gray-400">{rankedFilteredResidues.length} residues</div>
            </div>
            <div className="mt-2 text-[11px] text-gray-500">
              Ordered by current rule and current filters. Lower score first.
            </div>
            <div className="mt-3 max-h-72 overflow-auto rounded-md border border-gray-800">
              <table className="w-full text-xs">
                <thead className="bg-gray-950/70 sticky top-0">
                  <tr className="text-gray-300">
                    <th className="px-2 py-2 text-left">Residue</th>
                    <th className="px-2 py-2 text-right">JS A</th>
                    <th className="px-2 py-2 text-right">JS B</th>
                    <th className="px-2 py-2 text-right">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {!rankedFilteredResidues.length && (
                    <tr>
                      <td colSpan={4} className="px-2 py-3 text-center text-gray-500">
                        No residues pass current filters.
                      </td>
                    </tr>
                  )}
                  {rankedFilteredResidues.map((row) => (
                    <tr
                      key={`rank:${row.residueIndex}`}
                      className={`border-t border-gray-900 hover:bg-gray-800/40 cursor-pointer ${
                        row.residueIndex === selectedResidueIndex ? 'bg-cyan-950/20' : ''
                      }`}
                      onClick={() => setSelectedResidueIndex(row.residueIndex)}
                    >
                      <td className="px-2 py-1.5 text-gray-200">{row.residueLabel}</td>
                      <td className="px-2 py-1.5 text-right text-gray-300">{row.jsA.toFixed(4)}</td>
                      <td className="px-2 py-1.5 text-right text-gray-300">{row.jsB.toFixed(4)}</td>
                      <td className="px-2 py-1.5 text-right text-gray-100">{row.score.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
