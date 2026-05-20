import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { Plus, RefreshCw } from 'lucide-react';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import JsRangeFilterBuilder, {
  jsRulesFromPayload,
  jsRulesToRaw,
  normalizeJsRules,
  passesAnyJsFilter,
} from '../components/common/JsRangeFilterBuilder';
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

const JS_MAX = Math.log(2);
const groupPalette = ['#f97316', '#22d3ee', '#10b981', '#eab308', '#a78bfa', '#f43f5e'];

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function normJs(x) {
  return clamp01(Number(x) / JS_MAX);
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
  const cA = [227, 74, 51];
  const cB = [49, 130, 189];
  const cShared = [44, 162, 95];
  const cOther = [148, 103, 189];
  const r = w.A * cA[0] + w.B * cB[0] + w.shared * cShared[0] + w.other * cOther[0];
  const g = w.A * cA[1] + w.B * cB[1] + w.shared * cShared[1] + w.other * cOther[1];
  const b = w.A * cA[2] + w.B * cB[2] + w.shared * cShared[2] + w.other * cOther[2];
  return rgba(r, g, b, alpha);
}

function jsForDisplay(value, normalized) {
  const v = Number(value);
  if (!Number.isFinite(v)) return Number.NaN;
  return normalized ? (v / JS_MAX) : v;
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

function formatAnalysisShortMeta(entry) {
  const ts = String(entry?.updated_at || entry?.created_at || '').slice(0, 19) || 'n/a';
  const n = Number(entry?.summary?.n_samples || 0);
  const md = String(entry?.md_label_mode || 'assigned');
  const invalid = Boolean(entry?.drop_invalid) ? 'drop invalid' : 'keep invalid';
  return `${ts} · n=${n} · ${md} · ${invalid}`;
}

function parseGroupResidues(text) {
  return String(text || '')
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean);
}

function residueNumber(label) {
  const m = String(label || '').match(/-?\d+/);
  if (!m) return null;
  const v = Number(m[0]);
  return Number.isFinite(v) ? v : null;
}

function expandGroupSelection(inputText, residueLabels) {
  const tokens = parseGroupResidues(inputText);
  if (!tokens.length) return [];
  const labels = Array.isArray(residueLabels) ? residueLabels.map(String) : [];
  const byNumber = new Map();
  labels.forEach((label, idx) => {
    const n = residueNumber(label);
    if (n == null) return;
    if (!byNumber.has(n)) byNumber.set(n, []);
    byNumber.get(n).push(idx);
  });
  const out = new Set();
  for (const token of tokens) {
    if (labels.includes(token)) {
      out.add(token);
      continue;
    }
    const idxMatch = token.match(/^#(\d+)$/);
    if (idxMatch) {
      const i = Number(idxMatch[1]);
      if (Number.isInteger(i) && i >= 0 && i < labels.length) out.add(labels[i]);
      continue;
    }
    const idxRange = token.match(/^#(\d+)\s*-\s*#(\d+)$/);
    if (idxRange) {
      let a = Number(idxRange[1]);
      let b = Number(idxRange[2]);
      if (a > b) [a, b] = [b, a];
      for (let i = a; i <= b; i += 1) if (i >= 0 && i < labels.length) out.add(labels[i]);
      continue;
    }
    const numRange = token.match(/^(-?\d+)\s*-\s*(-?\d+)$/);
    if (numRange) {
      let a = Number(numRange[1]);
      let b = Number(numRange[2]);
      if (a > b) [a, b] = [b, a];
      for (let n = a; n <= b; n += 1) {
        const idxs = byNumber.get(n) || [];
        idxs.forEach((idx) => out.add(labels[idx]));
      }
      continue;
    }
    const paired = token.match(/^([A-Za-z_]+)?(-?\d+)\s*-\s*([A-Za-z_]+)?(-?\d+)$/);
    if (paired) {
      let a = Number(paired[2]);
      let b = Number(paired[4]);
      if (a > b) [a, b] = [b, a];
      for (let n = a; n <= b; n += 1) {
        const idxs = byNumber.get(n) || [];
        idxs.forEach((idx) => out.add(labels[idx]));
      }
    }
  }
  return [...out];
}

export default function DeltaJsTablePage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [clusterInfo, setClusterInfo] = useState(null);
  const [clusterInfoError, setClusterInfoError] = useState(null);

  const [analyses, setAnalyses] = useState([]);
  const [analysesError, setAnalysesError] = useState(null);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');

  const [data, setData] = useState(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState(null);
  const analysisDataCacheRef = useRef({});
  const analysisInFlightRef = useRef({});

  const [selectedSampleIds, setSelectedSampleIds] = useState([]);
  const [jsFilters, setJsFilters] = useState([{ aMin: 0, aMax: 1, bMin: 0, bMax: 1 }]);
  const [edgeSmoothEnabled, setEdgeSmoothEnabled] = useState(false);
  const [edgeSmoothStrength, setEdgeSmoothStrength] = useState(0.75);
  const [displayNormalizedJs, setDisplayNormalizedJs] = useState(true);

  const [filterSetups, setFilterSetups] = useState([]);
  const [filterSetupsError, setFilterSetupsError] = useState(null);
  const [selectedFilterSetupId, setSelectedFilterSetupId] = useState('');
  const [newFilterSetupName, setNewFilterSetupName] = useState('');

  const [rowGroups, setRowGroups] = useState([]);
  const [groupSetups, setGroupSetups] = useState([]);
  const [groupSetupsError, setGroupSetupsError] = useState(null);
  const [selectedGroupSetupId, setSelectedGroupSetupId] = useState('');
  const [newGroupSetupName, setNewGroupSetupName] = useState('');
  const [newGroupName, setNewGroupName] = useState('');
  const [newGroupResidues, setNewGroupResidues] = useState('');
  const [groupHelpOpen, setGroupHelpOpen] = useState(false);

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

  useEffect(() => {
    if (!clusterOptions.length) return;
    const requested = String(new URLSearchParams(location.search || '').get('cluster_id') || '').trim();
    if (requested && clusterOptions.some((c) => c.cluster_id === requested)) {
      if (selectedClusterId !== requested) setSelectedClusterId(requested);
      return;
    }
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterId, location.search]);

  const loadClusterInfo = useCallback(async () => {
    if (!selectedClusterId) return;
    setClusterInfoError(null);
    try {
      setClusterInfo(await fetchPottsClusterInfo(projectId, systemId, selectedClusterId));
    } catch (err) {
      setClusterInfo(null);
      setClusterInfoError(err.message || 'Failed to load cluster info.');
    }
  }, [projectId, systemId, selectedClusterId]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesError(null);
    try {
      const payload = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_js' });
      const list = Array.isArray(payload?.analyses) ? payload.analyses : [];
      setAnalyses(list);
      setSelectedAnalysisId((prev) => (prev && list.some((a) => String(a.analysis_id) === String(prev)) ? prev : (list[0]?.analysis_id || '')));
    } catch (err) {
      setAnalyses([]);
      setSelectedAnalysisId('');
      setAnalysesError(err.message || 'Failed to load analyses.');
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
      setSelectedFilterSetupId((prev) =>
        arr.some((x) => String(x?.setup_id) === String(prev)) ? prev : (arr[0]?.setup_id || '')
      );
    } catch (err) {
      setFilterSetups([]);
      setFilterSetupsError(err.message || 'Failed to load filter setups.');
    }
  }, [projectId, systemId, selectedClusterId]);

  const loadGroupSetups = useCallback(async () => {
    if (!selectedClusterId) return;
    setGroupSetupsError(null);
    try {
      const res = await fetchClusterUiSetups(projectId, systemId, selectedClusterId, {
        setupType: 'delta_js_residue_groups',
        page: 'delta_js_table',
      });
      const arr = Array.isArray(res?.setups) ? res.setups : [];
      setGroupSetups(arr);
      setSelectedGroupSetupId((prev) =>
        arr.some((x) => String(x?.setup_id) === String(prev)) ? prev : (arr[0]?.setup_id || '')
      );
    } catch (err) {
      setGroupSetups([]);
      setGroupSetupsError(err.message || 'Failed to load group setups.');
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    analysisDataCacheRef.current = {};
    analysisInFlightRef.current = {};
    setData(null);
    setDataError(null);
    loadClusterInfo();
    loadAnalyses();
    loadFilterSetups();
    loadGroupSetups();
  }, [selectedClusterId, loadClusterInfo, loadAnalyses, loadFilterSetups, loadGroupSetups]);

  const matchingAnalyses = useMemo(() => {
    if (!analyses.length) return [];
    return [...analyses].sort((x, y) => {
      const nx = Number(x?.summary?.n_samples || 0);
      const ny = Number(y?.summary?.n_samples || 0);
      if (ny !== nx) return ny - nx;
      const tx = Date.parse(String(x?.updated_at || x?.created_at || ''));
      const ty = Date.parse(String(y?.updated_at || y?.created_at || ''));
      return (Number.isFinite(ty) ? ty : 0) - (Number.isFinite(tx) ? tx : 0);
    });
  }, [analyses]);

  const loadAnalysisData = useCallback(async (analysisId) => {
    if (!analysisId) return null;
    const key = `delta_js_table:${analysisId}`;
    if (Object.prototype.hasOwnProperty.call(analysisDataCacheRef.current, key)) return analysisDataCacheRef.current[key];
    if (analysisInFlightRef.current[key]) return analysisInFlightRef.current[key];
    const p = fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'delta_js', analysisId)
      .then((payload) => {
        analysisDataCacheRef.current = { ...analysisDataCacheRef.current, [key]: payload };
        delete analysisInFlightRef.current[key];
        return payload;
      })
      .catch((err) => {
        delete analysisInFlightRef.current[key];
        throw err;
      });
    analysisInFlightRef.current[key] = p;
    return p;
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setDataError(null);
      setData(null);
      if (!selectedAnalysisId) return;
      setDataLoading(true);
      try {
        const payload = await loadAnalysisData(selectedAnalysisId);
        if (cancelled) return;
        setData(payload);
      } catch (err) {
        if (cancelled) return;
        setDataError(err.message || 'Failed to load analysis.');
      } finally {
        if (!cancelled) setDataLoading(false);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [selectedAnalysisId, loadAnalysisData]);

  const residueLabels = useMemo(() => {
    const display = clusterInfo?.residue_display_labels || [];
    if (Array.isArray(display) && display.length) return display;
    const keys = clusterInfo?.residue_keys || [];
    if (Array.isArray(keys) && keys.length) return keys;
    const n = Number(clusterInfo?.n_residues || 0);
    return Array.from({ length: n }, (_, i) => `res_${i}`);
  }, [clusterInfo]);

  const analysisSampleIds = useMemo(() => (Array.isArray(data?.data?.sample_ids) ? data.data.sample_ids.map(String) : []), [data]);
  const analysisSampleLabels = useMemo(() => {
    const labels = Array.isArray(data?.data?.sample_labels) ? data.data.sample_labels.map(String) : [];
    return analysisSampleIds.map((sid, idx) => labels[idx] || sid);
  }, [data, analysisSampleIds]);
  const sampleIndexById = useMemo(() => {
    const m = new Map();
    analysisSampleIds.forEach((sid, idx) => m.set(String(sid), idx));
    return m;
  }, [analysisSampleIds]);
  const jsEdgeA = useMemo(() => (Array.isArray(data?.data?.js_edge_a) ? data.data.js_edge_a : []), [data]);
  const jsEdgeB = useMemo(() => (Array.isArray(data?.data?.js_edge_b) ? data.data.js_edge_b : []), [data]);
  const topEdgeIndices = useMemo(
    () => (Array.isArray(data?.data?.top_edge_indices) ? data.data.top_edge_indices : []),
    [data]
  );
  const edgesAll = useMemo(() => (Array.isArray(data?.data?.edges) ? data.data.edges : []), [data]);
  const DEdge = useMemo(() => (Array.isArray(data?.data?.D_edge) ? data.data.D_edge : []), [data]);

  useEffect(() => {
    if (!analysisSampleIds.length) {
      setSelectedSampleIds([]);
      return;
    }
    setSelectedSampleIds((prev) => {
      const kept = prev.filter((sid) => sampleIndexById.has(String(sid)));
      if (kept.length) return kept;
      return [analysisSampleIds[0]];
    });
  }, [analysisSampleIds, sampleIndexById]);

  const tableRows = useMemo(() => {
    const nodeA = Array.isArray(data?.data?.js_node_a) ? data.data.js_node_a : [];
    const nodeB = Array.isArray(data?.data?.js_node_b) ? data.data.js_node_b : [];
    if (!nodeA.length || !nodeB.length || !selectedSampleIds.length || !residueLabels.length) return [];

    const selectedRows = selectedSampleIds
      .map((sid) => sampleIndexById.get(String(sid)))
      .filter((idx) => Number.isInteger(idx) && idx >= 0);
    if (!selectedRows.length) return [];

    const include = new Array(residueLabels.length).fill(false);
    const perTrajPass = {};
    const values = {};
    for (const sid of selectedSampleIds) {
      const idx = sampleIndexById.get(String(sid));
      if (idx == null) continue;
      perTrajPass[sid] = new Array(residueLabels.length).fill(false);
      values[sid] = new Array(residueLabels.length).fill(null);
      const useEdgeBlend =
        edgeSmoothEnabled &&
        edgeSmoothStrength > 0 &&
        Array.isArray(jsEdgeA?.[idx]) &&
        Array.isArray(jsEdgeB?.[idx]) &&
        topEdgeIndices.length > 0;
      let blendedA = null;
      let blendedB = null;
      if (useEdgeBlend) {
        const nRes = residueLabels.length;
        const dA0 = new Array(nRes).fill(Number.NaN);
        const dB0 = new Array(nRes).fill(Number.NaN);
        for (let r = 0; r < nRes; r += 1) {
          dA0[r] = Number(nodeA?.[idx]?.[r]);
          dB0[r] = Number(nodeB?.[idx]?.[r]);
        }
        const sumW = new Array(nRes).fill(0);
        const sumWA = new Array(nRes).fill(0);
        const sumWB = new Array(nRes).fill(0);
        for (let col = 0; col < topEdgeIndices.length; col += 1) {
          const eidx = Number(topEdgeIndices[col]);
          const e = edgesAll?.[eidx];
          if (!Array.isArray(e) || e.length < 2) continue;
          const r = Number(e[0]);
          const s = Number(e[1]);
          if (!Number.isInteger(r) || !Number.isInteger(s) || r < 0 || s < 0 || r >= nRes || s >= nRes) continue;
          const dAe = Number(jsEdgeA[idx][col]);
          const dBe = Number(jsEdgeB[idx][col]);
          if (!Number.isFinite(dAe) || !Number.isFinite(dBe)) continue;
          const wRaw = Number.isFinite(Number(DEdge?.[eidx])) ? Math.abs(Number(DEdge[eidx])) : 1.0;
          const w = wRaw > 1e-12 ? wRaw : 1.0;
          sumW[r] += w;
          sumWA[r] += w * dAe;
          sumWB[r] += w * dBe;
          sumW[s] += w;
          sumWA[s] += w * dAe;
          sumWB[s] += w * dBe;
        }
        blendedA = new Array(nRes).fill(Number.NaN);
        blendedB = new Array(nRes).fill(Number.NaN);
        const strength = clamp01(Number(edgeSmoothStrength));
        for (let r = 0; r < nRes; r += 1) {
          const dAVal = dA0[r];
          const dBVal = dB0[r];
          const eA = sumW[r] > 0 ? sumWA[r] / sumW[r] : dAVal;
          const eB = sumW[r] > 0 ? sumWB[r] / sumW[r] : dBVal;
          blendedA[r] = Number.isFinite(dAVal) ? ((1 - strength) * dAVal + strength * eA) : Number.NaN;
          blendedB[r] = Number.isFinite(dBVal) ? ((1 - strength) * dBVal + strength * eB) : Number.NaN;
        }
      }
      for (let r = 0; r < residueLabels.length; r += 1) {
        const dA = blendedA ? Number(blendedA[r]) : Number(nodeA?.[idx]?.[r]);
        const dB = blendedB ? Number(blendedB[r]) : Number(nodeB?.[idx]?.[r]);
        if (!Number.isFinite(dA) || !Number.isFinite(dB)) continue;
        values[sid][r] = { dA, dB };
        const pass = passesFiltersByMode(dA, dB, jsFilters, displayNormalizedJs);
        perTrajPass[sid][r] = pass;
        if (pass) include[r] = true;
      }
    }

    const rows = [];
    for (let r = 0; r < residueLabels.length; r += 1) {
      if (!include[r]) continue;
      const residue = residueLabels[r] || String(r);
      let assignedGroup = '';
      for (const grp of rowGroups) {
        const items = Array.isArray(grp?.residues) ? grp.residues.map(String) : [];
        if (items.includes(String(residue)) || items.includes(String(r)) || items.includes(String(r + 1))) {
          assignedGroup = String(grp.name || '');
          break;
        }
      }
      rows.push({
        residueIndex: r,
        residue,
        group: assignedGroup || 'Ungrouped',
        cellValues: selectedSampleIds.map((sid) => ({
          sampleId: sid,
          value: values?.[sid]?.[r] || null,
          passes: Boolean(perTrajPass?.[sid]?.[r]),
        })),
      });
    }
    rows.sort((a, b) => {
      const ga = a.group || 'Ungrouped';
      const gb = b.group || 'Ungrouped';
      if (ga !== gb) {
        const ia = rowGroups.findIndex((g) => String(g.name) === String(ga));
        const ib = rowGroups.findIndex((g) => String(g.name) === String(gb));
        const oa = ia >= 0 ? ia : Number.MAX_SAFE_INTEGER;
        const ob = ib >= 0 ? ib : Number.MAX_SAFE_INTEGER;
        if (oa !== ob) return oa - ob;
        return String(ga).localeCompare(String(gb));
      }
      return String(a.residue).localeCompare(String(b.residue));
    });
    return rows;
  }, [
    data,
    selectedSampleIds,
    sampleIndexById,
    residueLabels,
    jsFilters,
    displayNormalizedJs,
    rowGroups,
    edgeSmoothEnabled,
    edgeSmoothStrength,
    jsEdgeA,
    jsEdgeB,
    topEdgeIndices,
    edgesAll,
    DEdge,
  ]);

  const groupedRows = useMemo(() => {
    const out = [];
    let current = '';
    for (const row of tableRows) {
      if (row.group !== current) {
        current = row.group;
        out.push({ kind: 'header', group: current });
      }
      out.push({ kind: 'row', row });
    }
    return out;
  }, [tableRows]);

  const handleSaveFilterSetup = useCallback(async () => {
    if (!selectedClusterId) return;
    const name = String(newFilterSetupName || '').trim();
    if (!name) return;
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
  }, [projectId, systemId, selectedClusterId, newFilterSetupName, jsFilters, loadFilterSetups]);

  const handleLoadFilterSetup = useCallback(() => {
    if (!selectedFilterSetupId) return;
    const entry = filterSetups.find((x) => String(x?.setup_id) === String(selectedFilterSetupId));
    const rules = jsRulesFromPayload(entry?.payload, JS_MAX);
    if (Array.isArray(rules) && rules.length) {
      setJsFilters(rules);
      setFilterSetupsError(null);
    }
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

  const handleAddGroup = useCallback(() => {
    const name = String(newGroupName || '').trim();
    const residues = expandGroupSelection(newGroupResidues, residueLabels);
    if (!name || !residues.length) return;
    setRowGroups((prev) => {
      const rest = prev.filter((g) => String(g.name) !== name);
      return [...rest, { name, residues, color: groupPalette[rest.length % groupPalette.length] }];
    });
    setNewGroupName('');
    setNewGroupResidues('');
  }, [newGroupName, newGroupResidues, residueLabels]);

  const moveSelectedSample = useCallback((sid, dir) => {
    setSelectedSampleIds((prev) => {
      const idx = prev.indexOf(sid);
      if (idx < 0) return prev;
      const nextIdx = idx + dir;
      if (nextIdx < 0 || nextIdx >= prev.length) return prev;
      const next = [...prev];
      [next[idx], next[nextIdx]] = [next[nextIdx], next[idx]];
      return next;
    });
  }, []);

  const handleSaveGroupSetup = useCallback(async () => {
    if (!selectedClusterId) return;
    const name = String(newGroupSetupName || '').trim();
    if (!name) return;
    setGroupSetupsError(null);
    try {
      const saved = await saveClusterUiSetup(projectId, systemId, selectedClusterId, {
        name,
        setup_type: 'delta_js_residue_groups',
        page: 'delta_js_table',
        payload: { groups: rowGroups },
      });
      setNewGroupSetupName('');
      await loadGroupSetups();
      if (saved?.setup_id) setSelectedGroupSetupId(String(saved.setup_id));
    } catch (err) {
      setGroupSetupsError(err.message || 'Failed to save group setup.');
    }
  }, [projectId, systemId, selectedClusterId, newGroupSetupName, rowGroups, loadGroupSetups]);

  const handleLoadGroupSetup = useCallback(() => {
    if (!selectedGroupSetupId) return;
    const entry = groupSetups.find((x) => String(x?.setup_id) === String(selectedGroupSetupId));
    const groups = entry?.payload?.groups;
    if (Array.isArray(groups)) {
      setRowGroups(
        groups.map((g, idx) => ({
          name: String(g?.name || `group_${idx + 1}`),
          residues: Array.isArray(g?.residues) ? g.residues.map(String) : [],
          color: String(g?.color || groupPalette[idx % groupPalette.length]),
        }))
      );
      setGroupSetupsError(null);
    }
  }, [selectedGroupSetupId, groupSetups]);

  const handleDeleteGroupSetup = useCallback(async () => {
    if (!selectedClusterId || !selectedGroupSetupId) return;
    setGroupSetupsError(null);
    try {
      await deleteClusterUiSetup(projectId, systemId, selectedClusterId, selectedGroupSetupId);
      const removedId = selectedGroupSetupId;
      await loadGroupSetups();
      if (String(selectedGroupSetupId) === String(removedId)) setSelectedGroupSetupId('');
    } catch (err) {
      setGroupSetupsError(err.message || 'Failed to delete group setup.');
    }
  }, [projectId, systemId, selectedClusterId, selectedGroupSetupId, loadGroupSetups]);

  if (loadingSystem) return <Loader message="Loading Delta JS table..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_js`)}
            className="text-cyan-400 hover:text-cyan-300 text-sm"
          >
            ← Back to Delta JS Evaluation
          </button>
          <h1 className="text-2xl font-semibold text-white">Delta JS Residue Table</h1>
          <p className="text-sm text-gray-400">Compare residue A/B-likeness across multiple trajectories in one grouped matrix table.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_js_3d${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            3D JS View
          </button>
          <button
            type="button"
            onClick={async () => {
              await loadClusterInfo();
              await loadAnalyses();
            }}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-4">
        <aside className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
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
              <label className="block text-xs text-gray-400 mb-1">Analysis</label>
              <select
                value={selectedAnalysisId}
                onChange={(e) => setSelectedAnalysisId(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {matchingAnalyses.map((a) => (
                  <option key={a.analysis_id} value={a.analysis_id}>
                    {(a.model_a_name || a.model_a_id || 'A')} vs {(a.model_b_name || a.model_b_id || 'B')} · {formatAnalysisShortMeta(a)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="block text-xs text-gray-400">Trajectories</label>
                <span className="text-[10px] text-gray-500">{selectedSampleIds.length} selected</span>
              </div>
              <div className="max-h-52 overflow-auto rounded-md border border-gray-800 bg-gray-950/30 p-2 space-y-1">
                {!analysisSampleIds.length && <p className="text-xs text-gray-500">No trajectories in selected analysis.</p>}
                {analysisSampleIds.map((sid, idx) => (
                  <label key={sid} className="flex items-center gap-2 text-xs text-gray-200">
                    <input
                      type="checkbox"
                      checked={selectedSampleIds.includes(sid)}
                      onChange={(e) => {
                        const checked = e.target.checked;
                        setSelectedSampleIds((prev) => {
                          const set = new Set(prev);
                          if (checked) set.add(sid);
                          else set.delete(sid);
                          const arr = [...set];
                          return arr.length ? arr : [sid];
                        });
                      }}
                    />
                    <span>{analysisSampleLabels[idx] || sid}</span>
                  </label>
                ))}
              </div>
              {!!selectedSampleIds.length && (
                <div className="mt-2 space-y-1 rounded-md border border-gray-800 bg-gray-950/20 p-2">
                  <p className="text-[10px] text-gray-500">Column order</p>
                  {selectedSampleIds.map((sid, pos) => {
                    const idx = sampleIndexById.get(String(sid));
                    const label = idx != null ? (analysisSampleLabels[idx] || sid) : sid;
                    return (
                      <div key={`ord:${sid}`} className="flex items-center justify-between gap-2 text-xs text-gray-200">
                        <span className="truncate">{pos + 1}. {label}</span>
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => moveSelectedSample(sid, -1)}
                            disabled={pos === 0}
                            className="px-1.5 py-0.5 rounded border border-gray-700 disabled:opacity-40"
                          >
                            ↑
                          </button>
                          <button
                            type="button"
                            onClick={() => moveSelectedSample(sid, 1)}
                            disabled={pos === selectedSampleIds.length - 1}
                            className="px-1.5 py-0.5 rounded border border-gray-700 disabled:opacity-40"
                          >
                            ↓
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <JsRangeFilterBuilder
              rules={jsFilters}
              onChange={setJsFilters}
              displayMax={displayNormalizedJs ? 1 : JS_MAX}
              unitLabel={displayNormalizedJs ? 'normalized JS [0,1]' : 'raw JS [0, ln(2)]'}
            />
            <div className="rounded-md border border-gray-800 bg-gray-950/30 p-3 space-y-2">
              <div className="text-xs text-gray-300">Residue score mode</div>
              <label className="flex items-center gap-2 text-xs text-gray-200">
                <input
                  type="checkbox"
                  checked={edgeSmoothEnabled}
                  onChange={(e) => setEdgeSmoothEnabled(e.target.checked)}
                />
                Edge-weighted node blending
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={edgeSmoothStrength}
                  onChange={(e) => setEdgeSmoothStrength(Number(e.target.value))}
                  disabled={!edgeSmoothEnabled}
                  className="w-full"
                />
                <span className="text-xs text-gray-400 w-12 text-right">{edgeSmoothStrength.toFixed(2)}</span>
              </div>
              <p className="text-[11px] text-gray-500">
                Off: node-only JS. On: blend node JS with adjacent edge JS using the selected weight.
              </p>
            </div>
            <div className="rounded-md border border-gray-800 bg-gray-950/30 p-3 space-y-2">
              <div className="text-xs text-gray-300">JS units</div>
              <label className="flex items-center gap-2 text-xs text-gray-200">
                <input
                  type="checkbox"
                  checked={displayNormalizedJs}
                  onChange={(e) => setDisplayNormalizedJs(e.target.checked)}
                />
                Show normalized JS in [0,1] (default)
              </label>
              <p className="text-[11px] text-gray-500">
                Filters are saved as normalized JS and mirrored as raw JS. Toggle off to view/edit the same ranges in raw nats [0, ln(2)=0.693].
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

            <div className="rounded-md border border-gray-800 bg-gray-950/30 p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs text-gray-300">Residue row groups</div>
                <button
                  type="button"
                  onClick={() => setGroupHelpOpen((prev) => !prev)}
                  className="text-[10px] px-2 py-1 rounded border border-gray-700 text-gray-300 hover:border-gray-500"
                >
                  {groupHelpOpen ? 'Hide examples' : 'Selection examples'}
                </button>
              </div>
              {groupHelpOpen && (
                <div className="rounded-md border border-gray-800 bg-gray-950/60 p-2 text-[11px] text-gray-300 space-y-1">
                  <div>Use comma-separated selectors:</div>
                  <div><code>ARG131,LEU132,GLY133</code> exact residue labels</div>
                  <div><code>131-160</code> residue-number range (inclusive)</div>
                  <div><code>#0-#24</code> zero-based row-index range</div>
                  <div><code>#10</code> one explicit zero-based row index</div>
                </div>
              )}
              <div className="grid grid-cols-1 gap-2">
                <input
                  type="text"
                  value={newGroupName}
                  onChange={(e) => setNewGroupName(e.target.value)}
                  placeholder="Group name (e.g. H6)"
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
                <input
                  type="text"
                  value={newGroupResidues}
                  onChange={(e) => setNewGroupResidues(e.target.value)}
                  placeholder="Residues/ranges (e.g. ARG131,LEU132 or 131-160 or #0-#24)"
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                />
                <button
                  type="button"
                  onClick={handleAddGroup}
                  className="inline-flex items-center justify-center gap-2 rounded-md border border-cyan-700 text-cyan-200 hover:border-cyan-500 px-2 py-2 text-sm"
                >
                  <Plus className="h-4 w-4" />
                  Add / Replace group
                </button>
              </div>
              {!!rowGroups.length && (
                <div className="space-y-1">
                  {rowGroups.map((g, idx) => (
                    <div key={`${g.name}:${idx}`} className="flex items-center justify-between gap-2 text-xs text-gray-200 rounded border border-gray-800 px-2 py-1">
                      <div className="min-w-0">
                        <span className="font-semibold" style={{ color: g.color || '#e5e7eb' }}>{g.name}</span>
                        <span className="text-gray-500 ml-2">{(g.residues || []).length} residues</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => setRowGroups((prev) => prev.filter((x) => x !== g))}
                        className="text-red-300 hover:text-red-200"
                      >
                        Delete
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <FilterSetupManager
              setups={groupSetups}
              selectedSetupId={selectedGroupSetupId}
              onSelectedSetupIdChange={setSelectedGroupSetupId}
              newSetupName={newGroupSetupName}
              onNewSetupNameChange={setNewGroupSetupName}
              onLoad={handleLoadGroupSetup}
              onSave={handleSaveGroupSetup}
              onDelete={handleDeleteGroupSetup}
              error={groupSetupsError}
            />

            {clusterInfoError && <ErrorMessage message={clusterInfoError} />}
            {analysesError && <ErrorMessage message={analysesError} />}
            {dataError && <ErrorMessage message={dataError} />}
          </div>
        </aside>

        <main className="space-y-3 min-w-0">
          {dataLoading && <Loader message="Loading analysis..." />}
          {!dataLoading && !tableRows.length && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 text-sm text-gray-400">
              No rows to display with current analysis / trajectories / filters.
            </div>
          )}
          {!!tableRows.length && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
              <div className="flex items-center justify-between mb-3">
                <div className="text-sm text-gray-200 font-semibold">Residue × trajectory table</div>
                <div className="text-xs text-gray-500">{tableRows.length} residues</div>
              </div>
              <div className="overflow-auto max-h-[72vh] rounded border border-gray-800">
                <table className="min-w-full text-xs">
                  <thead className="sticky top-0 bg-gray-950 z-10">
                    <tr>
                      <th className="px-2 py-2 text-left text-gray-300 border-b border-gray-800">Group</th>
                      <th className="px-2 py-2 text-left text-gray-300 border-b border-gray-800">Residue</th>
                      {selectedSampleIds.map((sid) => {
                        const idx = sampleIndexById.get(String(sid));
                        return (
                          <th key={`col:${sid}`} className="px-2 py-2 text-left text-gray-300 border-b border-gray-800 whitespace-nowrap">
                            {idx != null ? (analysisSampleLabels[idx] || sid) : sid}
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {groupedRows.map((item, idx) => {
                      if (item.kind === 'header') {
                        const grp = rowGroups.find((g) => String(g.name) === String(item.group));
                        return (
                          <tr key={`gh:${idx}`} className="bg-gray-950/80">
                            <td colSpan={2 + selectedSampleIds.length} className="px-2 py-1 border-t border-b border-gray-800">
                              <span className="text-[11px] font-semibold" style={{ color: grp?.color || '#9ca3af' }}>
                                {item.group}
                              </span>
                            </td>
                          </tr>
                        );
                      }
                      const row = item.row;
                      return (
                        <tr key={`r:${row.residueIndex}`} className="border-b border-gray-900/80">
                          <td className="px-2 py-1 text-gray-500">{row.group}</td>
                          <td className="px-2 py-1 text-gray-200 font-medium whitespace-nowrap">{row.residue}</td>
                          {row.cellValues.map((cell) => {
                            const dA = Number(cell?.value?.dA);
                            const dB = Number(cell?.value?.dB);
                            const dAShow = jsForDisplay(dA, displayNormalizedJs);
                            const dBShow = jsForDisplay(dB, displayNormalizedJs);
                            const has = Number.isFinite(dA) && Number.isFinite(dB);
                            const bg = has ? jsABOColor(dA, dB, cell.passes ? 0.95 : 0.25) : 'rgba(31,41,55,1)';
                            return (
                              <td
                                key={`c:${row.residueIndex}:${cell.sampleId}`}
                                className="px-2 py-1 text-gray-100 border-l border-gray-900 whitespace-nowrap"
                                style={{ background: bg }}
                                title={
                                  has
                                    ? `JS(A): raw=${dA.toFixed(4)}, norm=${(dA / JS_MAX).toFixed(4)}; JS(B): raw=${dB.toFixed(4)}, norm=${(dB / JS_MAX).toFixed(4)}${cell.passes ? '' : ' (not passing filter)'}`
                                    : 'No value'
                                }
                              >
                                {has ? `${dAShow.toFixed(2)} / ${dBShow.toFixed(2)}` : '-'}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
            <div className="text-sm text-gray-200 font-semibold mb-2">Color legend</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              {[
                { label: 'A-like', dA: 0.1, dB: 0.9 },
                { label: 'B-like', dA: 0.9, dB: 0.1 },
                { label: 'Similar to both', dA: 0.1, dB: 0.1 },
                { label: 'Far from both', dA: 0.9, dB: 0.9 },
              ].map((row) => (
                <div key={row.label} className="rounded border border-gray-800 p-2" style={{ background: jsABOColor(row.dA, row.dB, 0.95) }}>
                  <div className="font-semibold text-white">{row.label}</div>
                  <div className="text-gray-100">
                    {displayNormalizedJs
                      ? `JS(A)=${(row.dA / JS_MAX).toFixed(2)}, JS(B)=${(row.dB / JS_MAX).toFixed(2)}`
                      : `JS(A)=${row.dA.toFixed(2)}, JS(B)=${row.dB.toFixed(2)}`}
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[11px] text-gray-500">
              Colors are always computed from raw JS; this toggle changes only displayed numeric units.
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}
