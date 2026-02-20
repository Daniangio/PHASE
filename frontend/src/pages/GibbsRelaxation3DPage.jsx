import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { CircleHelp, Pause, Play, RefreshCw, RotateCcw } from 'lucide-react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Script } from 'molstar/lib/mol-script/script';
import { StructureElement, StructureProperties, StructureSelection } from 'molstar/lib/mol-model/structure';
import { clearStructureOverpaint, setStructureOverpaint } from 'molstar/lib/mol-plugin-state/helpers/structure-overpaint';
import 'molstar/build/viewer/molstar.css';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function clamp(x, lo, hi) {
  if (!Number.isFinite(x)) return lo;
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
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

function hexToInt(colorHex) {
  if (!colorHex) return 0xffffff;
  const s = String(colorHex).trim();
  const hex = s.startsWith('#') ? s.slice(1) : s;
  const v = parseInt(hex, 16);
  return Number.isFinite(v) ? v : 0xffffff;
}

function parseResidueId(label) {
  if (label == null) return null;
  const m = String(label).match(/-?\d+/);
  if (!m) return null;
  const v = Number(m[0]);
  return Number.isFinite(v) ? v : null;
}

function percentileColor(q) {
  const t = clamp01(q);
  if (t <= 0.5) {
    const u = t / 0.5;
    return rgbToHex(lerp(59, 255, u), lerp(130, 255, u), lerp(246, 255, u));
  }
  const u = (t - 0.5) / 0.5;
  return rgbToHex(lerp(255, 239, u), lerp(255, 68, u), lerp(255, 68, u));
}

function Gibbs3DPane({
  paneId,
  title,
  projectId,
  systemId,
  stateOptions,
  analysisData,
  analysisMeta,
  analysisLoading,
  analysisError,
  residueIdMode,
  flipStat,
  animateWave,
  currentStep,
  dataKind = 'percentile',
  deltaRange = 0.25,
}) {
  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  const baseComponentRef = useRef(null);
  const lociLabelProviderRef = useRef(null);
  const loadedOnceRef = useRef(false);

  const [viewerError, setViewerError] = useState(null);
  const [viewerStatus, setViewerStatus] = useState('initializing');
  const [structureLoading, setStructureLoading] = useState(false);
  const [loadedStructureStateId, setLoadedStructureStateId] = useState('');
  const loadedStateResidShift = useMemo(() => {
    if (!loadedStructureStateId) return 0;
    const st =
      stateOptions.find((s) => String(s?.state_id || '') === String(loadedStructureStateId)) ||
      stateOptions.find((s) => String(s?.id || '') === String(loadedStructureStateId)) ||
      null;
    const raw = Number(st?.resid_shift);
    return Number.isFinite(raw) ? Math.trunc(raw) : 0;
  }, [stateOptions, loadedStructureStateId]);

  const hoverRef = useRef({
    residueIdMode: 'auth',
    flipStat: 'mean',
    currentStep: 0,
    animateWave: true,
    authToData: new Map(),
    labelSeqToData: [],
  });

  const residueLabels = useMemo(() => {
    const keys = analysisData?.data?.residue_keys;
    if (Array.isArray(keys) && keys.length) return keys.map((x) => String(x));
    const mean = analysisData?.data?.mean_first_flip_steps;
    const n = Array.isArray(mean) ? mean.length : 0;
    return Array.from({ length: n }, (_, i) => `res_${i}`);
  }, [analysisData]);

  const colorMetric = useMemo(() => {
    if (dataKind === 'delta') {
      const arr = analysisData?.data?.delta_flip_percentile;
      return Array.isArray(arr) ? arr.map((v) => Number(v)) : [];
    }
    const arr = analysisData?.data?.flip_percentile_fast;
    return Array.isArray(arr) ? arr.map((v) => Number(v)) : [];
  }, [analysisData, dataKind]);

  const firstFlipByStat = useMemo(() => {
    const data = analysisData?.data || {};
    const map = {
      mean: data.mean_first_flip_steps,
      median: data.median_first_flip_steps,
      q25: data.q25_first_flip_steps,
      q75: data.q75_first_flip_steps,
    };
    const raw = map[flipStat] || map.mean || [];
    return Array.isArray(raw) ? raw.map((v) => Number(v)) : [];
  }, [analysisData, flipStat]);

  useEffect(() => {
    let disposed = false;
    let rafId;
    const tryInit = async () => {
      if (disposed) return;
      if (!containerRef.current) {
        rafId = requestAnimationFrame(tryInit);
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
    tryInit();
    return () => {
      disposed = true;
      if (rafId) cancelAnimationFrame(rafId);
      if (pluginRef.current) {
        try {
          pluginRef.current.dispose?.();
        } catch {
          // ignore
        }
        pluginRef.current = null;
      }
    };
  }, []);

  const getBaseComponentWrapper = useCallback(() => {
    const plugin = pluginRef.current;
    const baseRef = baseComponentRef.current;
    if (!plugin || !baseRef) return null;
    const root = plugin.managers.structure.hierarchy.current.structures[0];
    const comps = root?.components;
    if (!Array.isArray(comps)) return null;
    const found = comps.find((c) => c?.cell?.transform?.ref === baseRef) || null;
    if (!found) return null;
    if (!Array.isArray(found.representations)) return null;
    return found;
  }, []);

  const clearOverpaint = useCallback(async () => {
    const plugin = pluginRef.current;
    const base = getBaseComponentWrapper();
    if (!plugin || !base) return;
    try {
      await clearStructureOverpaint(plugin, [base], ['cartoon']);
    } catch {
      // ignore
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
      // best effort
    }

    const allExpr = MS.struct.generator.all();
    const baseComponent = await plugin.builders.structure.tryCreateComponentFromExpression(
      structureCell,
      allExpr,
      `phase-gibbs-relax-base-${paneId}`
    );
    if (!baseComponent) return null;

    await plugin.builders.structure.representation.addRepresentation(baseComponent, {
      type: 'cartoon',
      color: 'uniform',
      colorParams: { value: hexToInt('#9ca3af') },
    });
    baseComponentRef.current = baseComponent.ref;
    return getBaseComponentWrapper();
  }, [getBaseComponentWrapper, paneId]);

  const loadStructure = useCallback(
    async (stateIdOverride) => {
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
          const data = await plugin.builders.data.download(
            { url: Asset.Url(url), label: sid },
            { state: { isGhost: true } }
          );
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
    },
    [projectId, systemId, loadedStructureStateId, ensureBaseComponent, clearOverpaint]
  );

  useEffect(() => {
    if (viewerStatus !== 'ready') return;
    if (!stateOptions.length) return;
    if (!loadedStructureStateId) {
      const first = stateOptions[0]?.state_id;
      if (first) setLoadedStructureStateId(first);
    }
  }, [viewerStatus, stateOptions, loadedStructureStateId]);

  useEffect(() => {
    if (viewerStatus !== 'ready') return;
    if (loadedOnceRef.current) return;
    if (!loadedStructureStateId) return;
    loadedOnceRef.current = true;
    loadStructure(loadedStructureStateId);
  }, [viewerStatus, loadedStructureStateId, loadStructure]);

  const coloringPayload = useMemo(() => {
    if (!residueLabels.length || !colorMetric.length) return null;
    const n = dataKind === 'delta'
      ? Math.min(residueLabels.length, colorMetric.length)
      : Math.min(residueLabels.length, colorMetric.length, firstFlipByStat.length);
    const residueIdsAuth = [];
    const valuesAuth = [];
    const firstAuth = [];
    const residueIdsLabel = [];
    const valuesLabel = [];
    const firstLabel = [];

    for (let i = 0; i < n; i += 1) {
      const v = Number(colorMetric[i]);
      const ff = Number(firstFlipByStat[i]);
      const canonicalAuth = parseResidueId(residueLabels[i]);
      const auth = Number.isFinite(canonicalAuth)
        ? Number(canonicalAuth) - Number(loadedStateResidShift || 0)
        : null;
      if (auth !== null) {
        residueIdsAuth.push(auth);
        valuesAuth.push(v);
        if (dataKind !== 'delta') firstAuth.push(ff);
      }
      residueIdsLabel.push(i + 1);
      valuesLabel.push(v);
      if (dataKind !== 'delta') firstLabel.push(ff);
    }

    return {
      residueIdsAuth,
      valuesAuth,
      firstAuth,
      residueIdsLabel,
      valuesLabel,
      firstLabel,
    };
  }, [residueLabels, colorMetric, firstFlipByStat, dataKind, loadedStateResidShift]);

  useEffect(() => {
    const authToData = new Map();
    const labelSeqToData = [];
    if (coloringPayload) {
      const { residueIdsAuth, valuesAuth, firstAuth, residueIdsLabel, valuesLabel, firstLabel } = coloringPayload;
      for (let i = 0; i < residueIdsAuth.length; i += 1) {
        authToData.set(Number(residueIdsAuth[i]), {
          value: Number(valuesAuth[i]),
          first: Number(dataKind === 'delta' ? NaN : firstAuth[i]),
        });
      }
      for (let i = 0; i < residueIdsLabel.length; i += 1) {
        const rid = Number(residueIdsLabel[i]);
        const idx = Math.floor(rid) - 1;
        if (idx >= 0) {
          labelSeqToData[idx] = {
            value: Number(valuesLabel[i]),
            first: Number(dataKind === 'delta' ? NaN : firstLabel[i]),
          };
        }
      }
    }
    hoverRef.current = {
      residueIdMode,
      flipStat,
      currentStep: Number(currentStep),
      animateWave: Boolean(animateWave),
      authToData,
      labelSeqToData,
    };
  }, [coloringPayload, residueIdMode, flipStat, currentStep, animateWave, dataKind]);

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
        const labelSeq = Number(StructureProperties.residue.label_seq_id(loc));

        let row = null;
        if (h.residueIdMode === 'auth') {
          row = h.authToData.get(auth) || null;
        } else {
          const idx = Number.isFinite(labelSeq) ? Math.floor(labelSeq) - 1 : -1;
          if (idx >= 0 && idx < h.labelSeqToData.length) row = h.labelSeqToData[idx] || null;
        }

        const q = Number(row?.value);
        const ff = Number(row?.first);
        const qText = Number.isFinite(q) ? q.toFixed(3) : 'n/a';
        const ffText = Number.isFinite(ff) ? ff.toFixed(2) : 'n/a';
        const statLabel = h.flipStat === 'median' ? 'Median' : h.flipStat === 'q25' ? 'Q25' : h.flipStat === 'q75' ? 'Q75' : 'Mean';
        const wave = dataKind !== 'delta' && h.animateWave && Number.isFinite(ff)
          ? (ff > h.currentStep ? ` · not flipped @ ${h.currentStep}` : ` · flipped @ ${h.currentStep}`)
          : '';
        if (dataKind === 'delta') {
          return `Δ flip pct (A-B): ${qText}`;
        }
        return `Flip pct: ${qText} · ${statLabel} first flip: ${ffText}${wave}`;
      },
      group: (label) => `phase-gibbs-relax-${paneId}:${label}`,
    };

    plugin.managers.lociLabels.addProvider(provider);
    lociLabelProviderRef.current = provider;
    return () => {
      try {
        plugin.managers.lociLabels.removeProvider(provider);
      } catch {
        // ignore
      }
      if (lociLabelProviderRef.current === provider) lociLabelProviderRef.current = null;
    };
  }, [viewerStatus, paneId, dataKind]);

  const applyColoring = useCallback(async () => {
    if (viewerStatus !== 'ready') return;
    if (structureLoading) return;
    if (!analysisData || !coloringPayload) return;
    const plugin = pluginRef.current;
    const base = getBaseComponentWrapper();
    if (!plugin || !base) return;

    await clearOverpaint();

    const useAuth = residueIdMode === 'auth';
    const residueIds = useAuth ? coloringPayload.residueIdsAuth : coloringPayload.residueIdsLabel;
    const qValues = useAuth ? coloringPayload.valuesAuth : coloringPayload.valuesLabel;
    const firstFlip = useAuth ? coloringPayload.firstAuth : coloringPayload.firstLabel;
    if (!residueIds.length) return;

    const bins = 21;
    const buckets = Array.from({ length: bins }, () => []);
    for (let i = 0; i < residueIds.length; i += 1) {
      const q = Number(qValues[i]);
      if (!Number.isFinite(q)) continue;
      const ff = Number(firstFlip[i]);
      if (dataKind !== 'delta' && animateWave && Number.isFinite(ff) && ff > Number(currentStep)) continue;
      const qVis = dataKind === 'delta'
        ? clamp01((clamp(q, -deltaRange, deltaRange) / Math.max(deltaRange, 1e-9) + 1) * 0.5)
        : clamp01(q);
      const b = Math.max(0, Math.min(bins - 1, Math.floor(qVis * (bins - 1) + 1e-9)));
      buckets[b].push(residueIds[i]);
    }

    const propFn =
      residueIdMode === 'auth'
        ? MS.struct.atomProperty.macromolecular.auth_seq_id()
        : MS.struct.atomProperty.macromolecular.label_seq_id();

    for (let b = 0; b < bins; b += 1) {
      const ids = buckets[b];
      if (!ids.length) continue;
      const colorHex = percentileColor(b / (bins - 1));
      const colorValue = hexToInt(colorHex);
      const residueTests = ids.length === 1 ? MS.core.rel.eq([propFn, ids[0]]) : MS.core.set.has([MS.set(...ids), propFn]);
      const expression = MS.struct.generator.atomGroups({ 'residue-test': residueTests });
      const lociGetter = async (structure) => {
        const sel = Script.getStructureSelection(expression, structure);
        return StructureSelection.toLociWithSourceUnits(sel);
      };
      // eslint-disable-next-line no-await-in-loop
      await setStructureOverpaint(plugin, [base], colorValue, lociGetter, ['cartoon']);
    }
  }, [
    viewerStatus,
    structureLoading,
    analysisData,
    coloringPayload,
    residueIdMode,
    animateWave,
    currentStep,
    dataKind,
    deltaRange,
    clearOverpaint,
    getBaseComponentWrapper,
  ]);

  useEffect(() => {
    applyColoring();
  }, [applyColoring]);

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-gray-200">{title}</h2>
          <p className="text-[11px] text-gray-500">
            {analysisMeta ? `${analysisMeta.model_name || analysisMeta.model_id} · ${analysisMeta.start_sample_name || analysisMeta.start_sample_id}` : 'Select analysis'}
            {dataKind === 'delta' ? ' · Delta mode (A - B)' : ''}
          </p>
        </div>
        <button
          type="button"
          onClick={() => loadStructure()}
          className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
        >
          <RefreshCw className="h-4 w-4" />
          Reload
        </button>
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">Structure</label>
        <div className="flex flex-wrap gap-2">
          {stateOptions.map((st) => {
            const sid = st?.state_id || st?.id || '';
            if (!sid) return null;
            const active = sid === loadedStructureStateId;
            return (
              <button
                key={`${paneId}-${sid}`}
                type="button"
                onClick={async () => {
                  setLoadedStructureStateId(sid);
                  await loadStructure(sid);
                }}
                className={`text-xs px-3 py-2 rounded-md border ${
                  active ? 'border-cyan-500 text-cyan-200' : 'border-gray-700 text-gray-200 hover:border-gray-500'
                }`}
              >
                {st?.name || sid}
              </button>
            );
          })}
        </div>
      </div>

      {(viewerError || viewerStatus === 'error') && <ErrorMessage message={viewerError || 'Viewer error'} />}
      {analysisError && <ErrorMessage message={analysisError} />}

      <div className="h-[64vh] min-h-[440px] rounded-md border border-gray-800 bg-black/20 overflow-hidden relative">
        {viewerStatus !== 'ready' && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/60">
            <Loader message="Initializing viewer..." />
          </div>
        )}
        {viewerStatus === 'ready' && (structureLoading || analysisLoading) && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50">
            <Loader message={structureLoading ? 'Loading structure...' : 'Loading analysis...'} />
          </div>
        )}
        <div ref={containerRef} className="w-full h-full relative" />
      </div>
    </div>
  );
}

export default function GibbsRelaxation3DPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [helpOpen, setHelpOpen] = useState(false);

  const [compareEnabled, setCompareEnabled] = useState(true);
  const [showDeltaPane, setShowDeltaPane] = useState(true);
  const [deltaRange, setDeltaRange] = useState(0.25);

  const [selectedClusterIdA, setSelectedClusterIdA] = useState('');
  const [selectedAnalysisIdA, setSelectedAnalysisIdA] = useState('');
  const [analysesA, setAnalysesA] = useState([]);
  const [analysisDataA, setAnalysisDataA] = useState(null);
  const [analysisDataLoadingA, setAnalysisDataLoadingA] = useState(false);
  const [analysisDataErrorA, setAnalysisDataErrorA] = useState(null);
  const [analysesErrorA, setAnalysesErrorA] = useState(null);

  const [selectedClusterIdB, setSelectedClusterIdB] = useState('');
  const [selectedAnalysisIdB, setSelectedAnalysisIdB] = useState('');
  const [analysesB, setAnalysesB] = useState([]);
  const [analysisDataB, setAnalysisDataB] = useState(null);
  const [analysisDataLoadingB, setAnalysisDataLoadingB] = useState(false);
  const [analysisDataErrorB, setAnalysisDataErrorB] = useState(null);
  const [analysesErrorB, setAnalysesErrorB] = useState(null);

  const [residueIdMode, setResidueIdMode] = useState('auth');
  const [flipStat, setFlipStat] = useState('mean');
  const [animateWave, setAnimateWave] = useState(true);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackMs, setPlaybackMs] = useState(120);
  const [loopPlayback, setLoopPlayback] = useState(false);

  const requestedClusterId = useMemo(() => String(searchParams.get('cluster_id') || '').trim(), [searchParams]);
  const requestedAnalysisId = useMemo(() => String(searchParams.get('analysis_id') || '').trim(), [searchParams]);

  const clusterOptions = useMemo(
    () => (system?.metastable_clusters || []).filter((run) => run.path && run.status !== 'failed'),
    [system]
  );

  const stateOptions = useMemo(() => {
    const raw = system?.states;
    if (!raw) return [];
    if (Array.isArray(raw)) return raw;
    if (typeof raw === 'object') return Object.values(raw);
    return [];
  }, [system]);

  const analysisMetaA = useMemo(
    () => analysesA.find((a) => a.analysis_id === selectedAnalysisIdA) || null,
    [analysesA, selectedAnalysisIdA]
  );
  const analysisMetaB = useMemo(
    () => analysesB.find((a) => a.analysis_id === selectedAnalysisIdB) || null,
    [analysesB, selectedAnalysisIdB]
  );

  useEffect(() => {
    const load = async () => {
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
    load();
  }, [projectId, systemId]);

  useEffect(() => {
    if (!clusterOptions.length) return;
    if (!selectedClusterIdA || !clusterOptions.some((c) => c.cluster_id === selectedClusterIdA)) {
      const fallback = clusterOptions[clusterOptions.length - 1].cluster_id;
      const requested = requestedClusterId && clusterOptions.some((c) => c.cluster_id === requestedClusterId) ? requestedClusterId : '';
      setSelectedClusterIdA(requested || fallback);
    }
    if (!selectedClusterIdB || !clusterOptions.some((c) => c.cluster_id === selectedClusterIdB)) {
      setSelectedClusterIdB(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterIdA, selectedClusterIdB, requestedClusterId]);

  const loadAnalysesA = useCallback(async () => {
    if (!selectedClusterIdA) return;
    setAnalysesErrorA(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterIdA, { analysisType: 'gibbs_relaxation' });
      const list = Array.isArray(data?.analyses) ? data.analyses : [];
      setAnalysesA(list);
      setSelectedAnalysisIdA((prev) => {
        if (prev && list.some((a) => a.analysis_id === prev)) return prev;
        if (requestedAnalysisId && list.some((a) => a.analysis_id === requestedAnalysisId)) return requestedAnalysisId;
        return list[0]?.analysis_id || '';
      });
    } catch (err) {
      setAnalysesErrorA(err.message || 'Failed to load analyses.');
      setAnalysesA([]);
      setSelectedAnalysisIdA('');
    }
  }, [projectId, systemId, selectedClusterIdA, requestedAnalysisId]);

  const loadAnalysesB = useCallback(async () => {
    if (!selectedClusterIdB) return;
    setAnalysesErrorB(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterIdB, { analysisType: 'gibbs_relaxation' });
      const list = Array.isArray(data?.analyses) ? data.analyses : [];
      setAnalysesB(list);
      setSelectedAnalysisIdB((prev) => (prev && list.some((a) => a.analysis_id === prev) ? prev : list[0]?.analysis_id || ''));
    } catch (err) {
      setAnalysesErrorB(err.message || 'Failed to load analyses.');
      setAnalysesB([]);
      setSelectedAnalysisIdB('');
    }
  }, [projectId, systemId, selectedClusterIdB]);

  useEffect(() => {
    if (!selectedClusterIdA) return;
    setAnalysesA([]);
    setSelectedAnalysisIdA('');
    setAnalysisDataA(null);
    loadAnalysesA();
  }, [selectedClusterIdA, loadAnalysesA]);

  useEffect(() => {
    if (!selectedClusterIdB) return;
    setAnalysesB([]);
    setSelectedAnalysisIdB('');
    setAnalysisDataB(null);
    loadAnalysesB();
  }, [selectedClusterIdB, loadAnalysesB]);

  useEffect(() => {
    const run = async () => {
      setAnalysisDataErrorA(null);
      setAnalysisDataA(null);
      if (!selectedClusterIdA || !selectedAnalysisIdA) return;
      setAnalysisDataLoadingA(true);
      try {
        const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterIdA, 'gibbs_relaxation', selectedAnalysisIdA);
        setAnalysisDataA(payload);
        setCurrentStep(0);
        setIsPlaying(false);
      } catch (err) {
        setAnalysisDataErrorA(err.message || 'Failed to load analysis.');
      } finally {
        setAnalysisDataLoadingA(false);
      }
    };
    run();
  }, [projectId, systemId, selectedClusterIdA, selectedAnalysisIdA]);

  useEffect(() => {
    const run = async () => {
      setAnalysisDataErrorB(null);
      setAnalysisDataB(null);
      if (!selectedClusterIdB || !selectedAnalysisIdB) return;
      setAnalysisDataLoadingB(true);
      try {
        const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterIdB, 'gibbs_relaxation', selectedAnalysisIdB);
        setAnalysisDataB(payload);
      } catch (err) {
        setAnalysisDataErrorB(err.message || 'Failed to load analysis.');
      } finally {
        setAnalysisDataLoadingB(false);
      }
    };
    run();
  }, [projectId, systemId, selectedClusterIdB, selectedAnalysisIdB]);

  const maxStep = useMemo(() => {
    const getMax = (data) => {
      const fromMeta = Number(data?.metadata?.gibbs_sweeps);
      const arr = data?.data?.[`${flipStat}_first_flip_steps`] || data?.data?.mean_first_flip_steps || [];
      const fromFirst = Array.isArray(arr) ? arr.reduce((m, v) => Math.max(m, Number(v) || 0), 0) : 0;
      return Math.max(Number.isFinite(fromMeta) ? fromMeta : 0, fromFirst);
    };
    return Math.max(1, Math.ceil(Math.max(getMax(analysisDataA), getMax(analysisDataB)) || 1));
  }, [analysisDataA, analysisDataB, flipStat]);

  const deltaAnalysisData = useMemo(() => {
    if (!compareEnabled || !showDeltaPane) return null;
    const keysA = Array.isArray(analysisDataA?.data?.residue_keys) ? analysisDataA.data.residue_keys.map((x) => String(x)) : [];
    const keysB = Array.isArray(analysisDataB?.data?.residue_keys) ? analysisDataB.data.residue_keys.map((x) => String(x)) : [];
    const qA = Array.isArray(analysisDataA?.data?.flip_percentile_fast) ? analysisDataA.data.flip_percentile_fast.map((v) => Number(v)) : [];
    const qB = Array.isArray(analysisDataB?.data?.flip_percentile_fast) ? analysisDataB.data.flip_percentile_fast.map((v) => Number(v)) : [];
    if (!keysA.length || !keysB.length || !qA.length || !qB.length) return null;

    const bMap = new Map();
    for (let i = 0; i < Math.min(keysB.length, qB.length); i += 1) {
      bMap.set(keysB[i], qB[i]);
    }
    const outKeys = [];
    const outDelta = [];
    for (let i = 0; i < Math.min(keysA.length, qA.length); i += 1) {
      const k = keysA[i];
      const bv = bMap.get(k);
      const av = qA[i];
      if (!Number.isFinite(av) || !Number.isFinite(bv)) continue;
      outKeys.push(k);
      outDelta.push(av - bv);
    }
    if (!outKeys.length) return null;
    return {
      data: {
        residue_keys: outKeys,
        delta_flip_percentile: outDelta,
      },
      metadata: {
        left_model_name: analysisMetaA?.model_name || analysisMetaA?.model_id || '',
        right_model_name: analysisMetaB?.model_name || analysisMetaB?.model_id || '',
      },
    };
  }, [compareEnabled, showDeltaPane, analysisDataA, analysisDataB, analysisMetaA, analysisMetaB]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    if (!animateWave) return undefined;
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= maxStep) {
          if (loopPlayback) return 0;
          return maxStep;
        }
        return prev + 1;
      });
    }, Math.max(30, Number(playbackMs) || 120));
    return () => clearInterval(interval);
  }, [isPlaying, animateWave, maxStep, playbackMs, loopPlayback]);

  useEffect(() => {
    if (!isPlaying) return;
    if (currentStep >= maxStep && !loopPlayback) setIsPlaying(false);
  }, [currentStep, maxStep, isPlaying, loopPlayback]);

  if (loadingSystem) return <Loader message="Loading Gibbs relaxation 3D viewer..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Gibbs Relaxation (3D): How To Read It"
        docPath="/docs/gibbs_relaxation_3d_help.md"
        onClose={() => setHelpOpen(false)}
      />

      <div className="flex items-start justify-between gap-3">
        <div>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/gibbs_relaxation`)}
            className="text-cyan-400 hover:text-cyan-300 text-sm"
          >
            ← Back to Gibbs Relaxation
          </button>
          <h1 className="text-2xl font-semibold text-white">Gibbs Relaxation (3D)</h1>
          <p className="text-sm text-gray-400">Compare two analyses side by side with independent structures.</p>
        </div>
        <button
          type="button"
          onClick={() => setHelpOpen(true)}
          className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
        >
          <CircleHelp className="h-4 w-4" />
          Help
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[300px_1fr] gap-4">
        <aside className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
            <h2 className="text-sm font-semibold text-gray-200">Comparison Setup</h2>

            <label className="flex items-center gap-2 text-sm text-gray-200">
              <input
                type="checkbox"
                checked={compareEnabled}
                onChange={(e) => setCompareEnabled(e.target.checked)}
                className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
              />
              Side-by-side compare
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-200">
              <input
                type="checkbox"
                checked={showDeltaPane}
                onChange={(e) => setShowDeltaPane(e.target.checked)}
                disabled={!compareEnabled}
                className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950 disabled:opacity-50"
              />
              Show delta view (A-B)
            </label>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Delta color range (|Δ|)</label>
              <input
                type="number"
                min={0.01}
                max={1}
                step={0.01}
                value={deltaRange}
                onChange={(e) => setDeltaRange(Math.max(0.01, Number(e.target.value) || 0.25))}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                disabled={!compareEnabled || !showDeltaPane}
              />
              <p className="text-[11px] text-gray-500 mt-1">Blue: B faster, Red: A faster. Values clip at ±range.</p>
            </div>

            <div className="rounded-md border border-gray-800 bg-gray-950/40 p-2 space-y-2">
              <p className="text-xs font-semibold text-gray-300">Left analysis</p>
              <select
                value={selectedClusterIdA}
                onChange={(e) => setSelectedClusterIdA(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {clusterOptions.map((c) => (
                  <option key={`a-${c.cluster_id}`} value={c.cluster_id}>{c.name || c.cluster_id}</option>
                ))}
              </select>
              <select
                value={selectedAnalysisIdA}
                onChange={(e) => setSelectedAnalysisIdA(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {analysesA.map((a) => (
                  <option key={`a-${a.analysis_id}`} value={a.analysis_id}>
                    {(a.model_name || a.model_id || 'model')} · {(a.start_sample_name || a.start_sample_id || 'sample')}
                  </option>
                ))}
              </select>
              {analysesErrorA && <ErrorMessage message={analysesErrorA} />}
            </div>

            {compareEnabled && (
              <div className="rounded-md border border-gray-800 bg-gray-950/40 p-2 space-y-2">
                <p className="text-xs font-semibold text-gray-300">Right analysis</p>
                <select
                  value={selectedClusterIdB}
                  onChange={(e) => setSelectedClusterIdB(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  {clusterOptions.map((c) => (
                    <option key={`b-${c.cluster_id}`} value={c.cluster_id}>{c.name || c.cluster_id}</option>
                  ))}
                </select>
                <select
                  value={selectedAnalysisIdB}
                  onChange={(e) => setSelectedAnalysisIdB(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                >
                  {analysesB.map((a) => (
                    <option key={`b-${a.analysis_id}`} value={a.analysis_id}>
                      {(a.model_name || a.model_id || 'model')} · {(a.start_sample_name || a.start_sample_id || 'sample')}
                    </option>
                  ))}
                </select>
                {analysesErrorB && <ErrorMessage message={analysesErrorB} />}
              </div>
            )}

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

            <div>
              <label className="block text-xs text-gray-400 mb-1">Flip statistic</label>
              <select
                value={flipStat}
                onChange={(e) => setFlipStat(e.target.value)}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                <option value="mean">Mean first flip</option>
                <option value="median">Median first flip</option>
                <option value="q25">Q25 first flip</option>
                <option value="q75">Q75 first flip</option>
              </select>
            </div>

            <div className="rounded-md border border-gray-800 bg-gray-950/40 p-2 space-y-2">
              <label className="flex items-center gap-2 text-sm text-gray-200">
                <input
                  type="checkbox"
                  checked={animateWave}
                  onChange={(e) => setAnimateWave(e.target.checked)}
                  className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                />
                Animate wave
              </label>

              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Step: {currentStep} / {maxStep}
                </label>
                <input
                  type="range"
                  min={0}
                  max={maxStep}
                  step={1}
                  value={currentStep}
                  onChange={(e) => setCurrentStep(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => setIsPlaying((v) => !v)}
                  disabled={!animateWave}
                  className="text-xs px-2 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 disabled:opacity-50 inline-flex items-center justify-center gap-1"
                >
                  {isPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                  {isPlaying ? 'Pause' : 'Play'}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCurrentStep(0);
                    setIsPlaying(false);
                  }}
                  className="text-xs px-2 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center justify-center gap-1"
                >
                  <RotateCcw className="h-3 w-3" />
                  Reset
                </button>
                <select
                  value={String(playbackMs)}
                  onChange={(e) => setPlaybackMs(Number(e.target.value))}
                  className="text-xs bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-gray-100"
                >
                  <option value="60">Fast</option>
                  <option value="120">Normal</option>
                  <option value="220">Slow</option>
                </select>
              </div>

              <label className="flex items-center gap-2 text-xs text-gray-300">
                <input
                  type="checkbox"
                  checked={loopPlayback}
                  onChange={(e) => setLoopPlayback(e.target.checked)}
                  className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                />
                Loop animation
              </label>
            </div>
          </div>
        </aside>

        <main className={`grid gap-4 ${
          compareEnabled
            ? (showDeltaPane ? 'grid-cols-1 2xl:grid-cols-3' : 'grid-cols-1 2xl:grid-cols-2')
            : 'grid-cols-1'
        }`}>
          <Gibbs3DPane
            paneId="left"
            title="Left"
            projectId={projectId}
            systemId={systemId}
            stateOptions={stateOptions}
            analysisData={analysisDataA}
            analysisMeta={analysisMetaA}
            analysisLoading={analysisDataLoadingA}
            analysisError={analysisDataErrorA}
            residueIdMode={residueIdMode}
            flipStat={flipStat}
            animateWave={animateWave}
            currentStep={currentStep}
            dataKind="percentile"
            deltaRange={deltaRange}
          />

          {compareEnabled && (
            <Gibbs3DPane
              paneId="right"
              title="Right"
              projectId={projectId}
              systemId={systemId}
              stateOptions={stateOptions}
              analysisData={analysisDataB}
              analysisMeta={analysisMetaB}
              analysisLoading={analysisDataLoadingB}
              analysisError={analysisDataErrorB}
              residueIdMode={residueIdMode}
              flipStat={flipStat}
              animateWave={animateWave}
              currentStep={currentStep}
              dataKind="percentile"
              deltaRange={deltaRange}
            />
          )}
          {compareEnabled && showDeltaPane && (
            <Gibbs3DPane
              paneId="delta"
              title="Delta (A - B)"
              projectId={projectId}
              systemId={systemId}
              stateOptions={stateOptions}
              analysisData={deltaAnalysisData}
              analysisMeta={{ model_name: `${analysisMetaA?.model_name || analysisMetaA?.model_id || 'A'} minus ${analysisMetaB?.model_name || analysisMetaB?.model_id || 'B'}`, start_sample_name: '' }}
              analysisLoading={analysisDataLoadingA || analysisDataLoadingB}
              analysisError={null}
              residueIdMode={residueIdMode}
              flipStat={flipStat}
              animateWave={false}
              currentStep={currentStep}
              dataKind="delta"
              deltaRange={deltaRange}
            />
          )}
        </main>
      </div>
    </div>
  );
}
