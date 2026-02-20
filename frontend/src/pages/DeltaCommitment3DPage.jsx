import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, RefreshCw } from 'lucide-react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import { PluginCommands } from 'molstar/lib/mol-plugin/commands';
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Script } from 'molstar/lib/mol-script/script';
import { StructureElement, StructureSelection, StructureProperties } from 'molstar/lib/mol-model/structure';
import { clearStructureOverpaint, setStructureOverpaint } from 'molstar/lib/mol-plugin-state/helpers/structure-overpaint';
import { StateSelection } from 'molstar/lib/mol-state';
import { PluginStateObject, PluginStateTransform } from 'molstar/lib/mol-plugin-state/objects';
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms';
import { ParamDefinition as PD } from 'molstar/lib/mol-util/param-definition';
import { Task } from 'molstar/lib/mol-task';
import { Vec3 } from 'molstar/lib/mol-math/linear-algebra';
import { Mesh } from 'molstar/lib/mol-geo/geometry/mesh/mesh';
import { MeshBuilder } from 'molstar/lib/mol-geo/geometry/mesh/mesh-builder';
import { addCylinder } from 'molstar/lib/mol-geo/geometry/mesh/builder/cylinder';
import { Shape } from 'molstar/lib/mol-model/shape';
import 'molstar/build/viewer/molstar.css';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import {
  fetchClusterAnalyses,
  fetchClusterAnalysisData,
  fetchPottsClusterInfo,
  fetchSampleResidueProfile,
  fetchSystem,
} from '../api/projects';

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  if (x < 0) return 0;
  if (x > 1) return 1;
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

function commitmentColor(q) {
  // Diverging map: 0 -> blue, 0.5 -> white, 1 -> red
  const t = clamp01(q);
  if (t <= 0.5) {
    const u = t / 0.5;
    return rgbToHex(lerp(59, 255, u), lerp(130, 255, u), lerp(246, 255, u));
  }
  const u = (t - 0.5) / 0.5;
  return rgbToHex(lerp(255, 239, u), lerp(255, 68, u), lerp(255, 68, u));
}

function sigmoid(x) {
  if (!Number.isFinite(x)) return 0.5;
  // numerically stable sigmoid
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

function parseResidueId(label) {
  if (label == null) return null;
  const m = String(label).match(/-?\d+/);
  if (!m) return null;
  const v = Number(m[0]);
  return Number.isFinite(v) ? v : null;
}

function pieColor(idx) {
  const hue = (idx * 137.507764) % 360;
  return `hsl(${hue} 70% 55%)`;
}

function clusterColorFromKey(key, fallbackIdx = 0) {
  if (key == null) return pieColor(fallbackIdx);
  const nums = String(key).match(/-?\d+/g);
  if (!nums || !nums.length) return pieColor(fallbackIdx);
  let hash = 17;
  nums.forEach((tok) => {
    const n = Number(tok);
    if (Number.isFinite(n)) hash = hash * 31 + Math.abs(Math.trunc(n));
  });
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue} 70% 55%)`;
}

function compressPieSlices(rawSlices, maxSlices = 8) {
  const clean = (rawSlices || [])
    .map((s) => ({
      label: String(s?.label ?? ''),
      value: Number(s?.value ?? 0),
      tooltip: String(s?.tooltip || ''),
      colorKey: s?.colorKey,
      color: s?.color,
    }))
    .filter((s) => Number.isFinite(s.value) && s.value > 0);
  if (!clean.length) return [];
  clean.sort((a, b) => b.value - a.value);
  const keep = clean.slice(0, maxSlices);
  const rest = clean.slice(maxSlices);
  const other = rest.reduce((acc, s) => acc + s.value, 0);
  const out = keep.map((s, i) => ({
    ...s,
    color: s.color || clusterColorFromKey(s.colorKey || s.label, i),
  }));
  if (other > 0) out.push({ label: 'other', value: other, color: '#6b7280' });
  const total = out.reduce((acc, s) => acc + s.value, 0);
  if (total <= 0) return [];
  return out.map((s) => ({ ...s, value: s.value / total }));
}

function vecToSlices(values, labelPrefix = 'c', clusterIds = null) {
  if (!Array.isArray(values)) return [];
  return values.map((v, idx) => {
    const cid = Array.isArray(clusterIds) && idx < clusterIds.length ? Number(clusterIds[idx]) : idx;
    const clusterId = Number.isFinite(cid) ? cid : idx;
    return {
      label: `${labelPrefix}${clusterId}`,
      value: Number(v) || 0,
      colorKey: `c${clusterId}`,
      tooltip: `cluster ${labelPrefix}${clusterId}`,
    };
  });
}

function jointToSlices(joint, clusterIdsR = null, clusterIdsS = null, rLabel = 'res_source', sLabel = 'res_target') {
  if (!Array.isArray(joint)) return [];
  const out = [];
  for (let a = 0; a < joint.length; a += 1) {
    const row = joint[a];
    if (!Array.isArray(row)) continue;
    for (let b = 0; b < row.length; b += 1) {
      const cidA =
        Array.isArray(clusterIdsR) && a < clusterIdsR.length && Number.isFinite(Number(clusterIdsR[a]))
          ? Number(clusterIdsR[a])
          : a;
      const cidB =
        Array.isArray(clusterIdsS) && b < clusterIdsS.length && Number.isFinite(Number(clusterIdsS[b]))
          ? Number(clusterIdsS[b])
          : b;
      out.push({
        label: `c${cidA}-c${cidB}`,
        value: Number(row[b]) || 0,
        colorKey: `c${cidA}-c${cidB}`,
        tooltip: `${rLabel}: c${cidA} · ${sLabel}: c${cidB}`,
      });
    }
  }
  return out;
}

function piePath(cx, cy, r, start, end) {
  const x0 = cx + r * Math.cos(start);
  const y0 = cy + r * Math.sin(start);
  const x1 = cx + r * Math.cos(end);
  const y1 = cy + r * Math.sin(end);
  const large = end - start > Math.PI ? 1 : 0;
  return `M ${cx} ${cy} L ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1} Z`;
}

function PieChart({ slices, size = 120, onSliceClick = null }) {
  const radius = size * 0.45;
  const center = size / 2;
  let acc = -Math.PI / 2;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      <circle cx={center} cy={center} r={radius} fill="#111827" />
      {slices.map((slice, idx) => {
        const v = Number(slice.value) || 0;
        const next = acc + v * Math.PI * 2;
        const d = piePath(center, center, radius, acc, next);
        acc = next;
        const pct = Number.isFinite(v) ? `${(100 * v).toFixed(2)}%` : 'n/a';
        const title = `${slice.tooltip || slice.label}: ${pct}`;
        return (
          <path
            key={`${slice.label}:${idx}`}
            d={d}
            fill={slice.color || pieColor(idx)}
            onClick={onSliceClick ? () => onSliceClick(slice, idx) : undefined}
            className={onSliceClick ? 'cursor-pointer' : undefined}
          >
            <title>{title}</title>
          </path>
        );
      })}
      <circle cx={center} cy={center} r={radius * 0.35} fill="#0b1220" />
    </svg>
  );
}

const EDGE_LINK_TAG = 'phase-delta-commitment-edge-link';

// Custom Mol* transform that builds a mesh of cylinders (one per edge link).
// This avoids the "distance measurement" machinery which can be surprisingly expensive.
const EdgeLinksShape3D = PluginStateTransform.BuiltIn({
  name: 'phase-edge-links-shape-3d',
  display: { name: 'Edge Links' },
  from: PluginStateObject.Root,
  to: PluginStateObject.Shape.Provider,
  params: {
    payload: PD.Value({ links: [], colors: [], radii: [], labels: [] }, { isHidden: true }),
  },
})({
  canAutoUpdate() {
    return true;
  },
  apply({ params }) {
    return Task.create('Edge Links Shape', async () => {
      const provider = new PluginStateObject.Shape.Provider(
        {
          label: 'Edge Links',
          data: params.payload,
          params: Mesh.Params,
          getShape: (_, data, __, prev) => {
            const links = Array.isArray(data?.links) ? data.links : [];
            const colors = Array.isArray(data?.colors) ? data.colors : [];
            const radii = Array.isArray(data?.radii) ? data.radii : [];
            const labels = Array.isArray(data?.labels) ? data.labels : [];

            const meshState = MeshBuilder.createState(Math.max(256, links.length * 256), 128, prev?.geometry);
            const a = Vec3();
            const b = Vec3();
            for (let i = 0; i < links.length; i += 1) {
              const link = links[i];
              if (!link || !Array.isArray(link.a) || !Array.isArray(link.b)) continue;
              if (link.a.length < 3 || link.b.length < 3) continue;
              Vec3.set(a, Number(link.a[0]), Number(link.a[1]), Number(link.a[2]));
              Vec3.set(b, Number(link.b[0]), Number(link.b[1]), Number(link.b[2]));
              if (!Number.isFinite(a[0]) || !Number.isFinite(b[0])) continue;

              meshState.currentGroup = i;
              const r = Number(radii[i]);
              const radius = Number.isFinite(r) ? r : 0.12;
              addCylinder(meshState, a, b, 1, {
                radiusTop: radius,
                radiusBottom: radius,
                radialSegments: 18,
                topCap: true,
                bottomCap: true,
              });
            }
            const mesh = MeshBuilder.getMesh(meshState);
            return Shape.create(
              'EdgeLinks',
              data,
              mesh,
              (groupId) => colors[groupId] ?? 0xffffff,
              () => 1,
              (groupId) => labels[groupId] ?? `edge_${groupId}`,
              undefined,
              links.length
            );
          },
          geometryUtils: Mesh.Utils,
        },
        { label: 'Edge Links' }
      );
      return provider;
    });
  },
});

export default function DeltaCommitment3DPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();

  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  // Single base component; we overpaint it instead of stacking duplicate cartoons (which causes z-fighting).
  // Store the component's state-tree ref so we can re-find the corresponding StructureComponentRef wrapper.
  const baseComponentRef = useRef(null); // string | null

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [clusterInfo, setClusterInfo] = useState(null);
  const [clusterInfoError, setClusterInfoError] = useState(null);
  const [loadedStructureStateId, setLoadedStructureStateId] = useState('');

  const [modelAId, setModelAId] = useState('');
  const [modelBId, setModelBId] = useState('');

  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [keepInvalid, setKeepInvalid] = useState(false);
  const dropInvalid = !keepInvalid;

  // Note: commitment analysis stores all residues; filtering is done in visualization (not here).

  const [analyses, setAnalyses] = useState([]);
  const [analysesError, setAnalysesError] = useState(null);

  const [analysisData, setAnalysisData] = useState(null);
  const [analysisDataError, setAnalysisDataError] = useState(null);
  const [analysisDataLoading, setAnalysisDataLoading] = useState(false);

  const [helpOpen, setHelpOpen] = useState(false);
  const [viewerError, setViewerError] = useState(null);
  const [viewerStatus, setViewerStatus] = useState('initializing'); // initializing | ready | error
  const [structureLoading, setStructureLoading] = useState(false);

  const [commitmentRowIndex, setCommitmentRowIndex] = useState(0);
  const [commitmentMode, setCommitmentMode] = useState('prob'); // prob | centered | mu_sigmoid
  const [referenceSampleIds, setReferenceSampleIds] = useState([]); // used for centered mode
  const [edgeSmoothEnabled, setEdgeSmoothEnabled] = useState(false);
  const [edgeSmoothStrength, setEdgeSmoothStrength] = useState(0.75); // 0..1

  // Residue-id mapping between cluster residues and the loaded PDB.
  // In practice, "label" (sequential) is the most robust across PDBs; "auth" depends on PDB numbering.
  const [residueIdMode, setResidueIdMode] = useState('auth'); // label | auth
  const [coloringDebug, setColoringDebug] = useState(null);

  const [showEdgeLinks, setShowEdgeLinks] = useState(false);
  const [maxEdgeLinks, setMaxEdgeLinks] = useState(60);
  const [edgeDebug, setEdgeDebug] = useState(null);
  const [selectedResidueIndex, setSelectedResidueIndex] = useState(0);
  const [residueProfile, setResidueProfile] = useState(null);
  const [residueProfileLoading, setResidueProfileLoading] = useState(false);
  const [residueProfileError, setResidueProfileError] = useState(null);
  const [pieModal, setPieModal] = useState(null);

  const edgeRunIdRef = useRef(0);
  const lociLabelProviderRef = useRef(null);
  const hoverRef = useRef({
    residueIdMode: 'auth',
    commitmentMode: 'prob',
    sampleLabel: '',
    authToQ: new Map(),
    labelSeqToQ: [],
  });

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

  const clusterOptions = useMemo(
    () => (system?.metastable_clusters || []).filter((run) => run.path && run.status !== 'failed'),
    [system]
  );
  const selectedCluster = useMemo(
    () => clusterOptions.find((c) => c.cluster_id === selectedClusterId) || null,
    [clusterOptions, selectedClusterId]
  );

  const pottsModels = useMemo(() => selectedCluster?.potts_models || [], [selectedCluster]);
  const deltaModels = useMemo(() => {
    return pottsModels.filter((m) => {
      const params = m.params || {};
      if (params.fit_mode === 'delta') return true;
      const kind = params.delta_kind || '';
      return typeof kind === 'string' && kind.startsWith('delta_');
    });
  }, [pottsModels]);

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

  const selectedCommitmentMeta = useMemo(() => {
    if (!modelAId || !modelBId) return null;
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
  }, [analyses, modelAId, modelBId, mdLabelMode, dropInvalid]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_commitment' });
      setAnalyses(Array.isArray(data?.analyses) ? data.analyses : []);
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
    }
  }, [projectId, systemId, selectedClusterId]);

  const loadClusterInfo = useCallback(async () => {
    if (!selectedClusterId) return;
    setClusterInfoError(null);
    try {
      const data = await fetchPottsClusterInfo(projectId, systemId, selectedClusterId, {
        modelId: modelAId || undefined,
      });
      setClusterInfo(data);
    } catch (err) {
      setClusterInfoError(err.message || 'Failed to load cluster info.');
      setClusterInfo(null);
    }
  }, [projectId, systemId, selectedClusterId, modelAId]);

  useEffect(() => {
    if (!clusterOptions.length) return;
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[clusterOptions.length - 1].cluster_id);
    }
  }, [clusterOptions, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    setAnalyses([]);
    setAnalysisData(null);
    loadClusterInfo();
    loadAnalyses();
  }, [selectedClusterId, loadClusterInfo, loadAnalyses]);

  useEffect(() => {
    if (!deltaModels.length) {
      setModelAId('');
      setModelBId('');
      return;
    }
    const ids = new Set(deltaModels.map((m) => m.model_id));
    if (!modelAId || !ids.has(modelAId)) setModelAId(deltaModels[0].model_id);
    if (!modelBId || !ids.has(modelBId)) setModelBId(deltaModels[0].model_id);
  }, [deltaModels, modelAId, modelBId]);

  // Mol* init
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
      const timeout = setTimeout(() => {
        if (disposed) return;
        setViewerStatus('error');
        setViewerError('3D viewer initialization timed out.');
      }, 8000);
      try {
        const plugin = await createPluginUI({ target: containerRef.current, render: renderReact18 });
        if (disposed) {
          plugin.dispose?.();
          clearTimeout(timeout);
          return;
        }
        pluginRef.current = plugin;
        setViewerStatus('ready');
      } catch (err) {
        setViewerStatus('error');
        setViewerError('3D viewer initialization failed.');
      } finally {
        clearTimeout(timeout);
      }
    };
    tryInit();
    return () => {
      disposed = true;
      if (rafId) cancelAnimationFrame(rafId);
      if (pluginRef.current) {
        try {
          pluginRef.current.dispose?.();
        } catch (err) {
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
    // Overpaint helpers expect a StructureComponentRef with iterable `representations`.
    if (!Array.isArray(found.representations)) return null;
    return found;
  }, []);

  const clearOverpaint = useCallback(async () => {
    const plugin = pluginRef.current;
    const base = getBaseComponentWrapper();
    if (!plugin || !base) return;
    try {
      await clearStructureOverpaint(plugin, [base], ['cartoon']);
    } catch (err) {
      // best effort
    }
  }, [getBaseComponentWrapper]);

  const ensureBaseComponent = useCallback(async () => {
    const plugin = pluginRef.current;
    if (!plugin) return null;
    const structureCell = plugin.managers.structure.hierarchy.current.structures[0]?.cell;
    if (!structureCell) return null;

    // Remove any preset components/representations so we own the visuals on this page.
    try {
      const roots = plugin.managers.structure.hierarchy.current.structures;
      if (roots?.length) await plugin.managers.structure.component.clear(roots);
    } catch (err) {
      // best effort
    }

    const allExpr = MS.struct.generator.all();
    const baseComponent = await plugin.builders.structure.tryCreateComponentFromExpression(structureCell, allExpr, 'phase-base');
    if (!baseComponent) return null;

    const baseColor = hexToInt('#9ca3af');
    await plugin.builders.structure.representation.addRepresentation(baseComponent, {
      type: 'cartoon',
      color: 'uniform',
      colorParams: { value: baseColor },
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
  }, [projectId, systemId, loadedStructureStateId, ensureBaseComponent, clearOverpaint]);

  useEffect(() => {
    if (viewerStatus !== 'ready') return;
    if (!stateOptions.length) return;
    if (!loadedStructureStateId) {
      const first = stateOptions[0]?.state_id;
      if (first) setLoadedStructureStateId(first);
    }
  }, [viewerStatus, stateOptions, loadedStructureStateId]);

  const loadedOnceRef = useRef(false);
  useEffect(() => {
    if (viewerStatus !== 'ready') return;
    if (loadedOnceRef.current) return;
    if (!loadedStructureStateId) return;
    loadedOnceRef.current = true;
    loadStructure(loadedStructureStateId);
  }, [viewerStatus, loadedStructureStateId, loadStructure]);

  // Load selected analysis payload.
  useEffect(() => {
    const run = async () => {
      setAnalysisDataError(null);
      setAnalysisData(null);
      // If the selection doesn't match an existing analysis, clear the overlay so the user
      // doesn't mistake previous coloring for the new selection.
      if (!selectedCommitmentMeta?.analysis_id) {
        await clearOverpaint();
        return;
      }
      setAnalysisDataLoading(true);
      try {
        await clearOverpaint();
        const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'delta_commitment', selectedCommitmentMeta.analysis_id);
        setAnalysisData(payload);
      } catch (err) {
        setAnalysisDataError(err.message || 'Failed to load analysis.');
      } finally {
        setAnalysisDataLoading(false);
      }
    };
    run();
  }, [projectId, systemId, selectedClusterId, selectedCommitmentMeta, clearOverpaint]);

  const commitmentSampleIds = useMemo(() => {
    const ids = analysisData?.data?.sample_ids;
    return Array.isArray(ids) ? ids.map(String) : [];
  }, [analysisData]);
  const sampleCatalogById = useMemo(() => {
    const out = new Map();
    const samples = Array.isArray(selectedCluster?.samples) ? selectedCluster.samples : [];
    for (const s of samples) {
      const sid = s?.sample_id ? String(s.sample_id) : '';
      if (!sid) continue;
      out.set(sid, {
        name: String(s?.name || sid),
        type: String(s?.type || ''),
      });
    }
    return out;
  }, [selectedCluster]);
  const commitmentLabels = useMemo(() => {
    if (!commitmentSampleIds.length) return [];
    const raw = Array.isArray(analysisData?.data?.sample_labels) ? analysisData.data.sample_labels.map(String) : [];
    return commitmentSampleIds.map((sid, idx) => {
      const item = sampleCatalogById.get(String(sid));
      if (item?.name) return item.name;
      if (idx < raw.length && raw[idx]) return raw[idx];
      return String(sid);
    });
  }, [analysisData, commitmentSampleIds, sampleCatalogById]);
  const commitmentTypes = useMemo(() => {
    if (!commitmentSampleIds.length) return [];
    const raw = Array.isArray(analysisData?.data?.sample_types) ? analysisData.data.sample_types.map(String) : [];
    return commitmentSampleIds.map((sid, idx) => {
      const item = sampleCatalogById.get(String(sid));
      if (item?.type) return item.type;
      return idx < raw.length ? String(raw[idx] || '') : '';
    });
  }, [analysisData, commitmentSampleIds, sampleCatalogById]);
  const missingMdCommitmentSamples = useMemo(() => {
    const samples = Array.isArray(selectedCluster?.samples) ? selectedCluster.samples : [];
    const analyzed = new Set(commitmentSampleIds.map((sid) => String(sid)));
    return samples
      .filter((s) => String(s?.type || '').toLowerCase() === 'md_eval')
      .filter((s) => {
        const sid = s?.sample_id ? String(s.sample_id) : '';
        return sid && !analyzed.has(sid);
      })
      .map((s) => ({
        sample_id: String(s.sample_id),
        name: String(s.name || s.sample_id),
        state_id: s.state_id ? String(s.state_id) : '',
      }));
  }, [selectedCluster, commitmentSampleIds]);

  const commitmentMatrix = useMemo(
    () => (Array.isArray(analysisData?.data?.q_residue_all) ? analysisData.data.q_residue_all : []),
    [analysisData]
  );
  const dhTable = useMemo(() => (Array.isArray(analysisData?.data?.dh) ? analysisData.data.dh : null), [analysisData]);
  const pNode = useMemo(() => (Array.isArray(analysisData?.data?.p_node) ? analysisData.data.p_node : null), [analysisData]);
  const kList = useMemo(() => (Array.isArray(analysisData?.data?.K_list) ? analysisData.data.K_list.map((v) => Number(v)) : null), [analysisData]);
  const qEdgeMatrix = useMemo(() => (Array.isArray(analysisData?.data?.q_edge) ? analysisData.data.q_edge : null), [analysisData]);
  const edgesAll = useMemo(() => (Array.isArray(analysisData?.data?.edges) ? analysisData.data.edges : []), [analysisData]);
  const topEdgeIndices = useMemo(
    () => (Array.isArray(analysisData?.data?.top_edge_indices) ? analysisData.data.top_edge_indices : []),
    [analysisData]
  );
  const dEdge = useMemo(() => (Array.isArray(analysisData?.data?.D_edge) ? analysisData.data.D_edge : null), [analysisData]);
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

  const hasAltCommitmentData = useMemo(() => {
    const okDh = Boolean(dhTable && Array.isArray(dhTable[0]) && dhTable.length > 0);
    const okP = Boolean(pNode && Array.isArray(pNode[0]) && pNode.length > 0);
    return okDh && okP;
  }, [dhTable, pNode]);

  useEffect(() => {
    if (!commitmentLabels.length) return;
    setCommitmentRowIndex((prev) => {
      const idx = Number(prev);
      if (Number.isInteger(idx) && idx >= 0 && idx < commitmentLabels.length) return idx;
      return 0;
    });
  }, [commitmentLabels]);

  // Default reference set for centered mode: use MD samples if present, else the first available sample.
  useEffect(() => {
    if (commitmentMode !== 'centered') return;
    if (referenceSampleIds.length) return;
    if (!commitmentSampleIds.length) return;
    const md = commitmentSampleIds.filter((sid, idx) => {
      const t = (commitmentTypes[idx] || '').toLowerCase();
      return t.includes('md');
    });
    if (md.length) setReferenceSampleIds(md);
    else setReferenceSampleIds([commitmentSampleIds[0]]);
  }, [commitmentMode, referenceSampleIds.length, commitmentSampleIds, commitmentTypes]);

  const centeredCalib = useMemo(() => {
    if (commitmentMode !== 'centered') return null;
    if (!dhTable || !pNode) return null;
    if (!referenceSampleIds.length) return null;
    if (!commitmentSampleIds.length) return null;

    const refIdxs = referenceSampleIds
      .map((sid) => commitmentSampleIds.indexOf(String(sid)))
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

      // Calibrate tie-handling such that the reference ensemble maps exactly to 0.5.
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
  }, [commitmentMode, dhTable, pNode, referenceSampleIds, commitmentSampleIds, kList]);

  const qRowValues = useMemo(() => {
    if (!Array.isArray(commitmentMatrix) || !commitmentMatrix.length) return null;
    const row = commitmentMatrix[commitmentRowIndex];
    if (!Array.isArray(row) || !row.length) return null;

    const N = residueLabels.length;
    const baseQ = () => {
      const q = new Array(N);
      for (let i = 0; i < N; i += 1) q[i] = Number(row[i]);
      return q;
    };
    if (commitmentMode === 'prob') return baseQ();

    // If the analysis was created before we started storing dh/p_node, fall back to the base q.
    if (!dhTable || !pNode) {
      return baseQ();
    }
    const Kmax = Array.isArray(dhTable[0]) ? dhTable[0].length : 0;
    if (Kmax <= 0) return null;

    if (commitmentMode === 'centered') {
      if (!centeredCalib) return null;
      const { thresholds, alphas, eps } = centeredCalib;
      const q = new Array(N);
      for (let i = 0; i < N; i += 1) {
        const dhRow = dhTable?.[i];
        const prow = pNode?.[commitmentRowIndex]?.[i];
        const Ki = kList && Number.isFinite(kList[i]) ? Math.max(0, Math.min(Kmax, Math.floor(kList[i]))) : Kmax;
        if (!Array.isArray(dhRow) || !Array.isArray(prow) || dhRow.length < Ki || prow.length < Ki || Ki <= 0) {
          q[i] = NaN;
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
        q[i] = accBefore + alpha * accAt;
      }
      return q;
    }

    if (commitmentMode === 'mu_sigmoid') {
      const mu = new Array(N);
      for (let i = 0; i < N; i += 1) {
        const dhRow = dhTable?.[i];
        const prow = pNode?.[commitmentRowIndex]?.[i];
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
      if (!finite.length) return mu.map(() => 0.5);
      const med = finite[Math.floor(finite.length / 2)];
      const absDev = finite.map((v) => Math.abs(v - med)).sort((a, b) => a - b);
      const mad = absDev[Math.floor(absDev.length / 2)];
      const scale = mad > 1e-9 ? mad : 1.0;
      return mu.map((m) => sigmoid(-(Number(m) || 0) / scale));
    }

    return null;
  }, [commitmentMatrix, commitmentRowIndex, commitmentMode, dhTable, pNode, centeredCalib, residueLabels, kList]);

  const centeredEdgeRefIdxs = useMemo(() => {
    if (commitmentMode !== 'centered') return [];
    if (!referenceSampleIds.length || !commitmentSampleIds.length) return [];
    return referenceSampleIds
      .map((sid) => commitmentSampleIds.indexOf(String(sid)))
      .filter((i) => Number.isInteger(i) && i >= 0);
  }, [commitmentMode, referenceSampleIds, commitmentSampleIds]);

  const qEdgeRowValues = useMemo(() => {
    if (!qEdgeMatrix || !Array.isArray(qEdgeMatrix[commitmentRowIndex])) return null;
    const row = qEdgeMatrix[commitmentRowIndex].map((v) => Number(v));
    if (commitmentMode !== 'centered') return row;
    if (!centeredEdgeRefIdxs.length) return row;

    const eps = 1e-9;
    const out = new Array(row.length).fill(NaN);
    for (let col = 0; col < row.length; col += 1) {
      const v = Number(row[col]);
      if (!Number.isFinite(v)) continue;
      let before = 0;
      let at = 0;
      let n = 0;
      for (const ridx of centeredEdgeRefIdxs) {
        const rv = Number(qEdgeMatrix?.[ridx]?.[col]);
        if (!Number.isFinite(rv)) continue;
        n += 1;
        if (rv < v - eps) before += 1;
        else if (Math.abs(rv - v) <= eps) at += 1;
      }
      out[col] = n > 0 ? (before + 0.5 * at) / n : v;
    }
    return out;
  }, [qEdgeMatrix, commitmentRowIndex, commitmentMode, centeredEdgeRefIdxs]);

  useEffect(() => {
    setSelectedResidueIndex((prev) => {
      const idx = Number(prev);
      if (Number.isInteger(idx) && idx >= 0 && idx < residueLabels.length) return idx;
      return 0;
    });
  }, [residueLabels.length]);

  const selectedSampleId = useMemo(() => {
    if (!Array.isArray(commitmentSampleIds) || !commitmentSampleIds.length) return '';
    const sid = commitmentSampleIds[commitmentRowIndex];
    return sid ? String(sid) : '';
  }, [commitmentSampleIds, commitmentRowIndex]);

  const selectedSampleLabel = useMemo(() => {
    if (!Array.isArray(commitmentLabels) || !commitmentLabels.length) return '';
    return String(commitmentLabels[commitmentRowIndex] || selectedSampleId || '');
  }, [commitmentLabels, commitmentRowIndex, selectedSampleId]);

  const qByEdgeIndex = useMemo(() => {
    const map = new Map();
    if (!Array.isArray(topEdgeIndices) || !Array.isArray(qEdgeRowValues)) return map;
    for (let col = 0; col < topEdgeIndices.length; col += 1) {
      const eidx = Number(topEdgeIndices[col]);
      if (!Number.isInteger(eidx) || eidx < 0) continue;
      const q = Number(qEdgeRowValues[col]);
      if (Number.isFinite(q)) map.set(eidx, q);
    }
    return map;
  }, [topEdgeIndices, qEdgeRowValues]);

  const selectedEdgeEntries = useMemo(() => {
    if (!Array.isArray(edgesAll)) return [];
    if (!Number.isInteger(selectedResidueIndex) || selectedResidueIndex < 0) return [];
    const out = [];
    for (let eidx = 0; eidx < edgesAll.length; eidx += 1) {
      const edge = edgesAll[eidx];
      if (!Array.isArray(edge) || edge.length < 2) continue;
      const r = Number(edge[0]);
      const s = Number(edge[1]);
      if (!Number.isInteger(r) || !Number.isInteger(s)) continue;
      if (r !== selectedResidueIndex && s !== selectedResidueIndex) continue;
      out.push({
        eidx,
        r,
        s,
        q: Number(qByEdgeIndex.get(eidx)),
        d: Number(Array.isArray(dEdge) ? dEdge[eidx] : NaN),
      });
    }
    out.sort((a, b) => Math.abs(Number(b.d) || 0) - Math.abs(Number(a.d) || 0));
    return out;
  }, [edgesAll, qByEdgeIndex, dEdge, selectedResidueIndex]);

  const qRowValuesEdgeSmoothed = useMemo(() => {
    // Optional visualization: smooth residue colors using edge commitment on top edges.
    if (!edgeSmoothEnabled) return qRowValues;
    if (!Array.isArray(qRowValues) || !qRowValues.length) return qRowValues;
    if (!Array.isArray(qEdgeRowValues) || !qEdgeRowValues.length) return qRowValues;
    if (!Array.isArray(topEdgeIndices) || !topEdgeIndices.length) return qRowValues;
    if (!Array.isArray(edgesAll) || !edgesAll.length) return qRowValues;

    const N = residueLabels.length;
    const strength = clamp(Number(edgeSmoothStrength), 0, 1);
    if (strength <= 0) return qRowValues;

    const rowQe = qEdgeRowValues.map((v) => Number(v));
    const sumW = new Array(N).fill(0);
    const sumWD = new Array(N).fill(0);

    for (let col = 0; col < topEdgeIndices.length && col < rowQe.length; col += 1) {
      const eidx = Number(topEdgeIndices[col]);
      const e = edgesAll[eidx];
      if (!Array.isArray(e) || e.length < 2) continue;
      const r = Number(e[0]);
      const s = Number(e[1]);
      if (!Number.isInteger(r) || !Number.isInteger(s) || r < 0 || s < 0 || r >= N || s >= N) continue;
      const q = rowQe[col];
      if (!Number.isFinite(q)) continue;

      const wRaw = dEdge && Number.isFinite(Number(dEdge[eidx])) ? Math.abs(Number(dEdge[eidx])) : 1.0;
      const w = wRaw > 1e-12 ? wRaw : 1.0;
      const d0 = clamp01(q) - 0.5;
      const qVis = 0.5 + Math.sign(d0) * Math.pow(Math.abs(d0) * 2, 0.65) / 2;
      const d = qVis - 0.5;
      sumW[r] += w;
      sumWD[r] += w * d;
      sumW[s] += w;
      sumWD[s] += w * d;
    }

    const out = new Array(N);
    for (let i = 0; i < N; i += 1) {
      const qi = clamp01(Number(qRowValues[i]));
      const di = qi - 0.5;
      const de = sumW[i] > 0 ? sumWD[i] / sumW[i] : 0;
      const dMix = (1 - strength) * di + strength * de;
      out[i] = 0.5 + dMix;
    }
    return out;
  }, [
    edgeSmoothEnabled,
    edgeSmoothStrength,
    qRowValues,
    qEdgeRowValues,
    topEdgeIndices,
    edgesAll,
    dEdge,
    residueLabels.length,
  ]);

  const coloringPayload = useMemo(() => {
    if (!Array.isArray(qRowValuesEdgeSmoothed) || !qRowValuesEdgeSmoothed.length) return null;

    const residueIdsAuth = [];
    const residueIdsLabel = [];
    const qValuesAuth = [];
    const qValuesLabel = [];
    const residueIdByIndexAuth = new Array(residueLabels.length).fill(null);
    const residueIdByIndexLabel = new Array(residueLabels.length).fill(null);
    const singleClusterByIndex = new Array(residueLabels.length).fill(false);
    const singleClusterAuth = [];
    const singleClusterLabel = [];

    // Color all residues; filtering is a visualization concern.
    for (let ridx = 0; ridx < residueLabels.length; ridx += 1) {
      const q = Number(qRowValuesEdgeSmoothed[ridx]);
      const isSingle = Boolean(singleClusterByResidue[ridx]);
      const label = residueLabels[ridx];
      const canonicalAuth = parseResidueId(label);
      const auth = Number.isFinite(canonicalAuth)
        ? Number(canonicalAuth) - Number(loadedStateResidShift || 0)
        : null;
      if (auth !== null) {
        residueIdsAuth.push(auth);
        qValuesAuth.push(Number.isFinite(q) ? q : NaN);
        singleClusterAuth.push(isSingle);
        residueIdByIndexAuth[ridx] = auth;
      }
      residueIdsLabel.push(ridx + 1); // Mol* label_seq_id is 1-based
      qValuesLabel.push(Number.isFinite(q) ? q : NaN);
      singleClusterLabel.push(isSingle);
      residueIdByIndexLabel[ridx] = ridx + 1;
      singleClusterByIndex[ridx] = isSingle;
    }
    return {
      residueIdsAuth,
      qValuesAuth,
      residueIdsLabel,
      qValuesLabel,
      residueIdByIndexAuth,
      residueIdByIndexLabel,
      singleClusterByIndex,
      singleClusterAuth,
      singleClusterLabel,
    };
  }, [qRowValuesEdgeSmoothed, residueLabels, singleClusterByResidue, loadedStateResidShift]);

  const residueIndexByAuth = useMemo(() => {
    const map = new Map();
    const byIdx = coloringPayload?.residueIdByIndexAuth || [];
    for (let i = 0; i < byIdx.length; i += 1) {
      const auth = Number(byIdx[i]);
      if (Number.isFinite(auth)) map.set(auth, i);
    }
    return map;
  }, [coloringPayload]);

  // Register a Mol* hover label provider to show the per-residue commitment being visualized.
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

        let q = NaN;
        if (h.residueIdMode === 'auth') {
          q = h.authToQ.get(auth);
        } else {
          const idx = Number.isFinite(labelSeq) ? Math.floor(labelSeq) - 1 : -1;
          if (idx >= 0 && idx < h.labelSeqToQ.length) q = h.labelSeqToQ[idx];
        }

        if (!Number.isFinite(q)) return `Commitment q: n/a`;
        const mode = h.commitmentMode || 'prob';
        const sample = h.sampleLabel ? ` · ${h.sampleLabel}` : '';
        return `Commitment q: ${q.toFixed(3)} · mode: ${mode}${sample}`;
      },
      group: (label) => `phase-commitment:${label}`,
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
  }, [viewerStatus]);

  useEffect(() => {
    const authToQ = new Map();
    const labelSeqToQ = [];
    if (coloringPayload) {
      const { residueIdsAuth, qValuesAuth, residueIdsLabel, qValuesLabel } = coloringPayload;
      for (let i = 0; i < residueIdsAuth.length; i += 1) authToQ.set(Number(residueIdsAuth[i]), Number(qValuesAuth[i]));
      for (let i = 0; i < residueIdsLabel.length; i += 1) {
        const rid = Number(residueIdsLabel[i]); // 1-based
        const idx = Math.floor(rid) - 1;
        if (idx >= 0) labelSeqToQ[idx] = Number(qValuesLabel[i]);
      }
    }
    hoverRef.current = {
      residueIdMode,
      commitmentMode,
      sampleLabel: commitmentLabels?.[commitmentRowIndex] ? String(commitmentLabels[commitmentRowIndex]) : '',
      authToQ,
      labelSeqToQ,
    };
  }, [coloringPayload, residueIdMode, commitmentMode, commitmentLabels, commitmentRowIndex]);

  // Allow residue selection directly from 3D clicks.
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
      if (Number.isInteger(idx) && idx >= 0 && idx < residueLabels.length) {
        setSelectedResidueIndex(idx);
      }
    });
    return () => {
      try {
        sub?.unsubscribe?.();
      } catch {
        // ignore
      }
    };
  }, [viewerStatus, residueIdMode, residueIndexByAuth, residueLabels.length]);

  const applyColoring = useCallback(async () => {
    if (viewerStatus !== 'ready') {
      setColoringDebug({ status: 'skip', reason: 'viewer-not-ready' });
      return;
    }
    if (structureLoading) {
      setColoringDebug({ status: 'skip', reason: 'structure-loading' });
      return;
    }
    if (!analysisData) {
      setColoringDebug({ status: 'skip', reason: 'no-analysis-data' });
      return;
    }
    if (!coloringPayload) {
      setColoringDebug({ status: 'skip', reason: 'no-coloring-payload' });
      return;
    }
    const plugin = pluginRef.current;
    const base = getBaseComponentWrapper();
    if (!plugin || !base) {
      setColoringDebug({ status: 'skip', reason: 'no-base-component' });
      return;
    }

    await clearOverpaint();

    const { residueIdsAuth, qValuesAuth, residueIdsLabel, qValuesLabel } = coloringPayload;
    const useAuth = residueIdMode === 'auth';
    const residueIds = useAuth ? residueIdsAuth : residueIdsLabel;
    const qValues = useAuth ? qValuesAuth : qValuesLabel;
    const singleFlags = useAuth ? coloringPayload.singleClusterAuth : coloringPayload.singleClusterLabel;
    const prop = useAuth ? 'auth' : 'label';
    if (!residueIds.length) {
      setColoringDebug({ status: 'skip', reason: 'no-residue-ids-for-mode', mode: residueIdMode });
      return;
    }

    const finiteQ = qValues.filter((v) => Number.isFinite(v));
    const qMin = finiteQ.length ? Math.min(...finiteQ) : NaN;
    const qMax = finiteQ.length ? Math.max(...finiteQ) : NaN;
    const qMean = finiteQ.length ? finiteQ.reduce((a, b) => a + b, 0) / finiteQ.length : NaN;

    const bins = 21;
    const bucket = Array.from({ length: bins }, () => []);
    const grayIds = [];
    for (let i = 0; i < residueIds.length; i += 1) {
      const q = qValues[i];
      if (!Number.isFinite(q)) continue;
      if (singleFlags?.[i]) {
        grayIds.push(residueIds[i]);
        continue;
      }
      const b = Math.max(0, Math.min(bins - 1, Math.floor(clamp01(q) * (bins - 1) + 1e-9)));
      bucket[b].push(residueIds[i]);
    }

    let layers = 0;
    const selectedElementsByBin = Array.from({ length: bins }, () => 0);
    const rootStructure = plugin.managers.structure.hierarchy.current.structures[0]?.cell?.obj?.data;
    for (let b = 0; b < bins; b += 1) {
      const ids = bucket[b];
      if (!ids.length) continue;
      const qCenter = b / (bins - 1);
      const colorHex = commitmentColor(qCenter);
      const colorValue = hexToInt(colorHex);
      const propFn =
        prop === 'auth'
          ? MS.struct.atomProperty.macromolecular.auth_seq_id()
          : MS.struct.atomProperty.macromolecular.label_seq_id();
      const residueTests =
        ids.length === 1
          ? MS.core.rel.eq([propFn, ids[0]])
          : MS.core.set.has([MS.set(...ids), propFn]);
      const expression = MS.struct.generator.atomGroups({ 'residue-test': residueTests });
      if (rootStructure) {
        const sel = Script.getStructureSelection(expression, rootStructure);
        selectedElementsByBin[b] = StructureSelection.unionStructure(sel).elementCount;
        if (selectedElementsByBin[b] === 0) continue;
      }
      const lociGetter = async (structure) => {
        const sel = Script.getStructureSelection(expression, structure);
        return StructureSelection.toLociWithSourceUnits(sel);
      };
      // eslint-disable-next-line no-await-in-loop
      await setStructureOverpaint(plugin, [base], colorValue, lociGetter, ['cartoon']);
      layers += 1;
    }
    if (grayIds.length) {
      const propFn =
        prop === 'auth'
          ? MS.struct.atomProperty.macromolecular.auth_seq_id()
          : MS.struct.atomProperty.macromolecular.label_seq_id();
      const residueTests =
        grayIds.length === 1
          ? MS.core.rel.eq([propFn, grayIds[0]])
          : MS.core.set.has([MS.set(...grayIds), propFn]);
      const expression = MS.struct.generator.atomGroups({ 'residue-test': residueTests });
      if (rootStructure) {
        const sel = Script.getStructureSelection(expression, rootStructure);
        const el = StructureSelection.unionStructure(sel).elementCount;
        if (el > 0) {
          const lociGetter = async (structure) => {
            const s2 = Script.getStructureSelection(expression, structure);
            return StructureSelection.toLociWithSourceUnits(s2);
          };
          await setStructureOverpaint(plugin, [base], hexToInt('#9ca3af'), lociGetter, ['cartoon']);
          layers += 1;
        }
      } else {
        const lociGetter = async (structure) => {
          const s2 = Script.getStructureSelection(expression, structure);
          return StructureSelection.toLociWithSourceUnits(s2);
        };
        await setStructureOverpaint(plugin, [base], hexToInt('#9ca3af'), lociGetter, ['cartoon']);
        layers += 1;
      }
    }
    setColoringDebug({
      status: 'run',
      prop,
      mode: residueIdMode,
      commitmentMode,
      refCount: commitmentMode === 'centered' ? referenceSampleIds.length : 0,
      residues: residueIds.length,
      qMin,
      qMax,
      qMean,
      bins: bucket.map((x) => x.length),
      singleClusterGray: grayIds.length,
      created: layers,
      note: 'overpaint layers applied',
      selectedElementsByBin,
    });
  }, [
    viewerStatus,
    structureLoading,
    analysisData,
    coloringPayload,
    residueIdMode,
    commitmentMode,
    referenceSampleIds.length,
    clearOverpaint,
    getBaseComponentWrapper,
  ]);

  const clearEdgeLinks = useCallback(async () => {
    const plugin = pluginRef.current;
    if (!plugin) return;
    const state = plugin.state.data;
    const update = state.build();

    // Delete the Shape provider roots; their children (representations) will be removed as well.
    const sel = state.select(StateSelection.Generators.ofType(PluginStateObject.Shape.Provider).withTag(EDGE_LINK_TAG));
    for (const obj of sel) update.delete(obj);

    if (update.editInfo.count === 0) return;
    await PluginCommands.State.Update(plugin, { state, tree: update, options: { doNotLogTiming: true } });
  }, []);

  const applyEdgeLinks = useCallback(async () => {
    if (viewerStatus !== 'ready') return;
    if (structureLoading) return;
    if (!analysisData) return;
    if (!coloringPayload) return;
    const plugin = pluginRef.current;
    if (!plugin) return;

    const runId = (edgeRunIdRef.current += 1);
    setEdgeDebug(null);

    // Always clear first so toggles/row changes are deterministic.
    await clearEdgeLinks();
    if (runId !== edgeRunIdRef.current) return;

    if (!showEdgeLinks) return;
    if (Number(maxEdgeLinks) <= 0) return;
    if (!Array.isArray(qEdgeRowValues) || !qEdgeRowValues.length) return;
    if (!Array.isArray(topEdgeIndices) || !topEdgeIndices.length) return;
    if (!Array.isArray(edgesAll) || !edgesAll.length) return;

    const rootStructure = plugin.managers.structure.hierarchy.current.structures[0]?.cell?.obj?.data;
    if (!rootStructure) return;

    const useAuth = residueIdMode === 'auth';
    const seqProp =
      useAuth ? MS.struct.atomProperty.macromolecular.auth_seq_id() : MS.struct.atomProperty.macromolecular.label_seq_id();
    const atomProp = MS.struct.atomProperty.macromolecular.label_atom_id();

    const residueIdByIndex = useAuth ? coloringPayload.residueIdByIndexAuth : coloringPayload.residueIdByIndexLabel;
    const singleByIndex = Array.isArray(coloringPayload.singleClusterByIndex) ? coloringPayload.singleClusterByIndex : [];
    const getResidueSeqId = (resIdx) => {
      if (!Array.isArray(residueIdByIndex)) return null;
      const v = residueIdByIndex[resIdx];
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    };

    // q_edge can be large (potentially O(n_edges)). Avoid mapping the whole row just to draw a few links.
    const rowQ = qEdgeRowValues;
    const limit = Math.min(Number(maxEdgeLinks) || 0, topEdgeIndices.length);

    const getCA = (rid) => {
      const expr = MS.struct.generator.atomGroups({
        'residue-test': MS.core.rel.eq([seqProp, rid]),
        'atom-test': MS.core.rel.eq([atomProp, MS.core.type.str('CA')]),
      });
      const sel = Script.getStructureSelection(expr, rootStructure);
      const loci = StructureSelection.toLociWithSourceUnits(sel);
      const loc = StructureElement.Loci.getFirstLocation(loci);
      if (!loc) return null;
      const p = Vec3();
      StructureElement.Location.position(p, loc);
      return [p[0], p[1], p[2]];
    };

    // Precompute coords for all residues that appear in the selected edges.
    const residueSeqIds = new Set();
    const edgesPicked = [];
    for (let col = 0; col < limit; col += 1) {
      const eidx = Number(topEdgeIndices[col]);
      const edge = edgesAll[eidx];
      if (!Array.isArray(edge) || edge.length < 2) continue;
      const r = Number(edge[0]);
      const s = Number(edge[1]);
      if (!Number.isInteger(r) || !Number.isInteger(s)) continue;
      const rid = getResidueSeqId(r);
      const sid = getResidueSeqId(s);
      if (rid == null || sid == null) continue;
      edgesPicked.push({ col, eidx, r, s, rid, sid });
      residueSeqIds.add(rid);
      residueSeqIds.add(sid);
    }
    if (!edgesPicked.length) return;

    const coordBySeq = new Map();
    for (const rid of residueSeqIds) {
      if (runId !== edgeRunIdRef.current) return;
      const xyz = getCA(rid);
      if (xyz) coordBySeq.set(rid, xyz);
    }

    // Thickness scaling based on |ΔJ| magnitude (D_edge) for the selected edges.
    let wMax = 0;
    for (const ep of edgesPicked) {
      const wRaw = dEdge && Number.isFinite(Number(dEdge[ep.eidx])) ? Math.abs(Number(dEdge[ep.eidx])) : 0;
      if (wRaw > wMax) wMax = wRaw;
    }
    if (!Number.isFinite(wMax) || wMax <= 0) wMax = 1;

    const links = [];
    const colors = [];
    const radii = [];
    const labels = [];
    const qUsed = [];
    const wUsed = [];
    let grayLinks = 0;
    for (const ep of edgesPicked) {
      if (runId !== edgeRunIdRef.current) return;
      const a = coordBySeq.get(ep.rid);
      const b = coordBySeq.get(ep.sid);
      if (!a || !b) continue;

      const qRaw = Array.isArray(rowQ) ? Number(rowQ[ep.col]) : Number.NaN;
      const q = Number.isFinite(qRaw) ? clamp01(qRaw) : 0.5;
      const isSingleEdge = Boolean(singleByIndex[ep.r]) || Boolean(singleByIndex[ep.s]);
      if (isSingleEdge) {
        colors.push(hexToInt('#9ca3af'));
        grayLinks += 1;
      } else {
        // Boost contrast a bit for edges (q values are often close to 0.5).
        const d = q - 0.5;
        const qVis = 0.5 + Math.sign(d) * Math.pow(Math.abs(d) * 2, 0.65) / 2;
        colors.push(hexToInt(commitmentColor(qVis)));
      }
      qUsed.push(q);

      const wRaw = dEdge && Number.isFinite(Number(dEdge[ep.eidx])) ? Math.abs(Number(dEdge[ep.eidx])) : 0;
      const wNorm = Math.max(0, Math.min(1, wRaw / wMax));
      // Make width variation more obvious.
      radii.push(0.05 + 0.45 * Math.sqrt(wNorm));
      wUsed.push(wRaw);

      const aLabel = residueLabels[ep.r] ?? `res_${ep.r}`;
      const bLabel = residueLabels[ep.s] ?? `res_${ep.s}`;
      labels.push(`${aLabel}–${bLabel} · q=${q.toFixed(3)} · |ΔJ|=${wRaw.toFixed(3)}${isSingleEdge ? ' · single-K gray' : ''}`);
      links.push({ a, b });
    }
    if (!links.length) return;

    const finiteQ = qUsed.filter((v) => Number.isFinite(v));
    const qMin = finiteQ.length ? Math.min(...finiteQ) : NaN;
    const qMax = finiteQ.length ? Math.max(...finiteQ) : NaN;
    const finiteW = wUsed.filter((v) => Number.isFinite(v));
    const wMin = finiteW.length ? Math.min(...finiteW) : NaN;
    const wMaxUsed = finiteW.length ? Math.max(...finiteW) : NaN;
    setEdgeDebug({
      status: 'run',
      mode: residueIdMode,
      requestedMax: Number(maxEdgeLinks) || 0,
      availableTopEdges: topEdgeIndices.length,
      picked: edgesPicked.length,
      residuesWithCoords: coordBySeq.size,
      links: links.length,
      grayLinks,
      qMin,
      qMax,
      wMin,
      wMax: wMaxUsed,
    });

    const payload = { links, colors, radii, labels };
    const state = plugin.state.data;
    const update = state.build();
    const provider = update.toRoot().apply(EdgeLinksShape3D, { payload }, { tags: [EDGE_LINK_TAG] });
    provider.apply(StateTransforms.Representation.ShapeRepresentation3D, { alpha: 1.0, quality: 'low' }, { tags: [EDGE_LINK_TAG] });
    if (runId !== edgeRunIdRef.current) return;
    await PluginCommands.State.Update(plugin, { state, tree: update, options: { doNotLogTiming: true } });
  }, [
    viewerStatus,
    structureLoading,
    analysisData,
    coloringPayload,
    clearEdgeLinks,
    showEdgeLinks,
    maxEdgeLinks,
    qEdgeRowValues,
    topEdgeIndices,
    edgesAll,
    dEdge,
    residueIdMode,
    residueLabels,
  ]);

  useEffect(() => {
    applyColoring();
  }, [applyColoring]);

  useEffect(() => {
    // Ensure recoloring happens when the selected row changes even if memoization keeps callback identity stable.
    applyColoring();
  }, [commitmentRowIndex, residueIdMode, analysisData, viewerStatus, structureLoading, applyColoring]);

  useEffect(() => {
    applyEdgeLinks();
  }, [applyEdgeLinks]);

  useEffect(() => {
    return () => {
      // best-effort cleanup
      clearEdgeLinks();
    };
  }, [clearEdgeLinks]);

  useEffect(() => {
    const run = async () => {
      setResidueProfileError(null);
      setResidueProfile(null);
      if (!selectedClusterId || !selectedSampleId) return;
      if (!Number.isInteger(selectedResidueIndex) || selectedResidueIndex < 0) return;
      setResidueProfileLoading(true);
      try {
        const payload = {
          residue_index: selectedResidueIndex,
          label_mode: mdLabelMode,
          edge_pairs: selectedEdgeEntries.map((e) => [e.r, e.s]),
        };
        const data = await fetchSampleResidueProfile(projectId, systemId, selectedClusterId, selectedSampleId, payload);
        setResidueProfile(data);
      } catch (err) {
        setResidueProfileError(err.message || 'Failed to compute residue profile.');
      } finally {
        setResidueProfileLoading(false);
      }
    };
    run();
  }, [
    projectId,
    systemId,
    selectedClusterId,
    selectedSampleId,
    selectedResidueIndex,
    mdLabelMode,
    selectedEdgeEntries,
  ]);

  const selectedResidueLabel = useMemo(() => {
    if (selectedResidueIndex < 0 || selectedResidueIndex >= residueLabels.length) return '';
    return String(residueLabels[selectedResidueIndex] || `res_${selectedResidueIndex}`);
  }, [selectedResidueIndex, residueLabels]);

  const selectedResidueQ = useMemo(() => {
    const q = Array.isArray(qRowValues) ? Number(qRowValues[selectedResidueIndex]) : NaN;
    return Number.isFinite(q) ? q : NaN;
  }, [qRowValues, selectedResidueIndex]);

  const selectedResidueQVis = useMemo(() => {
    const q = Array.isArray(qRowValuesEdgeSmoothed) ? Number(qRowValuesEdgeSmoothed[selectedResidueIndex]) : NaN;
    return Number.isFinite(q) ? q : selectedResidueQ;
  }, [qRowValuesEdgeSmoothed, selectedResidueIndex, selectedResidueQ]);

  const nodeSlices = useMemo(() => {
    const probs = residueProfile?.node_probs;
    const ids = Array.isArray(residueProfile?.node_cluster_ids) ? residueProfile.node_cluster_ids : null;
    return compressPieSlices(vecToSlices(probs, 'c', ids), 8);
  }, [residueProfile]);

  const edgeProfileByKey = useMemo(() => {
    const map = new Map();
    const rows = Array.isArray(residueProfile?.edge_profiles) ? residueProfile.edge_profiles : [];
    for (const row of rows) {
      const r = Number(row?.r);
      const s = Number(row?.s);
      if (!Number.isInteger(r) || !Number.isInteger(s)) continue;
      const key = `${Math.min(r, s)}-${Math.max(r, s)}`;
      map.set(key, row);
    }
    return map;
  }, [residueProfile]);

  if (loadingSystem) return <Loader message="Loading 3D commitment viewer..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  const missing = Boolean(
    selectedClusterId &&
      modelAId &&
      modelBId &&
      !selectedCommitmentMeta
  );

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Delta Commitment (3D): How To Read It"
        docPath="/docs/delta_commitment_3d_help.md"
        onClose={() => setHelpOpen(false)}
      />

      <div className="flex items-start justify-between gap-3">
        <div>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_eval`)}
            className="text-cyan-400 hover:text-cyan-300 text-sm"
          >
            ← Back to Delta Potts Evaluation
          </button>
          <h1 className="text-2xl font-semibold text-white">Delta Commitment (3D)</h1>
          <p className="text-sm text-gray-400">
            Load a structure and color residues by commitment <code>q_i</code> for a selected ensemble under a fixed model pair (A,B).
            The base cartoon is gray; colored residues are overpainted on top.
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
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-4">
        <aside className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-200">Selection</h2>
              <button
                type="button"
                onClick={() => {
                  loadClusterInfo();
                  loadAnalyses();
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
              <label className="block text-xs text-gray-400 mb-1">Load structure (PDB)</label>
              {!stateOptions.length && <p className="text-xs text-gray-500">No states with PDBs available.</p>}
              {!!stateOptions.length && (
                <div className="flex flex-wrap gap-2">
                  {stateOptions.map((st) => {
                    const sid = st?.state_id || st?.id || '';
                    if (!sid) return null;
                    const label = st?.name || sid;
                    const active = sid === loadedStructureStateId;
                    return (
                      <button
                        key={sid}
                        type="button"
                        onClick={async () => {
                          setLoadedStructureStateId(sid);
                          // Load immediately so the user gets feedback like in other viz pages.
                          await loadStructure(sid);
                        }}
                        className={`text-xs px-3 py-2 rounded-md border ${
                          active ? 'border-cyan-500 text-cyan-200' : 'border-gray-700 text-gray-200 hover:border-gray-500'
                        }`}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Model A</label>
                <select
                  value={modelAId}
                  onChange={(e) => setModelAId(e.target.value)}
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
                <label className="block text-xs text-gray-400 mb-1">Model B</label>
                <select
                  value={modelBId}
                  onChange={(e) => setModelBId(e.target.value)}
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

            <div>
              <label className="block text-xs text-gray-400 mb-1">Color by commitment</label>
              <select
                value={String(commitmentRowIndex)}
                onChange={(e) => setCommitmentRowIndex(Number(e.target.value))}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {commitmentLabels.map((name, idx) => (
                  <option key={`${idx}:${name}`} value={String(idx)}>
                    {commitmentTypes[idx] ? `${name} (${commitmentTypes[idx]})` : name}
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-gray-500 mt-1">
                Colors are derived from the selected commitment mode (see below).
              </p>
              {selectedCommitmentMeta && commitmentSampleIds.length > 0 && missingMdCommitmentSamples.length > 0 && (
                <div className="mt-2 rounded-md border border-yellow-900 bg-yellow-950/20 p-2 text-[11px] text-yellow-200">
                  Commitment rows missing for {missingMdCommitmentSamples.length} MD samples:
                  {' '}
                  {missingMdCommitmentSamples
                    .slice(0, 6)
                    .map((s) => (s.state_id ? `${s.name} (${s.state_id})` : s.name))
                    .join(', ')}
                  {missingMdCommitmentSamples.length > 6 ? ' ...' : ''}.
                  {' '}Run Delta Commitment analysis again and include these samples to append them.
                </div>
              )}
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Residue mapping</label>
                  <select
                    value={residueIdMode}
                    onChange={(e) => setResidueIdMode(e.target.value)}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                  >
                    <option value="label">Sequential (label_seq_id)</option>
                    <option value="auth">PDB numbering (auth_seq_id)</option>
                  </select>
                  <p className="text-[11px] text-gray-500 mt-1">
                    If nothing is colored, use <span className="font-mono">Sequential</span>. <span className="font-mono">PDB numbering</span> requires
                    matching residue numbers in cluster labels (e.g. <span className="font-mono">res_279</span>).
                  </p>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Commitment mode</label>
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
                  <p className="text-[11px] text-gray-500 mt-1">
                    Centered mode calibrates each residue by a reference ensemble so "neutral" residues appear closer to white (0.5).
                  </p>
                  {!hasAltCommitmentData && (
                    <p className="text-[11px] text-yellow-300 mt-1">
                      Centered/Mean require analysis artifacts generated with the latest backend. Re-run commitment analysis for this (A,B) pair.
                    </p>
                  )}
                </div>
              </div>

              {commitmentMode === 'centered' && (
                <div className="mt-3">
                  <label className="block text-xs text-gray-400 mb-1">Reference ensemble(s)</label>
                  <select
                    multiple
                    value={referenceSampleIds}
                    onChange={(e) => {
                      const opts = Array.from(e.target.selectedOptions).map((o) => String(o.value));
                      setReferenceSampleIds(opts);
                    }}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100 h-24"
                  >
                    {commitmentSampleIds.map((sid, idx) => (
                      <option key={`ref:${sid}`} value={sid}>
                        {commitmentTypes[idx] ? `${commitmentLabels[idx] || sid} (${commitmentTypes[idx]})` : commitmentLabels[idx] || sid}
                      </option>
                    ))}
                  </select>
                  <p className="text-[11px] text-gray-500 mt-1">
                    Tip: select all MD ensembles (or any set you treat as a "baseline") to reduce artifacts from mixed-state marginals.
                  </p>
                </div>
              )}

              <div className="mt-3 rounded-md border border-gray-800 bg-gray-950/30 p-2 space-y-2">
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
                    Each residue color is blended with the average commitment of its incident top edges (weighted by |ΔJ|).
                  </p>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
                <div className="flex items-end">
                  <label className="flex items-center gap-2 text-sm text-gray-200">
                    <input
                      type="checkbox"
                      checked={showEdgeLinks}
                      onChange={(e) => setShowEdgeLinks(e.target.checked)}
                      className="h-4 w-4 text-cyan-500 rounded border-gray-700 bg-gray-950"
                    />
                    Show coupling links (top edges)
                  </label>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Max links</label>
                  <input
                    type="number"
                    min={0}
                    max={500}
                    value={maxEdgeLinks}
                    onChange={(e) => setMaxEdgeLinks(Number(e.target.value))}
                    className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
                  />
                </div>
              </div>

              {coloringDebug && (
                <div className="mt-2 rounded-md border border-gray-800 bg-gray-950/40 p-2 text-[11px] text-gray-300">
                  <div>
                    overlay: <span className="font-mono">{coloringDebug.status}</span>{' '}
                    {coloringDebug.reason ? <span className="text-gray-500">({coloringDebug.reason})</span> : null}
                  </div>
                  {coloringDebug.status === 'run' && (
                    <>
                      <div>
                        prop: <span className="font-mono">{coloringDebug.prop}</span> · mode:{' '}
                        <span className="font-mono">{coloringDebug.mode}</span> · residues:{' '}
                        <span className="font-mono">{coloringDebug.residues}</span>
                      </div>
                      {coloringDebug.commitmentMode && (
                        <div>
                          q: <span className="font-mono">{coloringDebug.commitmentMode}</span>
                          {coloringDebug.refCount ? (
                            <>
                              {' '}
                              · ref: <span className="font-mono">{coloringDebug.refCount}</span>
                            </>
                          ) : null}
                        </div>
                      )}
                      {coloringDebug.created !== undefined && (
                        <div>
                          created: <span className="font-mono">{coloringDebug.created}</span>{' '}
                          {coloringDebug.note ? <span className="text-gray-500">({coloringDebug.note})</span> : null}
                        </div>
                      )}
                      {(coloringDebug.qMin !== undefined || coloringDebug.qMax !== undefined) && (
                        <div>
                          q: min <span className="font-mono">{String(coloringDebug.qMin)}</span> · max{' '}
                          <span className="font-mono">{String(coloringDebug.qMax)}</span> · mean{' '}
                          <span className="font-mono">{String(coloringDebug.qMean)}</span>
                        </div>
                      )}
                      <div className="font-mono">bins: {JSON.stringify(coloringDebug.bins)}</div>
                      {coloringDebug.selectedElementsByBin && (
                        <div className="font-mono">elements: {JSON.stringify(coloringDebug.selectedElementsByBin)}</div>
                      )}
                    </>
                  )}
                </div>
              )}

              {edgeDebug && (
                <div className="mt-2 rounded-md border border-gray-800 bg-gray-950/40 p-2 text-[11px] text-gray-300">
                  <div>
                    edges: <span className="font-mono">{edgeDebug.status}</span> · mode{' '}
                    <span className="font-mono">{edgeDebug.mode}</span>
                  </div>
                  <div>
                    requested <span className="font-mono">{edgeDebug.requestedMax}</span> · available{' '}
                    <span className="font-mono">{edgeDebug.availableTopEdges}</span> · parsed{' '}
                    <span className="font-mono">{edgeDebug.picked}</span> · coords{' '}
                    <span className="font-mono">{edgeDebug.residuesWithCoords}</span> · shown{' '}
                    <span className="font-mono">{edgeDebug.links}</span>
                  </div>
                  <div>
                    q: <span className="font-mono">{String(edgeDebug.qMin)}</span>..{' '}
                    <span className="font-mono">{String(edgeDebug.qMax)}</span> · |ΔJ|:{' '}
                    <span className="font-mono">{String(edgeDebug.wMin)}</span>..{' '}
                    <span className="font-mono">{String(edgeDebug.wMax)}</span>
                  </div>
                </div>
              )}
            </div>

            {clusterInfoError && <ErrorMessage message={clusterInfoError} />}
            {analysesError && <ErrorMessage message={analysesError} />}

            {missing && (
              <div className="rounded-md border border-yellow-800 bg-yellow-950/30 p-3 text-sm text-yellow-200 space-y-2">
                <div>No matching commitment analysis found for this (A,B,params) selection.</div>
                <div className="text-[11px] text-yellow-200">
                  Go back to <span className="font-mono">Delta Potts Evaluation</span> and run commitment on at least one sample.
                </div>
              </div>
            )}
          </div>
        </aside>

        <main className="space-y-3">
          {(viewerError || viewerStatus === 'error') && <ErrorMessage message={viewerError || 'Viewer error'} />}
          {analysisDataError && <ErrorMessage message={analysisDataError} />}

          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-gray-200">3D Viewer</h2>
                <p className="text-[11px] text-gray-500">
                  Blue ≈ q→0, white ≈ q≈0.5, red ≈ q→1.
                </p>
              </div>
              <button
                type="button"
                onClick={() => loadStructure()}
                className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Reload structure
              </button>
            </div>
            <p className="mt-2 text-[11px] text-gray-500">
              Note: some browsers/GPUs may print WebGL warnings for Mol* even when rendering is correct. If the viewer stays blank,
              try a different structure/state button, refresh the page, or switch browsers.
            </p>
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
          {!analysisDataLoading && selectedCommitmentMeta && !analysisData && (
            <p className="mt-2 text-sm text-gray-400">Select an analysis to color residues.</p>
          )}
        </div>

        <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-gray-200">Selected Residue Details</h2>
              <p className="text-[11px] text-gray-500">
                Click a residue in 3D or select it here. Distributions are computed from the sample selected in
                <span className="font-mono"> Color by commitment</span>.
              </p>
            </div>
            <div className="w-56">
              <select
                value={String(selectedResidueIndex)}
                onChange={(e) => setSelectedResidueIndex(Number(e.target.value))}
                className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
              >
                {residueLabels.map((name, idx) => (
                  <option key={`res-opt:${idx}:${name}`} value={String(idx)}>
                    {name} (#{idx})
                  </option>
                ))}
              </select>
            </div>
          </div>

          {(residueProfileLoading || !selectedSampleId) && (
            <p className="text-sm text-gray-400">
              {selectedSampleId ? 'Computing residue profile…' : 'Select a sample in Color by commitment.'}
            </p>
          )}
          {residueProfileError && <ErrorMessage message={residueProfileError} />}

          {residueProfile && (
            <div className="space-y-3">
              <div className="grid grid-cols-1 xl:grid-cols-[320px_1fr] gap-3">
                <div className="rounded-md border border-gray-800 bg-gray-950/40 p-3 space-y-2">
                  <p className="text-xs text-gray-400">Node distribution</p>
                  <div className="flex items-start gap-3">
                    <PieChart
                      slices={nodeSlices}
                      size={132}
                      onSliceClick={() =>
                        setPieModal({
                          title: `Node distribution · ${selectedResidueLabel}`,
                          subtitle: selectedSampleLabel || selectedSampleId,
                          slices: nodeSlices,
                        })
                      }
                    />
                    <div className="space-y-2 text-xs text-gray-200 min-w-0">
                      <p>
                        <span className="text-gray-400">Residue:</span> {selectedResidueLabel}
                      </p>
                      <p>
                        <span className="text-gray-400">Sample:</span> {selectedSampleLabel || selectedSampleId}
                      </p>
                      <p>
                        <span className="text-gray-400">Valid frames:</span> {residueProfile.node_valid_count} / {residueProfile.n_frames}
                      </p>
                      <div className="flex items-center gap-2">
                        <span className="text-gray-400">Commitment q:</span>
                        <span
                          className="inline-flex items-center rounded px-2 py-0.5 text-[11px] font-semibold text-black"
                          style={{ backgroundColor: commitmentColor(selectedResidueQVis) }}
                        >
                          {Number.isFinite(selectedResidueQ) ? selectedResidueQ.toFixed(3) : 'n/a'}
                        </span>
                        <span className="text-[11px] text-gray-500">
                          {Number.isFinite(selectedResidueQ)
                            ? selectedResidueQ > 0.5
                              ? 'towards A (red)'
                              : selectedResidueQ < 0.5
                              ? 'towards B (blue)'
                              : 'neutral'
                            : ''}
                        </span>
                      </div>
                      {nodeSlices.length > 0 && (
                        <button
                          type="button"
                          onClick={() =>
                            setPieModal({
                              title: `Node distribution · ${selectedResidueLabel}`,
                              subtitle: selectedSampleLabel || selectedSampleId,
                              slices: nodeSlices,
                            })
                          }
                          className="text-[11px] px-2 py-1 rounded border border-gray-700 text-gray-200 hover:border-gray-500"
                        >
                          Enlarge pie
                        </button>
                      )}
                    </div>
                  </div>
                  {nodeSlices.length > 0 && (
                    <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                      {nodeSlices.map((s) => (
                        <div key={`node-slice:${s.label}`} className="flex items-center gap-1 text-gray-300">
                          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: s.color }} />
                          <span className="truncate">{s.label}</span>
                          <span className="ml-auto text-gray-400">{(100 * s.value).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="rounded-md border border-gray-800 bg-gray-950/40 p-3 space-y-2">
                  <p className="text-xs text-gray-400">
                    Edge distributions for edges containing this residue (from analysis top edges)
                  </p>
                  {selectedEdgeEntries.length === 0 && (
                    <p className="text-sm text-gray-500">No analyzed top edge contains this residue.</p>
                  )}
                  {selectedEdgeEntries.length > 0 && (
                    <div className="grid grid-cols-1 2xl:grid-cols-2 gap-2 max-h-[420px] overflow-auto pr-1">
                      {selectedEdgeEntries.map((edge) => {
                        const key = `${Math.min(edge.r, edge.s)}-${Math.max(edge.r, edge.s)}`;
                        const profile = edgeProfileByKey.get(key);
                        const rLabel = residueLabels[edge.r] || `res_${edge.r}`;
                        const sLabel = residueLabels[edge.s] || `res_${edge.s}`;
                        const slices = compressPieSlices(
                          jointToSlices(
                            profile?.joint_probs,
                            profile?.cluster_ids_r,
                            profile?.cluster_ids_s,
                            rLabel,
                            sLabel
                          ),
                          10
                        );
                        const q = Number(edge.q);
                        return (
                          <div key={`edge-card:${key}:${edge.eidx}`} className="rounded border border-gray-800 bg-gray-900/50 p-2 space-y-2">
                            <div className="flex items-center justify-between gap-2 text-xs">
                              <span className="text-gray-200 truncate">
                                {rLabel} - {sLabel}
                              </span>
                              <span
                                className="inline-flex items-center rounded px-2 py-0.5 text-[11px] font-semibold text-black"
                                style={{ backgroundColor: Number.isFinite(q) ? commitmentColor(q) : '#9ca3af' }}
                              >
                                {Number.isFinite(q) ? q.toFixed(3) : 'n/a'}
                              </span>
                            </div>
                            <div className="flex items-start gap-3">
                              <PieChart
                                slices={slices}
                                size={96}
                                onSliceClick={() =>
                                  setPieModal({
                                    title: `Edge distribution · ${rLabel} - ${sLabel}`,
                                    subtitle: selectedSampleLabel || selectedSampleId,
                                    slices,
                                  })
                                }
                              />
                              <div className="text-[11px] text-gray-400 space-y-1">
                                <div>edge idx: {edge.eidx}</div>
                                <div>|ΔJ|: {Number.isFinite(edge.d) ? Math.abs(edge.d).toFixed(3) : 'n/a'}</div>
                                <div>
                                  valid: {profile?.valid_count ?? 0} / {residueProfile.n_frames}
                                </div>
                                <div>
                                  {Number.isFinite(q)
                                    ? q > 0.5
                                      ? 'towards A (red)'
                                      : q < 0.5
                                      ? 'towards B (blue)'
                                      : 'neutral'
                                    : ''}
                                </div>
                                {slices.length > 0 && (
                                  <button
                                    type="button"
                                    onClick={() =>
                                      setPieModal({
                                        title: `Edge distribution · ${rLabel} - ${sLabel}`,
                                        subtitle: selectedSampleLabel || selectedSampleId,
                                        slices,
                                      })
                                    }
                                    className="text-[11px] px-2 py-1 rounded border border-gray-700 text-gray-200 hover:border-gray-500"
                                  >
                                    Enlarge pie
                                  </button>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
      </div>
      {pieModal && (
        <div className="fixed inset-0 z-40 bg-black/70 flex items-center justify-center p-4">
          <div className="w-full max-w-xl rounded-lg border border-gray-700 bg-gray-900 p-4 space-y-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-white">{pieModal.title}</h3>
                {pieModal.subtitle && <p className="text-xs text-gray-400">{pieModal.subtitle}</p>}
              </div>
              <button
                type="button"
                onClick={() => setPieModal(null)}
                className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-200 hover:border-gray-500"
              >
                Close
              </button>
            </div>
            <div className="flex flex-col md:flex-row items-start gap-4">
              <PieChart slices={pieModal.slices || []} size={280} />
              <div className="flex-1 max-h-72 overflow-auto pr-1 space-y-1 text-xs">
                {(pieModal.slices || []).map((s) => (
                  <div key={`modal-slice:${s.label}`} className="flex items-center gap-2 text-gray-200">
                    <span className="inline-block h-3 w-3 rounded-sm" style={{ backgroundColor: s.color }} />
                    <span className="truncate">{s.tooltip || s.label}</span>
                    <span className="ml-auto text-gray-400">{(100 * (Number(s.value) || 0)).toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
