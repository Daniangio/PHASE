import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { CircleHelp, RefreshCw } from 'lucide-react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Script } from 'molstar/lib/mol-script/script';
import { StructureSelection } from 'molstar/lib/mol-model/structure';
import { clearStructureOverpaint, setStructureOverpaint } from 'molstar/lib/mol-plugin-state/helpers/structure-overpaint';
import 'molstar/build/viewer/molstar.css';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';

function hexToInt(hex) {
  return parseInt(String(hex || '#9ca3af').replace('#', ''), 16);
}

function safeArray(x) {
  return Array.isArray(x) ? x : [];
}

function parseResid(raw) {
  const m = String(raw || '').match(/(-?\d+)/);
  return m ? Number(m[1]) : NaN;
}

function clamp01(x) {
  return Math.max(0, Math.min(1, Number(x) || 0));
}

function mixColor(a, b, t) {
  const ca = [parseInt(a.slice(1, 3), 16), parseInt(a.slice(3, 5), 16), parseInt(a.slice(5, 7), 16)];
  const cb = [parseInt(b.slice(1, 3), 16), parseInt(b.slice(3, 5), 16), parseInt(b.slice(5, 7), 16)];
  const out = ca.map((v, i) => Math.round(v + clamp01(t) * (cb[i] - v)));
  return `#${out.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

function selectedVectorColor(value, maxAbs, pairMode) {
  const t = clamp01(Math.abs(Number(value) || 0) / Math.max(1e-12, Number(maxAbs) || 0));
  if (pairMode) return mixColor('#e5e7eb', Number(value) >= 0 ? '#ef4444' : '#3b82f6', t);
  return mixColor('#e5e7eb', '#16a34a', t);
}

const PALETTE = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#06b6d4', '#f97316', '#ec4899'];

function allVectorColor(component, intensity) {
  return mixColor('#e5e7eb', PALETTE[component % PALETTE.length], intensity);
}

function communityColor(label) {
  const idx = Math.max(0, (Number(label) || 1) - 1);
  return PALETTE[idx % PALETTE.length];
}

function analysisTitle(meta) {
  if (!meta) return '';
  if (meta.mode === 'pair') return `${meta.state_a_name || meta.state_a_id} → ${meta.state_b_name || meta.state_b_id}`;
  return `${meta.state_name || meta.state_id}`;
}

export default function HamiltonianSpectral3DPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  const baseComponentRef = useRef(null);

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [error, setError] = useState(null);
  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [mode, setMode] = useState('single');
  const [spectralView, setSpectralView] = useState('laplacian');
  const [analyses, setAnalyses] = useState([]);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [selectedStateId, setSelectedStateId] = useState('');
  const [componentIndex, setComponentIndex] = useState(0);
  const [vectorMode, setVectorMode] = useState('selected');
  const [residueIdMode, setResidueIdMode] = useState('auth');
  const [viewerStatus, setViewerStatus] = useState('initializing');
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoadingSystem(true);
      try {
        const payload = await fetchSystem(projectId, systemId);
        if (!cancelled) setSystem(payload);
      } catch (err) {
        if (!cancelled) setError(err.message || 'Failed to load system.');
      } finally {
        if (!cancelled) setLoadingSystem(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [projectId, systemId]);

  const clusters = useMemo(() => safeArray(system?.metastable_clusters).filter((c) => c.cluster_id), [system]);
  const states = useMemo(() => {
    const raw = system?.states;
    if (Array.isArray(raw)) return raw;
    if (raw && typeof raw === 'object') return Object.values(raw);
    return [];
  }, [system]);

  useEffect(() => {
    if (!clusters.length) return;
    const qs = new URLSearchParams(location.search || '');
    const requested = String(qs.get('cluster_id') || '').trim();
    if (requested && clusters.some((c) => c.cluster_id === requested)) setSelectedClusterId(requested);
    else if (!selectedClusterId) setSelectedClusterId(clusters[0].cluster_id);
  }, [clusters, selectedClusterId, location.search]);

  useEffect(() => {
    if (!states.length || selectedStateId) return;
    const first = states.find((s) => s.pdb_file)?.state_id || states[0]?.state_id;
    if (first) setSelectedStateId(first);
  }, [states, selectedStateId]);

  const activeAnalysisType = mode === 'pair' ? 'hamiltonian_spectral_pair' : 'hamiltonian_spectral_single';

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    try {
      const payload = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: activeAnalysisType });
      const arr = safeArray(payload?.analyses);
      setAnalyses(arr);
      setSelectedAnalysisId((prev) => (prev && arr.some((a) => String(a.analysis_id) === String(prev)) ? prev : String(arr[0]?.analysis_id || '')));
    } catch (err) {
      setError(err.message || 'Failed to load analyses.');
    }
  }, [projectId, systemId, selectedClusterId, activeAnalysisType]);
  useEffect(() => { loadAnalyses(); }, [loadAnalyses]);

  useEffect(() => {
    if (!selectedAnalysisId || !selectedClusterId) {
      setAnalysisData(null);
      return;
    }
    fetchClusterAnalysisData(projectId, systemId, selectedClusterId, activeAnalysisType, selectedAnalysisId)
      .then((payload) => { setAnalysisData(payload); setComponentIndex(0); })
      .catch((err) => setError(err.message || 'Failed to load spectral analysis data.'));
  }, [projectId, systemId, selectedClusterId, activeAnalysisType, selectedAnalysisId]);

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
      try {
        const plugin = await createPluginUI({ target: containerRef.current, render: renderReact18 });
        if (disposed) { plugin.dispose?.(); return; }
        pluginRef.current = plugin;
        setViewerStatus('ready');
      } catch (err) {
        setViewerStatus('error');
        setError(err.message || '3D viewer initialization failed.');
      }
    };
    init();
    return () => {
      disposed = true;
      if (rafId) cancelAnimationFrame(rafId);
      if (pluginRef.current) pluginRef.current.dispose?.();
      pluginRef.current = null;
    };
  }, []);

  const getBase = useCallback(() => {
    const ref = baseComponentRef.current;
    const roots = pluginRef.current?.managers?.structure?.hierarchy?.current?.structures;
    const comps = roots?.[0]?.components || [];
    return comps.find((c) => c?.cell?.transform?.ref === ref) || null;
  }, []);

  const loadStructure = useCallback(async () => {
    const plugin = pluginRef.current;
    if (!plugin || !selectedStateId) return;
    baseComponentRef.current = null;
    await plugin.clear();
    await plugin.dataTransaction(async () => {
      const url = `/api/v1/projects/${projectId}/systems/${systemId}/structures/${encodeURIComponent(selectedStateId)}`;
      const data = await plugin.builders.data.download({ url: Asset.Url(url), label: selectedStateId }, { state: { isGhost: true } });
      const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
      await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
    });
    const structureCell = plugin.managers.structure.hierarchy.current.structures[0]?.cell;
    const roots = plugin.managers.structure.hierarchy.current.structures;
    if (roots?.length) await plugin.managers.structure.component.clear(roots);
    const base = await plugin.builders.structure.tryCreateComponentFromExpression(structureCell, MS.struct.generator.all(), 'spectral-base');
    await plugin.builders.structure.representation.addRepresentation(base, { type: 'cartoon', color: 'uniform', colorParams: { value: hexToInt('#9ca3af') } });
    baseComponentRef.current = base.ref;
  }, [projectId, systemId, selectedStateId]);

  useEffect(() => {
    if (viewerStatus !== 'ready' || !selectedStateId) return;
    loadStructure().catch((err) => setError(err.message || 'Failed to load structure.'));
  }, [viewerStatus, selectedStateId, loadStructure]);

  const spectral = analysisData?.data || {};
  const residueKeys = useMemo(() => safeArray(spectral.residue_keys).map(String), [spectral.residue_keys]);
  const pairMode = mode === 'pair';
  const hasLaplacian = Array.isArray(spectral.laplacian_top_eigenvectors);
  const viewMode = spectralView === 'laplacian' && hasLaplacian ? 'laplacian' : 'absolute';
  const topVectors = useMemo(() => safeArray(viewMode === 'laplacian' ? spectral.laplacian_top_eigenvectors : spectral.top_eigenvectors), [spectral.laplacian_top_eigenvectors, spectral.top_eigenvectors, viewMode]);
  const topValues = useMemo(() => safeArray(viewMode === 'laplacian' ? spectral.laplacian_top_eigenvalues : spectral.top_eigenvalues).map(Number), [spectral.laplacian_top_eigenvalues, spectral.top_eigenvalues, viewMode]);
  const communityIds = useMemo(() => safeArray(spectral.community_ids).map(Number), [spectral.community_ids]);

  useEffect(() => {
    if (vectorMode === 'communities' && (viewMode !== 'laplacian' || !communityIds.length)) {
      setVectorMode('selected');
    }
  }, [vectorMode, viewMode, communityIds.length]);

  const residueColors = useMemo(() => {
    const colors = new Map();
    if (!residueKeys.length) return colors;
    if (vectorMode === 'communities' && communityIds.length) {
      communityIds.forEach((label, ridx) => colors.set(ridx, communityColor(label)));
      return colors;
    }
    if (!topVectors.length) return colors;
    if (vectorMode === 'selected') {
      const idx = Math.min(componentIndex, topVectors.length - 1);
      const vec = safeArray(topVectors[idx]).map(Number);
      const maxAbs = Math.max(1e-12, ...vec.map((v) => Math.abs(v)));
      vec.forEach((value, ridx) => colors.set(ridx, selectedVectorColor(value, maxAbs, pairMode)));
      return colors;
    }
    const maxComponents = Math.min(topVectors.length, 8);
    const contributions = residueKeys.map(() => ({ component: 0, value: 0 }));
    for (let k = 0; k < maxComponents; k += 1) {
      const vec = safeArray(topVectors[k]).map(Number);
      const w = viewMode === 'laplacian' ? (maxComponents - k) / maxComponents : Math.abs(Number(topValues[k]) || 0);
      vec.forEach((value, ridx) => {
        const c = w * Math.abs(value);
        if (c > contributions[ridx].value) contributions[ridx] = { component: k, value: c };
      });
    }
    const max = Math.max(1e-12, ...contributions.map((x) => x.value));
    contributions.forEach((x, ridx) => colors.set(ridx, allVectorColor(x.component, x.value / max)));
    return colors;
  }, [residueKeys, topVectors, topValues, vectorMode, componentIndex, pairMode, viewMode, communityIds]);

  const applyColoring = useCallback(async () => {
    const plugin = pluginRef.current;
    const base = getBase();
    if (!plugin || !base) return;
    try { await clearStructureOverpaint(plugin, [base], ['cartoon']); } catch { /* noop */ }
    const dataRoot = plugin.managers.structure.hierarchy.current.structures[0]?.cell?.obj?.data;
    if (!dataRoot || residueColors.size === 0) return;
    const groups = new Map();
    for (const [ridx, hex] of residueColors.entries()) {
      const residueId = residueIdMode === 'label' ? Number(ridx) + 1 : parseResid(residueKeys[ridx]);
      if (!Number.isFinite(residueId)) continue;
      if (!groups.has(hex)) groups.set(hex, []);
      groups.get(hex).push(residueId);
    }
    const propFn = residueIdMode === 'label' ? MS.struct.atomProperty.macromolecular.label_seq_id() : MS.struct.atomProperty.macromolecular.auth_seq_id();
    for (const [hex, ids] of groups.entries()) {
      const residueTests = ids.length === 1 ? MS.core.rel.eq([propFn, ids[0]]) : MS.core.set.has([MS.set(...ids), propFn]);
      const expression = MS.struct.generator.atomGroups({ 'residue-test': residueTests });
      const lociGetter = () => StructureSelection.toLociWithSourceUnits(Script.getStructureSelection(expression, dataRoot));
      // eslint-disable-next-line no-await-in-loop
      await setStructureOverpaint(plugin, [base], hexToInt(hex), lociGetter, ['cartoon']);
    }
  }, [getBase, residueColors, residueIdMode, residueKeys]);

  useEffect(() => { applyColoring().catch(() => {}); }, [applyColoring]);

  if (loadingSystem) return <Loader message="Loading Hamiltonian spectral 3D viewer..." />;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-6 overflow-y-auto">
      <HelpDrawer open={helpOpen} onClose={() => setHelpOpen(false)} title="Hamiltonian spectral 3D help" docPath="/docs/hamiltonian_spectral.md" />
      <div className="max-w-7xl mx-auto space-y-4 pb-16">
        <div className="flex items-start justify-between gap-3">
          <div>
            <button type="button" onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/hamiltonian_spectral${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)} className="text-xs text-cyan-300 hover:text-cyan-200">Back to spectral plots</button>
            <h1 className="text-2xl font-semibold text-white mt-2">Hamiltonian Spectra 3D</h1>
            <p className="text-sm text-gray-400 max-w-3xl">Color a reference PDB by sector eigenvector loadings. Single mode maps |v_i|; pair mode can show signed ΔF rewiring or normalized Laplacian allostery loadings.</p>
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={() => setHelpOpen(true)} className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800"><CircleHelp className="h-4 w-4" /> Help</button>
            <button type="button" onClick={loadAnalyses} className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800"><RefreshCw className="h-4 w-4" /> Refresh</button>
            <button type="button" onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/spectral_intersection_3d${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)} className="px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800">Pistons 3D</button>
          </div>
        </div>
        {error ? <ErrorMessage message={error} /> : null}
        <div className="grid grid-cols-1 xl:grid-cols-[340px_minmax(0,1fr)] gap-4">
          <aside className="rounded-lg border border-gray-800 bg-gray-900/50 p-4 space-y-3">
            <label className="block text-xs text-gray-400">Cluster<select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm">{clusters.map((c) => <option key={c.cluster_id} value={c.cluster_id}>{c.name || c.cluster_id}</option>)}</select></label>
            <label className="block text-xs text-gray-400">Mode<select value={mode} onChange={(e) => setMode(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm"><option value="single">Single-state sectors</option><option value="pair">Pair rewiring sectors</option></select></label>
            <label className="block text-xs text-gray-400">Spectral view
              <select
                value={spectralView}
                onChange={(e) => { setSpectralView(e.target.value); setComponentIndex(0); }}
                className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm"
              >
                <option value="laplacian">{pairMode ? 'Differential Laplacian allostery' : 'Single-state Laplacian communities'}</option>
                <option value="absolute">{pairMode ? 'ΔF spectral rewiring' : 'Absolute entropy / Frobenius'}</option>
              </select>
              {spectralView === 'laplacian' && !hasLaplacian ? <span className="mt-1 block text-[11px] text-amber-300">Rerun this spectral analysis to add v3 Laplacian/community fields.</span> : null}
            </label>
            <label className="block text-xs text-gray-400">Analysis<select value={selectedAnalysisId} onChange={(e) => setSelectedAnalysisId(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm">{analyses.map((a) => <option key={a.analysis_id} value={a.analysis_id}>{analysisTitle(a)}</option>)}</select></label>
            <label className="block text-xs text-gray-400">Reference PDB/state<select value={selectedStateId} onChange={(e) => setSelectedStateId(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm">{states.map((s) => <option key={s.state_id} value={s.state_id}>{s.name || s.state_id}</option>)}</select></label>
            <label className="block text-xs text-gray-400">Color mode<select value={vectorMode} onChange={(e) => setVectorMode(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm"><option value="selected">Selected eigenvector</option><option value="all">Dominant among first 8 vectors</option>{viewMode === 'laplacian' && communityIds.length ? <option value="communities">DADApy communities</option> : null}</select></label>
            {vectorMode === 'selected' ? <label className="block text-xs text-gray-400">Eigenvector<select value={componentIndex} onChange={(e) => setComponentIndex(Number(e.target.value))} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm">{topValues.map((v, idx) => <option key={idx} value={idx}>{viewMode === 'laplacian' ? 'Fiedler ' : 'v'}{idx + 1} · λ={Number(v).toFixed(4)}</option>)}</select></label> : null}
            <label className="block text-xs text-gray-400">Residue numbering<select value={residueIdMode} onChange={(e) => setResidueIdMode(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm"><option value="auth">PDB/auth residue id from residue key</option><option value="label">Sequential label_seq_id</option></select></label>
            <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3 text-xs text-gray-400 space-y-1">
              <div className="text-gray-200 font-semibold">Interpretation</div>
              <p>{vectorMode === 'communities' ? 'Community mode uses categorical colors from DADApy density peak clustering in cosine-distance Laplacian spectral space.' : (pairMode ? 'Pair mode: red/blue are opposite signed sectors in ΔF view. Laplacian view uses normalized Fiedler-like loadings.' : 'Single mode: green saturation is proportional to |v_i| in Frobenius view; Laplacian view identifies structural modules.')}</p>
              <p>All-vectors mode assigns each residue the color of its dominant component. Community colors are labels, not scalar intensity.</p>
            </div>
          </aside>
          <section className="rounded-lg border border-gray-800 bg-gray-900 overflow-hidden min-h-[720px] relative">
            {viewerStatus === 'initializing' ? <div className="absolute inset-0 z-10 flex items-center justify-center bg-gray-950/80"><Loader message="Initializing Mol*..." /></div> : null}
            <div ref={containerRef} className="h-[760px] w-full" />
          </section>
        </div>
      </div>
    </div>
  );
}
