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

const INTERSECTION_TYPE = 'hamiltonian_spectral_intersection';
const PALETTE = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#a855f7', '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#14b8a6'];

function safeArray(x) { return Array.isArray(x) ? x : []; }
function hexToInt(hex) { return parseInt(String(hex || '#9ca3af').replace('#', ''), 16); }
function parseResid(raw) { const m = String(raw || '').match(/(-?\d+)/); return m ? Number(m[1]) : NaN; }
function pistonColor(id) { return PALETTE[Math.max(0, (Number(id) || 1) - 1) % PALETTE.length]; }
function analysisTitle(meta) {
  if (!meta) return '';
  return `${meta.single_state_name || meta.single_analysis_id} x ${meta.pair_state_a_name || meta.pair_state_a_id} -> ${meta.pair_state_b_name || meta.pair_state_b_id}`;
}
function parsePistonMembers(raw) {
  const first = Array.isArray(raw) ? raw[0] : raw;
  if (!first) return [];
  try { const parsed = JSON.parse(String(first)); return Array.isArray(parsed) ? parsed : []; } catch { return []; }
}

export default function SpectralIntersection3DPage() {
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
  const [analyses, setAnalyses] = useState([]);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [selectedStateId, setSelectedStateId] = useState('');
  const [selectedPistonId, setSelectedPistonId] = useState(0);
  const [residueIdMode, setResidueIdMode] = useState('auth');
  const [viewerStatus, setViewerStatus] = useState('initializing');
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoadingSystem(true);
      try { const payload = await fetchSystem(projectId, systemId); if (!cancelled) setSystem(payload); }
      catch (err) { if (!cancelled) setError(err.message || 'Failed to load system.'); }
      finally { if (!cancelled) setLoadingSystem(false); }
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
    else if (!selectedClusterId || !clusters.some((c) => c.cluster_id === selectedClusterId)) setSelectedClusterId(clusters[0].cluster_id);
  }, [clusters, selectedClusterId, location.search]);

  useEffect(() => {
    if (!states.length || selectedStateId) return;
    const first = states.find((s) => s.pdb_file)?.state_id || states[0]?.state_id;
    if (first) setSelectedStateId(first);
  }, [states, selectedStateId]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    try {
      const payload = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: INTERSECTION_TYPE });
      const arr = safeArray(payload?.analyses);
      setAnalyses(arr);
      setSelectedAnalysisId((prev) => (prev && arr.some((a) => String(a.analysis_id) === prev) ? prev : String(arr[0]?.analysis_id || '')));
    } catch (err) { setError(err.message || 'Failed to load intersections.'); }
  }, [projectId, systemId, selectedClusterId]);
  useEffect(() => { loadAnalyses(); }, [loadAnalyses]);

  useEffect(() => {
    if (!selectedAnalysisId || !selectedClusterId) { setAnalysisData(null); return; }
    fetchClusterAnalysisData(projectId, systemId, selectedClusterId, INTERSECTION_TYPE, selectedAnalysisId)
      .then((payload) => { setAnalysisData(payload); setSelectedPistonId(0); })
      .catch((err) => setError(err.message || 'Failed to load intersection data.'));
  }, [projectId, systemId, selectedClusterId, selectedAnalysisId]);

  useEffect(() => {
    let disposed = false;
    let rafId;
    const init = async () => {
      if (disposed) return;
      if (!containerRef.current) { rafId = requestAnimationFrame(init); return; }
      if (pluginRef.current) return;
      try {
        const plugin = await createPluginUI({ target: containerRef.current, render: renderReact18 });
        if (disposed) { plugin.dispose?.(); return; }
        pluginRef.current = plugin;
        setViewerStatus('ready');
      } catch (err) { setViewerStatus('error'); setError(err.message || '3D viewer initialization failed.'); }
    };
    init();
    return () => { disposed = true; if (rafId) cancelAnimationFrame(rafId); if (pluginRef.current) pluginRef.current.dispose?.(); pluginRef.current = null; };
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
    const base = await plugin.builders.structure.tryCreateComponentFromExpression(structureCell, MS.struct.generator.all(), 'piston-base');
    await plugin.builders.structure.representation.addRepresentation(base, { type: 'cartoon', color: 'uniform', colorParams: { value: hexToInt('#6b7280') } });
    baseComponentRef.current = base.ref;
  }, [projectId, systemId, selectedStateId]);

  useEffect(() => { if (viewerStatus === 'ready' && selectedStateId) loadStructure().catch((err) => setError(err.message || 'Failed to load structure.')); }, [viewerStatus, selectedStateId, loadStructure]);

  const data = analysisData?.data || {};
  const residueKeys = useMemo(() => safeArray(data.residue_keys).map(String), [data.residue_keys]);
  const pistonIds = useMemo(() => safeArray(data.piston_ids).map(Number), [data.piston_ids]);
  const pistonMembers = useMemo(() => parsePistonMembers(data.piston_members_json), [data.piston_members_json]);

  const residueColors = useMemo(() => {
    const colors = new Map();
    pistonIds.forEach((pid, idx) => {
      if (!pid) return;
      if (selectedPistonId && Number(pid) !== Number(selectedPistonId)) return;
      colors.set(idx, pistonColor(pid));
    });
    return colors;
  }, [pistonIds, selectedPistonId]);

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

  if (loadingSystem) return <Loader message="Loading spectral intersection 3D..." />;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-6 overflow-y-auto">
      <HelpDrawer open={helpOpen} onClose={() => setHelpOpen(false)} title="Spectral intersection 3D help" docPath="/docs/spectral_intersection.md" />
      <div className="mx-auto max-w-7xl space-y-4 pb-16">
        <div className="flex items-start justify-between gap-3">
          <div>
            <button type="button" onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/spectral_intersection${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)} className="text-xs text-cyan-300 hover:text-cyan-200">Back to intersection plots</button>
            <h1 className="mt-2 text-2xl font-semibold text-white">Allosteric Pistons 3D</h1>
            <p className="max-w-3xl text-sm text-gray-400">Base protein is monochrome; allosteric piston residues are highlighted with categorical colors by `(Cstruct, Cfunc)` group.</p>
          </div>
          <div className="flex gap-2"><button type="button" onClick={() => setHelpOpen(true)} className="inline-flex items-center gap-2 rounded-md border border-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-800"><CircleHelp className="h-4 w-4" /> Help</button><button type="button" onClick={loadAnalyses} className="inline-flex items-center gap-2 rounded-md border border-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-800"><RefreshCw className="h-4 w-4" /> Refresh</button></div>
        </div>
        {error ? <ErrorMessage message={error} /> : null}
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[360px_minmax(0,1fr)]">
          <aside className="space-y-3 rounded-lg border border-gray-800 bg-gray-900/50 p-4">
            <label className="block text-xs text-gray-400">Cluster<select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="mt-1 w-full rounded border border-gray-700 bg-gray-950 px-2 py-2 text-sm">{clusters.map((c) => <option key={c.cluster_id} value={c.cluster_id}>{c.name || c.cluster_id}</option>)}</select></label>
            <label className="block text-xs text-gray-400">Intersection<select value={selectedAnalysisId} onChange={(e) => setSelectedAnalysisId(e.target.value)} className="mt-1 w-full rounded border border-gray-700 bg-gray-950 px-2 py-2 text-sm">{analyses.map((a) => <option key={a.analysis_id} value={a.analysis_id}>{analysisTitle(a)}</option>)}</select></label>
            <label className="block text-xs text-gray-400">Reference PDB/state<select value={selectedStateId} onChange={(e) => setSelectedStateId(e.target.value)} className="mt-1 w-full rounded border border-gray-700 bg-gray-950 px-2 py-2 text-sm">{states.map((s) => <option key={s.state_id} value={s.state_id}>{s.name || s.state_id}</option>)}</select></label>
            <label className="block text-xs text-gray-400">Residue numbering<select value={residueIdMode} onChange={(e) => setResidueIdMode(e.target.value)} className="mt-1 w-full rounded border border-gray-700 bg-gray-950 px-2 py-2 text-sm"><option value="auth">PDB/auth residue id from residue key</option><option value="label">Sequential label_seq_id</option></select></label>
            <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3 text-xs text-gray-400"><div className="font-semibold text-gray-200">Piston isolation</div><p className="mt-1">Choose `All pistons` or isolate a single piston. Piston colors are categorical and do not encode magnitude.</p></div>
            <div className="space-y-2">
              <button type="button" onClick={() => setSelectedPistonId(0)} className={`w-full rounded-md border px-3 py-2 text-left text-sm ${selectedPistonId === 0 ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-700 bg-gray-950/50 hover:bg-gray-800'}`}>All pistons</button>
              {pistonMembers.map((p) => <button key={p.piston_id} type="button" onClick={() => setSelectedPistonId(Number(p.piston_id))} className={`w-full rounded-md border px-3 py-2 text-left text-sm ${Number(selectedPistonId) === Number(p.piston_id) ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-700 bg-gray-950/50 hover:bg-gray-800'}`}><div className="flex items-center gap-2"><span className="h-3 w-3 rounded-full" style={{ background: pistonColor(p.piston_id) }} />P{p.piston_id} · S{p.structural_community_id}/F{p.functional_community_id} · {p.size} residues</div><div className="mt-1 text-xs text-gray-400">{safeArray(p.residue_keys).slice(0, 8).join(', ')}{safeArray(p.residue_keys).length > 8 ? ' ...' : ''}</div></button>)}
              {!pistonMembers.length ? <p className="text-xs text-gray-500">No piston groups in this analysis.</p> : null}
            </div>
          </aside>
          <section className="relative min-h-[720px] overflow-hidden rounded-lg border border-gray-800 bg-gray-900">
            {viewerStatus === 'initializing' ? <div className="absolute inset-0 z-10 flex items-center justify-center bg-gray-950/80"><Loader message="Initializing Mol*..." /></div> : null}
            <div ref={containerRef} className="h-[760px] w-full" />
          </section>
        </div>
      </div>
    </div>
  );
}
