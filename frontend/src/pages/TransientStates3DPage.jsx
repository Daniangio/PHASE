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

function scoreColor(x, max) {
  const t = Math.max(0, Math.min(1, Number(max) > 0 ? Number(x) / Number(max) : 0));
  const r = Math.round(34 + t * (239 - 34));
  const g = Math.round(197 + t * (68 - 197));
  const b = Math.round(94 + t * (68 - 94));
  return `#${[r, g, b].map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

function parseResid(raw) {
  const m = String(raw || '').match(/(-?\d+)/);
  return m ? Number(m[1]) : NaN;
}

export default function TransientStates3DPage() {
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
  const [useTrajectoryFrame, setUseTrajectoryFrame] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);
  const [selectedSample, setSelectedSample] = useState('all');
  const [maxClusters, setMaxClusters] = useState(6);
  const [residueIdMode, setResidueIdMode] = useState('auth');
  const [viewerStatus, setViewerStatus] = useState('initializing');
  const [helpOpen, setHelpOpen] = useState(false);
  const [framePanelOpen, setFramePanelOpen] = useState(false);

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

  const clusters = useMemo(() => (system?.metastable_clusters || []).filter((c) => c.cluster_id), [system]);
  const states = useMemo(() => {
    const raw = system?.states;
    if (Array.isArray(raw)) return raw;
    if (raw && typeof raw === 'object') return Object.values(raw);
    return [];
  }, [system]);
  const selectedState = useMemo(
    () => states.find((s) => String(s.state_id) === String(selectedStateId)) || null,
    [states, selectedStateId]
  );
  const selectedStateFrameCount = Math.max(1, Number(selectedState?.n_frames || 1));

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

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    const payload = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'transient_states' });
    const arr = payload?.analyses || [];
    setAnalyses(arr);
    setSelectedAnalysisId((prev) => (prev && arr.some((a) => a.analysis_id === prev) ? prev : String(arr[0]?.analysis_id || '')));
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => { loadAnalyses().catch((err) => setError(err.message || 'Failed to load analyses.')); }, [loadAnalyses]);

  useEffect(() => {
    if (!selectedAnalysisId || !selectedClusterId) return;
    fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'transient_states', selectedAnalysisId)
      .then(setAnalysisData)
      .catch((err) => setError(err.message || 'Failed to load analysis data.'));
  }, [projectId, systemId, selectedClusterId, selectedAnalysisId]);

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
      const url = useTrajectoryFrame
        ? `/api/v1/projects/${projectId}/systems/${systemId}/states/${encodeURIComponent(selectedStateId)}/trajectory/frame?frame=${encodeURIComponent(frameIndex)}`
        : `/api/v1/projects/${projectId}/systems/${systemId}/structures/${encodeURIComponent(selectedStateId)}`;
      const data = await plugin.builders.data.download({ url: Asset.Url(url), label: selectedStateId }, { state: { isGhost: true } });
      const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
      await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
    });
    const structureCell = plugin.managers.structure.hierarchy.current.structures[0]?.cell;
    const roots = plugin.managers.structure.hierarchy.current.structures;
    if (roots?.length) await plugin.managers.structure.component.clear(roots);
    const base = await plugin.builders.structure.tryCreateComponentFromExpression(structureCell, MS.struct.generator.all(), 'transient-base');
    await plugin.builders.structure.representation.addRepresentation(base, { type: 'cartoon', color: 'uniform', colorParams: { value: hexToInt('#9ca3af') } });
    baseComponentRef.current = base.ref;
  }, [projectId, systemId, selectedStateId, useTrajectoryFrame, frameIndex]);

  useEffect(() => {
    if (viewerStatus !== 'ready' || !selectedStateId) return;
    loadStructure().catch((err) => setError(err.message || 'Failed to load structure.'));
  }, [viewerStatus, selectedStateId, loadStructure]);

  const residueScores = useMemo(() => {
    const d = analysisData?.data || {};
    const n = Array.isArray(d.node_score) ? d.node_score.length : 0;
    const kList = Array.isArray(d.K_list) ? d.K_list.map((x) => Number(x)) : [];
    const labels = Array.isArray(d.residue_labels) ? d.residue_labels : [];
    const map = new Map();
    for (let i = 0; i < n; i += 1) {
      const sampleIndex = Number(d.node_sample_index?.[i]);
      if (selectedSample !== 'all' && String(sampleIndex) !== String(selectedSample)) continue;
      const ridx = Number(d.node_residue_index?.[i]);
      const K = Number(kList[ridx]);
      if (Number(maxClusters) > 0 && Number.isFinite(K) && K > Number(maxClusters)) continue;
      const score = Number(d.node_score?.[i]);
      if (!Number.isFinite(ridx) || !Number.isFinite(score)) continue;
      const current = map.get(ridx) || { score: 0, hits: 0, label: labels[ridx] || String(ridx), K };
      current.score = Math.max(current.score, score);
      current.hits += 1;
      map.set(ridx, current);
    }
    return map;
  }, [analysisData, selectedSample, maxClusters]);

  const sampleLabels = useMemo(() => analysisData?.data?.sample_labels || [], [analysisData]);
  const maxScore = useMemo(() => Math.max(0, ...Array.from(residueScores.values()).map((x) => Number(x.score) || 0)), [residueScores]);

  const applyColoring = useCallback(async () => {
    const plugin = pluginRef.current;
    const base = getBase();
    if (!plugin || !base) return;
    try { await clearStructureOverpaint(plugin, [base], ['cartoon']); } catch { /* noop */ }
    const dataRoot = plugin.managers.structure.hierarchy.current.structures[0]?.cell?.obj?.data;
    if (!dataRoot || residueScores.size === 0) return;
    const groups = new Map();
    for (const [ridx, info] of residueScores.entries()) {
      const residueId = residueIdMode === 'label' ? Number(ridx) + 1 : parseResid(info.label);
      if (!Number.isFinite(residueId)) continue;
      const color = scoreColor(info.score, maxScore);
      if (!groups.has(color)) groups.set(color, []);
      groups.get(color).push(residueId);
    }
    const propFn = residueIdMode === 'label' ? MS.struct.atomProperty.macromolecular.label_seq_id() : MS.struct.atomProperty.macromolecular.auth_seq_id();
    for (const [hex, ids] of groups.entries()) {
      const residueTests = ids.length === 1 ? MS.core.rel.eq([propFn, ids[0]]) : MS.core.set.has([MS.set(...ids), propFn]);
      const expression = MS.struct.generator.atomGroups({ 'residue-test': residueTests });
      const lociGetter = () => StructureSelection.toLociWithSourceUnits(Script.getStructureSelection(expression, dataRoot));
      // eslint-disable-next-line no-await-in-loop
      await setStructureOverpaint(plugin, [base], hexToInt(hex), lociGetter, ['cartoon']);
    }
  }, [getBase, residueScores, residueIdMode, maxScore]);

  useEffect(() => { applyColoring().catch(() => {}); }, [applyColoring]);

  if (loadingSystem) return <Loader message="Loading transient-state 3D viewer..." />;

  return (
    <div className="space-y-4">
      <HelpDrawer open={helpOpen} onClose={() => setHelpOpen(false)} title="Transient-State Analysis Help" docPath="/docs/transient_states_help.md" />
      <div className="flex items-start justify-between gap-3">
        <div>
          <button type="button" onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/transient_states${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)} className="text-xs text-cyan-300 hover:text-cyan-200">← Back to transient table</button>
          <h1 className="text-2xl font-semibold text-white mt-2">Transient-State 3D Viewer</h1>
          <p className="text-sm text-gray-400">Residues are colored by the strongest transient-state score after the cluster-count flexibility filter.</p>
        </div>
        <div className="flex gap-2"><button type="button" onClick={() => setHelpOpen(true)} className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800"><CircleHelp className="h-4 w-4" /> Help</button><button type="button" onClick={loadAnalyses} className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800"><RefreshCw className="h-4 w-4" /> Refresh</button></div>
      </div>
      {error && <ErrorMessage message={error} />}
      <div className="grid grid-cols-1 xl:grid-cols-[340px_minmax(0,1fr)] gap-4">
        <aside className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
          <label className="block text-xs text-gray-400">Cluster<select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm">{clusters.map((c) => <option key={c.cluster_id} value={c.cluster_id}>{c.name || c.cluster_id}</option>)}</select></label>
          <label className="block text-xs text-gray-400">Analysis<select value={selectedAnalysisId} onChange={(e) => setSelectedAnalysisId(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm">{analyses.map((a) => <option key={a.analysis_id} value={a.analysis_id}>{String(a.updated_at || a.created_at || a.analysis_id).slice(0, 24)}</option>)}</select></label>
          <label className="block text-xs text-gray-400">Reference PDB/state<select value={selectedStateId} onChange={(e) => setSelectedStateId(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm">{states.map((s) => <option key={s.state_id} value={s.state_id}>{s.name || s.state_id}</option>)}</select></label>
          <div className="rounded-md border border-gray-800 bg-gray-950/30 p-2 space-y-2">
            <label className="flex items-center gap-2 text-xs text-gray-300">
              <input type="checkbox" checked={useTrajectoryFrame} onChange={(e) => setUseTrajectoryFrame(e.target.checked)} />
              Load frame from stored trajectory
            </label>
            <label className="block text-xs text-gray-400">
              Frame index
              <input type="number" min="0" value={frameIndex} onChange={(e) => setFrameIndex(e.target.value)} disabled={!useTrajectoryFrame} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-1.5 text-sm disabled:opacity-50" />
            </label>
            <button type="button" onClick={loadStructure} className="w-full rounded-md border border-gray-700 px-2 py-1.5 text-xs text-gray-200 hover:bg-gray-800">
              Load structure/frame
            </button>
            <button type="button" onClick={() => setFramePanelOpen(true)} className="w-full rounded-md border border-cyan-700 px-2 py-1.5 text-xs text-cyan-200 hover:bg-cyan-950/30">
              Open trajectory frame panel
            </button>
          </div>
          <label className="block text-xs text-gray-400">Sample filter<select value={selectedSample} onChange={(e) => setSelectedSample(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm"><option value="all">all samples</option>{sampleLabels.map((s, i) => <option key={i} value={i}>{s}</option>)}</select></label>
          <label className="block text-xs text-gray-400">Max residue clusters<input type="number" min="0" value={maxClusters} onChange={(e) => setMaxClusters(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm" /></label>
          <label className="block text-xs text-gray-400">Residue numbering<select value={residueIdMode} onChange={(e) => setResidueIdMode(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm"><option value="auth">PDB/auth numbering</option><option value="label">sequential label_seq_id</option></select></label>
          <div className="text-xs text-gray-400">Colored residues: {residueScores.size}; max score: {maxScore.toFixed(2)}</div>
          <div className="rounded-md border border-gray-800 bg-gray-950/30 p-2 text-[11px] text-gray-400 space-y-1">
            <div><span className="inline-block w-3 h-3 rounded-sm bg-[#22c55e] mr-1 align-middle" />Low transient score</div>
            <div><span className="inline-block w-3 h-3 rounded-sm bg-[#ef4444] mr-1 align-middle" />High transient score</div>
            <p>Gray residues are not selected by the current sample/flexibility filters.</p>
          </div>
        </aside>
        <section className="rounded-lg border border-gray-800 bg-gray-900/40 overflow-hidden min-h-[720px]"><div ref={containerRef} className="w-full h-[720px] bg-black" />{viewerStatus === 'initializing' && <div className="p-3 text-sm text-gray-400">Initializing viewer...</div>}</section>
      </div>
      {framePanelOpen && (
        <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4">
          <div className="w-full max-w-xl rounded-lg border border-gray-700 bg-gray-900 shadow-xl">
            <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
              <div>
                <h2 className="text-sm font-semibold text-white">Load State Structure Or Trajectory Frame</h2>
                <p className="text-xs text-gray-500">Use a static reference PDB or extract one stored trajectory frame as a PDB instance.</p>
              </div>
              <button type="button" onClick={() => setFramePanelOpen(false)} className="text-gray-400 hover:text-white">×</button>
            </div>
            <div className="p-4 space-y-3">
              <label className="block text-xs text-gray-400">
                State
                <select value={selectedStateId} onChange={(e) => setSelectedStateId(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
                  {states.map((s) => (
                    <option key={s.state_id} value={s.state_id}>
                      {s.name || s.state_id} {s.trajectory_file ? `(trajectory, ${s.n_frames || '?'} frames)` : '(PDB only)'}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-200">
                <input type="checkbox" checked={useTrajectoryFrame} onChange={(e) => setUseTrajectoryFrame(e.target.checked)} />
                Load from associated trajectory
              </label>
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs text-gray-400">
                  <span>Frame index</span>
                  <span>{frameIndex} / {Math.max(0, selectedStateFrameCount - 1)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max={Math.max(0, selectedStateFrameCount - 1)}
                  value={Math.max(0, Math.min(Number(frameIndex) || 0, Math.max(0, selectedStateFrameCount - 1)))}
                  onChange={(e) => setFrameIndex(Number(e.target.value))}
                  disabled={!useTrajectoryFrame}
                  className="w-full disabled:opacity-50"
                />
                <input
                  type="number"
                  min="0"
                  max={Math.max(0, selectedStateFrameCount - 1)}
                  value={frameIndex}
                  onChange={(e) => setFrameIndex(e.target.value)}
                  disabled={!useTrajectoryFrame}
                  className="w-full rounded bg-gray-950 border border-gray-700 px-2 py-1.5 text-sm text-gray-100 disabled:opacity-50"
                />
              </div>
              {!selectedState?.trajectory_file && (
                <p className="text-xs text-amber-300">This state has no stored trajectory. Use the System page → States → Show details to upload one.</p>
              )}
              <button
                type="button"
                onClick={async () => {
                  await loadStructure();
                  setFramePanelOpen(false);
                }}
                className="w-full rounded-md bg-cyan-600 px-3 py-2 text-sm font-semibold text-white hover:bg-cyan-500"
              >
                Load selected instance
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
