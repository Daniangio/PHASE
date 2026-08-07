import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useParams } from 'react-router-dom';
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
import {
  assignClusterStates,
  deleteClusterUiSetup,
  fetchClusterUiSetups,
  fetchPottsClusterInfo,
  fetchStateTrajectoryOverlay,
  fetchSystem,
  saveClusterUiSetup,
  uploadStateTrajectory,
} from '../api/projects';

const CLUSTER_COLORS = ['#22d3ee', '#f59e0b', '#f43f5e', '#84cc16', '#a78bfa', '#fb7185', '#14b8a6', '#f97316', '#60a5fa', '#e879f9'];

function hexToInt(hex) {
  const clean = String(hex || '#22d3ee').replace('#', '');
  return parseInt(clean, 16);
}

function residFromKey(key) {
  const m = String(key || '').match(/(-?\d+)(?!.*\d)/);
  return m ? Number(m[1]) : null;
}

function normalizeBlock(block) {
  const start = Number(block.start);
  const end = Number(block.end);
  const residues = String(block.residuesText || block.residues || '')
    .split(/[\s,;]+/)
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isInteger(v));
  return {
    id: block.id || crypto.randomUUID(),
    name: block.name || 'block',
    start: Number.isInteger(start) ? start : null,
    end: Number.isInteger(end) ? end : null,
    residues,
  };
}

function selectedResiduesFromBlocks(blocks, residueKeys) {
  const available = new Set(residueKeys.map(residFromKey).filter((v) => Number.isInteger(v)));
  const out = new Set();
  (blocks || []).forEach((raw) => {
    const block = normalizeBlock(raw);
    if (Number.isInteger(block.start) && Number.isInteger(block.end)) {
      const a = Math.min(block.start, block.end);
      const b = Math.max(block.start, block.end);
      for (let r = a; r <= b; r += 1) if (available.has(r)) out.add(r);
    }
    block.residues.forEach((r) => { if (available.has(r)) out.add(r); });
  });
  return Array.from(out).sort((a, b) => a - b);
}

function parseFrameSelection(value, frameCount) {
  const selected = new Set();
  String(value || '').split(',').map((part) => part.trim()).filter(Boolean).forEach((part) => {
    const match = part.match(/^(\d+)\s*-\s*(\d+)$/);
    if (match) {
      const start = Number(match[1]);
      const end = Number(match[2]);
      // Ranges are half-open: 500-1000 means frames 500..999 (500 frames).
      for (let idx = Math.min(start, end); idx < Math.max(start, end); idx += 1) selected.add(idx);
      return;
    }
    if (/^\d+$/.test(part)) selected.add(Number(part));
  });
  const frames = Array.from(selected).filter((idx) => idx >= 0 && (!frameCount || idx < frameCount)).sort((a, b) => a - b);
  if (!frames.length) throw new Error('Enter at least one valid frame, for example 0-19 or 0,5,10.');
  if (frames.length > 500) throw new Error('Select at most 500 frames for one overlay.');
  return frames;
}

export default function ResidueSelectionPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const query = new URLSearchParams(location.search);

  const [system, setSystem] = useState(null);
  const [selectedClusterId, setSelectedClusterId] = useState(query.get('cluster_id') || '');
  const [clusterInfo, setClusterInfo] = useState(null);
  const [setups, setSetups] = useState([]);
  const [selectedSetupId, setSelectedSetupId] = useState('');
  const [name, setName] = useState('New residue selection');
  const [blocks, setBlocks] = useState([{ id: crypto.randomUUID(), name: 'block_A', start: '', end: '', residuesText: '' }]);
  const [selectedStateId, setSelectedStateId] = useState('');
  const [viewMode, setViewMode] = useState('reference');
  const [frameSelection, setFrameSelection] = useState('0-100');
  const [clusterResidueIndex, setClusterResidueIndex] = useState('0');
  const [overlaySummary, setOverlaySummary] = useState(null);
  const [trajectoryFile, setTrajectoryFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [pickTarget, setPickTarget] = useState(null); // { blockId, field }
  const [status, setStatus] = useState('initializing');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const containerRef = useRef(null);
  const pluginRef = useRef(null);

  useEffect(() => {
    fetchSystem(projectId, systemId)
      .then((data) => {
        setSystem(data);
        const clusters = (data?.metastable_clusters || []).filter((c) => c.path && c.status !== 'failed');
        if (!selectedClusterId && clusters[0]?.cluster_id) setSelectedClusterId(clusters[0].cluster_id);
        const states = Object.values(data?.states || {}).filter((s) => s.pdb_file);
        if (states[0]?.state_id) setSelectedStateId(states[0].state_id);
      })
      .catch((err) => setError(err.message || 'Failed to load system.'));
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    fetchPottsClusterInfo(projectId, systemId, selectedClusterId)
      .then(setClusterInfo)
      .catch((err) => setError(err.message || 'Failed to load cluster residue metadata.'));
    fetchClusterUiSetups(projectId, systemId, selectedClusterId, { setupType: 'residue_selection' })
      .then((res) => setSetups(Array.isArray(res?.setups) ? res.setups : []))
      .catch(() => setSetups([]));
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    let disposed = false;
    const init = async () => {
      if (!containerRef.current || pluginRef.current) return;
      try {
        const plugin = await createPluginUI({ target: containerRef.current, render: renderReact18 });
        if (disposed) { plugin.dispose?.(); return; }
        pluginRef.current = plugin;
        setStatus('ready');
      } catch (err) {
        setStatus('error');
        setError(err.message || 'Mol* initialization failed.');
      }
    };
    init();
    return () => { disposed = true; try { pluginRef.current?.dispose?.(); } catch { /* noop */ } pluginRef.current = null; };
  }, []);

  const selectedResidues = useMemo(
    () => selectedResiduesFromBlocks(blocks, clusterInfo?.residue_keys || []),
    [blocks, clusterInfo]
  );

  const getBase = useCallback(() => pluginRef.current?.managers?.structure?.hierarchy?.current?.structures?.[0]?.components?.[0]?.cell || null, []);

  const applyHighlight = useCallback(async () => {
    const plugin = pluginRef.current;
    const base = getBase();
    if (!plugin || !base || status !== 'ready' || viewMode !== 'reference') return;
    try { await clearStructureOverpaint(plugin, [base]); } catch { /* noop */ }
    if (!selectedResidues.length) return;
    const propFn = MS.struct.atomProperty.macromolecular.auth_seq_id();
    const residueTests = selectedResidues.length === 1
      ? MS.core.rel.eq([propFn, selectedResidues[0]])
      : MS.core.set.has([MS.set(...selectedResidues), propFn]);
    const expression = MS.struct.generator.atomGroups({ 'residue-test': residueTests });
    const rootStructure = plugin.managers.structure.hierarchy.current.structures[0]?.cell?.obj?.data;
    if (rootStructure) {
      const sel = Script.getStructureSelection(expression, rootStructure);
      if (StructureSelection.unionStructure(sel).elementCount === 0) return;
    }
    await setStructureOverpaint(plugin, [base], hexToInt('#22d3ee'), async (structure) => {
      const sel = Script.getStructureSelection(expression, structure);
      return StructureSelection.toLociWithSourceUnits(sel);
    }, ['cartoon']);
  }, [getBase, selectedResidues, status, viewMode]);

  const loadStructure = useCallback(async () => {
    const plugin = pluginRef.current;
    if (!plugin || !selectedStateId) return;
    setBusy(true);
    setError(null);
    try {
      await plugin.clear();
      await plugin.dataTransaction(async () => {
        const url = `/api/v1/projects/${projectId}/systems/${systemId}/structures/${encodeURIComponent(selectedStateId)}`;
        const data = await plugin.builders.data.download({ url: Asset.Url(url), label: selectedStateId }, { state: { isGhost: true } });
        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
      });
      await applyHighlight();
      setOverlaySummary(null);
    } catch (err) {
      setError(err.message || 'Failed to load structure.');
    } finally {
      setBusy(false);
    }
  }, [applyHighlight, projectId, selectedStateId, systemId]);

  useEffect(() => { if (status === 'ready' && selectedStateId && viewMode === 'reference') loadStructure(); }, [status, selectedStateId, viewMode, loadStructure]);
  useEffect(() => { applyHighlight(); }, [applyHighlight]);

  useEffect(() => {
    const plugin = pluginRef.current;
    if (status !== 'ready' || !plugin) return undefined;
    const sub = plugin.behaviors.interaction.click.subscribe((evt) => {
      if (!pickTarget) return;
      const loci = evt?.current?.loci;
      if (!StructureElement.Loci.is(loci)) return;
      const loc = StructureElement.Loci.getFirstLocation(loci);
      if (!loc) return;
      const auth = Number(StructureProperties.residue.auth_seq_id(loc));
      if (!Number.isInteger(auth)) return;
      setBlocks((prev) => prev.map((b) => (b.id === pickTarget.blockId ? { ...b, [pickTarget.field]: String(auth) } : b)));
      setPickTarget(null);
    });
    return () => { try { sub?.unsubscribe?.(); } catch { /* noop */ } };
  }, [pickTarget, status]);

  const clusters = (system?.metastable_clusters || []).filter((c) => c.path && c.status !== 'failed');
  const states = Object.values(system?.states || {}).filter((s) => s.pdb_file);
  const selectedState = states.find((state) => String(state.state_id) === String(selectedStateId)) || null;

  const loadTrajectoryOverlay = useCallback(async () => {
    if (!selectedClusterId || !selectedStateId || !clusterInfo) return;
    const residueIndex = Number(clusterResidueIndex);
    if (!Number.isInteger(residueIndex) || residueIndex < 0) {
      setError('Select a residue to color.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const frames = parseFrameSelection(frameSelection, Number(selectedState?.n_frames || 0));
      const result = await fetchStateTrajectoryOverlay(projectId, systemId, selectedClusterId, selectedStateId, {
        frame_indices: frames,
        residue_indices: [residueIndex],
      });
      const plugin = pluginRef.current;
      if (!plugin) throw new Error('Mol* is not ready.');
      await plugin.clear();
      for (const item of result?.structures || []) {
        await plugin.dataTransaction(async () => {
          const data = await plugin.builders.data.rawData({
            data: item.pdb,
            label: `${selectedStateId} frame ${item.frame_index}`,
          });
          const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
          await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
        });
      }

      const roots = plugin.managers.structure.hierarchy.current.structures || [];
      if (!roots.length) throw new Error('Mol* did not create structures for the selected frames.');
      await plugin.managers.structure.component.clear(roots);
      const residueKey = clusterInfo.residue_keys?.[residueIndex];
      const resid = residFromKey(residueKey);
      if (!Number.isInteger(resid)) throw new Error(`Cannot determine residue number from '${residueKey}'.`);
      const residueExpression = MS.struct.generator.atomGroups({
        'residue-test': MS.core.rel.eq([MS.struct.atomProperty.macromolecular.auth_seq_id(), resid]),
      });
      for (let idx = 0; idx < roots.length; idx += 1) {
        const root = roots[idx];
        const item = result.structures[idx];
        const clusterId = Number(item?.clusters?.[String(residueIndex)] ?? -1);
        const color = clusterId >= 0 ? CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length] : '#64748b';
        const base = await plugin.builders.structure.tryCreateComponentFromExpression(root.cell, MS.struct.generator.all(), `overlay-base-${idx}`);
        if (base) {
          await plugin.builders.structure.representation.addRepresentation(base, {
            type: 'cartoon',
            color: 'uniform',
            colorParams: { value: hexToInt('#d1d5db') },
          });
        }
        const residueComponent = await plugin.builders.structure.tryCreateComponentFromExpression(root.cell, residueExpression, `overlay-residue-${idx}`);
        if (residueComponent) {
          await plugin.builders.structure.representation.addRepresentation(residueComponent, {
            type: 'ball-and-stick',
            color: 'uniform',
            colorParams: { value: hexToInt(color) },
          });
        }
      }
      setOverlaySummary({ ...result, residueIndex, residueKey });
    } catch (err) {
      setError(err.message || 'Failed to load trajectory overlay.');
    } finally {
      setBusy(false);
    }
  }, [clusterInfo, clusterResidueIndex, frameSelection, projectId, selectedClusterId, selectedState, selectedStateId, systemId]);

  const uploadTrajectory = useCallback(async () => {
    if (!trajectoryFile || !selectedStateId) return;
    setBusy(true);
    setError(null);
    setUploadProgress(0);
    try {
      const form = new FormData();
      form.append('trajectory', trajectoryFile);
      form.append('stride', '1');
      form.append('build_descriptors_after_upload', 'true');
      await uploadStateTrajectory(projectId, systemId, selectedStateId, form, {
        onUploadProgress: setUploadProgress,
      });
      if (selectedClusterId) await assignClusterStates(projectId, systemId, selectedClusterId, [selectedStateId]);
      const refreshed = await fetchSystem(projectId, systemId);
      setSystem(refreshed);
      setTrajectoryFile(null);
      setUploadProgress(null);
    } catch (err) {
      setError(err.message || 'Failed to store and assign trajectory.');
    } finally {
      setBusy(false);
    }
  }, [projectId, selectedClusterId, selectedStateId, systemId, trajectoryFile]);

  const refreshTrajectoryAssignment = useCallback(async () => {
    if (!selectedClusterId || !selectedStateId) return;
    setBusy(true);
    setError(null);
    try {
      await assignClusterStates(projectId, systemId, selectedClusterId, [selectedStateId]);
      setOverlaySummary(null);
    } catch (err) {
      setError(err.message || 'Failed to assign trajectory frames to the selected cluster.');
    } finally {
      setBusy(false);
    }
  }, [projectId, selectedClusterId, selectedStateId, systemId]);

  const updateBlock = (id, patch) => setBlocks((prev) => prev.map((b) => (b.id === id ? { ...b, ...patch } : b)));
  const loadSetup = (setup) => {
    setSelectedSetupId(setup.setup_id);
    setName(setup.name || setup.setup_id);
    const loadedBlocks = Array.isArray(setup?.payload?.blocks) ? setup.payload.blocks : [];
    setBlocks(loadedBlocks.length ? loadedBlocks.map((b) => ({ ...b, id: b.id || crypto.randomUUID(), residuesText: (b.residues || []).join(',') })) : []);
  };

  const saveSelection = async () => {
    if (!selectedClusterId) return;
    setBusy(true);
    setError(null);
    try {
      const normalized = blocks.map(normalizeBlock);
      const payload = {
        blocks: normalized,
        selected_residues: selectedResidues,
        selected_residue_indices: selectedResidues.map((resid) => (clusterInfo?.residue_keys || []).findIndex((k) => residFromKey(k) === resid)).filter((i) => i >= 0),
      };
      const saved = await saveClusterUiSetup(projectId, systemId, selectedClusterId, {
        setup_id: selectedSetupId || undefined,
        name,
        setup_type: 'residue_selection',
        page: 'residue_selection',
        payload,
      });
      setSelectedSetupId(saved.setup_id);
      const res = await fetchClusterUiSetups(projectId, systemId, selectedClusterId, { setupType: 'residue_selection' });
      setSetups(Array.isArray(res?.setups) ? res.setups : []);
    } catch (err) {
      setError(err.message || 'Failed to save selection.');
    } finally {
      setBusy(false);
    }
  };

  const deleteSelection = async () => {
    if (!selectedClusterId || !selectedSetupId) return;
    setBusy(true);
    try {
      await deleteClusterUiSetup(projectId, systemId, selectedClusterId, selectedSetupId);
      setSelectedSetupId('');
      setName('New residue selection');
      setBlocks([{ id: crypto.randomUUID(), name: 'block_A', start: '', end: '', residuesText: '' }]);
      const res = await fetchClusterUiSetups(projectId, systemId, selectedClusterId, { setupType: 'residue_selection' });
      setSetups(Array.isArray(res?.setups) ? res.setups : []);
    } catch (err) {
      setError(err.message || 'Failed to delete selection.');
    } finally {
      setBusy(false);
    }
  };

  if (!system) return <Loader message="Loading residue selection editor..." />;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Structures & residue selections</h1>
          <p className="text-sm text-gray-400">Build reusable selections or overlay trajectory frames with residues colored by their assigned clusters.</p>
        </div>
        <button type="button" onClick={saveSelection} disabled={busy || !name.trim()} className="rounded-md bg-cyan-500 px-4 py-2 text-sm font-semibold text-black disabled:opacity-50">Save selection</button>
      </div>
      {error ? <ErrorMessage message={error} /> : null}
      <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-4">
        <aside className="space-y-3 rounded-lg border border-gray-800 bg-gray-900/60 p-3">
          <label className="block text-xs text-gray-400">Cluster</label>
          <select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
            {clusters.map((c) => <option key={c.cluster_id} value={c.cluster_id}>{c.name || c.cluster_id}</option>)}
          </select>
          <label className="block text-xs text-gray-400">Saved selections</label>
          <select value={selectedSetupId} onChange={(e) => { const setup = setups.find((s) => s.setup_id === e.target.value); if (setup) loadSetup(setup); else setSelectedSetupId(''); }} className="w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
            <option value="">New selection</option>
            {setups.map((s) => <option key={s.setup_id} value={s.setup_id}>{s.name || s.setup_id}</option>)}
          </select>
          <input value={name} onChange={(e) => setName(e.target.value)} className="w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100" placeholder="Selection name" />
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold uppercase tracking-wide text-gray-400">OR blocks</div>
              <button type="button" onClick={() => setBlocks((prev) => [...prev, { id: crypto.randomUUID(), name: `block_${prev.length + 1}`, start: '', end: '', residuesText: '' }])} className="text-xs text-cyan-300">Add block</button>
            </div>
            {blocks.map((block) => (
              <div key={block.id} className="rounded border border-gray-800 bg-gray-950/60 p-2 space-y-2">
                <input value={block.name} onChange={(e) => updateBlock(block.id, { name: e.target.value })} className="w-full rounded bg-gray-900 border border-gray-700 px-2 py-1.5 text-xs text-gray-100" />
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[11px] text-gray-500">Start resid</label>
                    <div className="flex gap-1"><input value={block.start} onChange={(e) => updateBlock(block.id, { start: e.target.value })} className="min-w-0 flex-1 rounded bg-gray-900 border border-gray-700 px-2 py-1.5 text-xs text-gray-100" /><button type="button" onClick={() => setPickTarget({ blockId: block.id, field: 'start' })} className="rounded border border-gray-700 px-2 text-xs text-gray-300">pick</button></div>
                  </div>
                  <div>
                    <label className="text-[11px] text-gray-500">End resid</label>
                    <div className="flex gap-1"><input value={block.end} onChange={(e) => updateBlock(block.id, { end: e.target.value })} className="min-w-0 flex-1 rounded bg-gray-900 border border-gray-700 px-2 py-1.5 text-xs text-gray-100" /><button type="button" onClick={() => setPickTarget({ blockId: block.id, field: 'end' })} className="rounded border border-gray-700 px-2 text-xs text-gray-300">pick</button></div>
                  </div>
                </div>
                <input value={block.residuesText || ''} onChange={(e) => updateBlock(block.id, { residuesText: e.target.value })} className="w-full rounded bg-gray-900 border border-gray-700 px-2 py-1.5 text-xs text-gray-100" placeholder="Extra residues, e.g. 45, 89, 131" />
                <button type="button" onClick={() => setBlocks((prev) => prev.filter((b) => b.id !== block.id))} className="text-xs text-red-300">Remove block</button>
              </div>
            ))}
          </div>
          <div className="rounded border border-gray-800 bg-gray-950/50 p-2 text-xs text-gray-300">
            Selected residues: {selectedResidues.length ? selectedResidues.join(', ') : 'none'}
          </div>
          {selectedSetupId ? <button type="button" onClick={deleteSelection} disabled={busy} className="rounded border border-red-500/60 px-3 py-2 text-xs text-red-300">Delete selection</button> : null}
        </aside>
        <main className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => setViewMode('reference')} className={`rounded border px-3 py-2 text-xs ${viewMode === 'reference' ? 'border-cyan-500 bg-cyan-950/50 text-cyan-400' : 'border-gray-700 text-gray-300'}`}>Reference structure</button>
              <button type="button" onClick={() => setViewMode('overlay')} className={`rounded border px-3 py-2 text-xs ${viewMode === 'overlay' ? 'border-cyan-500 bg-cyan-950/50 text-cyan-400' : 'border-gray-700 text-gray-300'}`}>Trajectory overlay</button>
            </div>
            <div className="mb-3 grid gap-3 rounded border border-gray-800 bg-gray-950/40 p-3 lg:grid-cols-2">
              <label className="text-xs text-gray-400">
                State / topology
                <select value={selectedStateId} onChange={(e) => { setSelectedStateId(e.target.value); setOverlaySummary(null); }} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
                  {states.map((s) => <option key={s.state_id} value={s.state_id}>{s.name || s.state_id} {s.trajectory_file ? `(${s.n_frames || '?'} frames)` : '(PDB only)'}</option>)}
                </select>
              </label>
              {viewMode === 'reference' ? (
                <div className="flex items-end gap-2">
                  <button type="button" onClick={loadStructure} className="rounded border border-gray-700 px-3 py-2 text-xs text-gray-100">Reload structure</button>
                  {pickTarget ? <span className="text-xs text-cyan-300">Click a residue in Mol* to fill {pickTarget.field}.</span> : null}
                </div>
              ) : (
                <>
                  <label className="text-xs text-gray-400">
                    Frames (end-exclusive, maximum 500)
                    <input value={frameSelection} onChange={(e) => setFrameSelection(e.target.value)} placeholder="0-100 or 0,5,10" className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100" />
                    <span className="mt-1 block text-[11px] text-gray-500">`0-100` loads 100 frames; `500-1000` loads 500. Change the range and refresh.</span>
                  </label>
                  <label className="text-xs text-gray-400">
                    Residue colored by cluster
                    <select value={clusterResidueIndex} onChange={(e) => setClusterResidueIndex(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
                      {(clusterInfo?.residue_keys || []).map((key, index) => (
                        <option key={`${key}-${index}`} value={index}>
                          {clusterInfo?.residue_display_labels?.[index] || key} · {clusterInfo?.cluster_counts?.[index] || '?'} clusters
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className="flex flex-wrap items-end gap-2">
                    <button type="button" onClick={loadTrajectoryOverlay} disabled={busy} className="rounded bg-cyan-600 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-500 disabled:opacity-50">Refresh overlay</button>
                    <button type="button" onClick={refreshTrajectoryAssignment} disabled={busy} title="Recompute this state's cluster labels after replacing descriptors or a trajectory." className="rounded border border-gray-700 px-3 py-2 text-xs text-gray-200 disabled:opacity-50">Reassign clusters</button>
                  </div>
                </>
              )}
            </div>
            {viewMode === 'overlay' && !selectedState?.trajectory_file ? (
              <div className="mb-3 rounded border border-amber-500/40 bg-amber-950/20 p-3">
                <p className="text-xs text-amber-200">This state only has its PDB frame. Upload a matching trajectory; PHASE will store it, rebuild descriptors, and assign its frames to the selected cluster.</p>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <input type="file" accept=".xtc,.dcd,.trr,.nc,.nctraj,.pdb" onChange={(e) => setTrajectoryFile(e.target.files?.[0] || null)} className="min-w-0 flex-1 text-xs text-gray-300" />
                  <button type="button" disabled={!trajectoryFile || busy} onClick={uploadTrajectory} className="rounded border border-amber-400/60 px-3 py-2 text-xs text-amber-100 disabled:opacity-50">Upload, build & assign</button>
                  {uploadProgress !== null ? <span className="text-xs text-gray-300">{uploadProgress}% uploaded</span> : null}
                </div>
              </div>
            ) : null}
            {viewMode === 'overlay' && overlaySummary ? (
              <div className="mb-3 rounded border border-gray-800 bg-gray-950/50 p-3 text-xs text-gray-300">
                <div className="font-semibold text-gray-100">{overlaySummary.residueKey} across {overlaySummary.structures?.length || 0} structures</div>
                <div className="mt-2 flex flex-wrap gap-3">
                  {Array.from(new Set((overlaySummary.structures || []).map((item) => Number(item?.clusters?.[String(overlaySummary.residueIndex)] ?? -1)))).sort((a, b) => a - b).map((clusterId) => (
                    <span key={clusterId} className="inline-flex items-center gap-1.5">
                      <span className="h-3 w-3 rounded-full" style={{ backgroundColor: clusterId >= 0 ? CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length] : '#64748b' }} />
                      {clusterId >= 0 ? `cluster ${clusterId}` : 'unassigned'}
                    </span>
                  ))}
                </div>
                <p className="mt-2 text-gray-500">Each frame is a separate superposed structure. The selected residue uses ball-and-stick (licorice-style) rendering; the protein backbone remains an opaque cartoon.</p>
              </div>
            ) : null}
            <div className="relative h-[70vh] min-h-[520px] overflow-hidden rounded border border-gray-800 bg-black/30">
              {(status !== 'ready' || busy) ? <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50"><Loader message={busy ? (uploadProgress !== null ? 'Uploading and assigning trajectory...' : 'Loading structures...') : 'Initializing viewer...'} /></div> : null}
              <div ref={containerRef} className="h-full w-full" />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
