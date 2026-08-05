import { useEffect, useRef, useState, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchMetastableStates, renameMetastableState, metastablePdbUrl, fetchSystem } from '../api/projects';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import 'molstar/build/viewer/molstar.css';

export default function MetastableVizPage() {
  const { projectId, systemId } = useParams();

  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  const [metastableStates, setMetastableStates] = useState([]);
  const [systemName, setSystemName] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [loadedMap, setLoadedMap] = useState({});
  const [viewerStatus, setViewerStatus] = useState('initializing'); // initializing | ready | error
  const statusRef = useRef('initializing');

  const loadMetastable = useCallback(async () => {
    setError(null);
    setLoading(true);
    console.info('[MetastableViz] Loading metastable states…', { projectId, systemId });
    try {
      const [metaRes, sysRes] = await Promise.all([
        fetchMetastableStates(projectId, systemId),
        fetchSystem(projectId, systemId),
      ]);
      setMetastableStates(metaRes.metastable_states || []);
      setSystemName(sysRes?.name || systemId);
      console.info('[MetastableViz] Loaded metastable states', {
        count: (metaRes.metastable_states || []).length,
        systemName: sysRes?.name,
      });
    } catch (err) {
      setError(err.message || 'Failed to load metastable states.');
      console.error('[MetastableViz] Failed to load metastable states', err);
    } finally {
      setLoading(false);
    }
  }, [projectId, systemId]);

  useEffect(() => {
    let disposed = false;
    let rafId;

    const tryInit = async () => {
      if (disposed || loading) return;
      if (!containerRef.current) {
        console.warn('[MetastableViz] containerRef not ready, retrying...');
        rafId = requestAnimationFrame(tryInit);
        return;
      }
      if (pluginRef.current) {
        return;
      }
      setViewerStatus('initializing');
      statusRef.current = 'initializing';
      const timeout = setTimeout(() => {
        if (statusRef.current === 'initializing') {
          console.error('[MetastableViz] Mol* init timeout.');
          setViewerStatus('error');
          statusRef.current = 'error';
          setError('3D viewer initialization timed out.');
        }
      }, 8000);
      try {
        console.info('[MetastableViz] Initializing Mol* plugin…', { hasContainer: !!containerRef.current });
        const plugin = await createPluginUI({
          target: containerRef.current,
          render: renderReact18,
        });
        if (disposed) {
          plugin.dispose?.();
          clearTimeout(timeout);
          return;
        }
        pluginRef.current = plugin;
        setViewerStatus('ready');
        statusRef.current = 'ready';
        console.info('[MetastableViz] Mol* plugin ready');
        clearTimeout(timeout);
      } catch (viewerErr) {
        console.error('Failed to initialize Mol* viewer', viewerErr);
        setError('3D viewer initialization failed.');
        setViewerStatus('error');
        statusRef.current = 'error';
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
          console.warn('Failed to dispose Mol* viewer', err);
        }
        pluginRef.current = null;
      }
    };
  }, [loading]);

  useEffect(() => {
    loadMetastable();
  }, [loadMetastable]);

  const removeMetastable = async (metastableId) => {
    const plugin = pluginRef.current;
    if (!plugin) return;

    // Clear entire scene and re-load remaining loaded metastables (simpler than per-node removal).
    const remainingKeys = Object.keys(loadedMap).filter((k) => k !== metastableId);
    const remainingMetas = metastableStates.filter((m) => remainingKeys.includes(m.metastable_id));

    try {
      await plugin.clear();
      setLoadedMap({});
      for (const meta of remainingMetas) {
        await loadStructureRaw(meta);
      }
      console.info('[MetastableViz] Unloaded metastable and reloaded remaining', metastableId);
    } catch (err) {
      console.warn('[MetastableViz] Failed to unload metastable', metastableId, err);
    }
  };

  const loadStructureRaw = async (meta) => {
    if (!meta.metastable_id) return;
    setActionError(null);
    const plugin = pluginRef.current;
    if (!plugin) {
      console.error('[MetastableViz] Plugin not initialized; cannot load structure');
      return;
    }
    try {
      const url = metastablePdbUrl(projectId, systemId, meta.metastable_id);
      console.info('[MetastableViz] Loading PDB', { url, meta });
      await plugin.dataTransaction(async () => {
        const data = await plugin.builders.data.download(
          { url: Asset.Url(url), label: meta.name || meta.metastable_id },
          { state: { isGhost: true } }
        );
        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
        const preset = await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
        const structureRef = preset?.structure?.cell?.transform?.ref;
        setLoadedMap((prev) => ({
          ...prev,
          [meta.metastable_id]: {
            dataRef: data.ref,
            trajectoryRef: trajectory.ref,
            structureRef,
          },
        }));
      });
      console.info('[MetastableViz] Loaded metastable', meta.metastable_id);
    } catch (err) {
      setActionError(err.message || 'Failed to load structure.');
      console.error('[MetastableViz] Failed to load structure', err);
    }
  };

  const loadStructure = async (meta) => {
    if (!meta.metastable_id) return;
    if (loadedMap[meta.metastable_id]) {
      await removeMetastable(meta.metastable_id);
      return;
    }
    await loadStructureRaw(meta);
  };

  const handleRename = async (meta, name) => {
    if (!meta.metastable_id || !name.trim()) return;
    setActionError(null);
    try {
      await renameMetastableState(projectId, systemId, meta.metastable_id, name.trim());
      await loadMetastable();
    } catch (err) {
      setActionError(err.message || 'Rename failed.');
    }
  };

  if (loading) return <Loader message="Loading metastable states..." />;
  if (error) return <ErrorMessage message={error} />;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Metastable Visualization</h1>
          <p className="text-sm text-gray-400">{systemName}</p>
        </div>
        <button
          onClick={loadMetastable}
          className="text-xs px-3 py-1 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10"
        >
          Refresh
        </button>
      </div>

      {actionError && <ErrorMessage message={actionError} />}

      <div className="grid md:grid-cols-3 gap-3">
        <div className="md:col-span-1 space-y-2">
          <h2 className="text-sm font-semibold text-white">Metastable States</h2>
          {metastableStates.length === 0 && (
            <p className="text-xs text-gray-400">No metastable states computed yet.</p>
          )}
          <div className="space-y-2">
            {metastableStates.map((meta) => (
              <MetastableListItem
                key={meta.metastable_id || `${meta.macro_state}-${meta.metastable_index}`}
                meta={meta}
                loaded={!!loadedMap[meta.metastable_id]}
                onToggle={() => loadStructure(meta)}
                onRename={handleRename}
              />
            ))}
          </div>
        </div>
        <div className="md:col-span-2">
          <div className="relative">
            {viewerStatus === 'initializing' && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                <Loader message="Initializing viewer..." />
              </div>
            )}
            {viewerStatus === 'error' && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-10">
                <ErrorMessage message="Viewer failed to load. Check console logs." />
              </div>
            )}
            <div
              ref={containerRef}
              className="w-full h-[520px] bg-black rounded-lg overflow-hidden border border-gray-700"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function MetastableListItem({ meta, loaded, onToggle, onRename }) {
  const [name, setName] = useState(meta.name || meta.default_name || meta.metastable_id);
  const [saving, setSaving] = useState(false);
  const dirty = name.trim() !== (meta.name || meta.default_name || meta.metastable_id);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onRename(meta, name);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-md p-2 space-y-2">
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span>{meta.macro_state}</span>
        <span>Frames: {meta.n_frames ?? '—'}</span>
      </div>
      <div className="space-y-1">
        <label className="block text-[11px] text-gray-500">Name</label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white text-sm"
        />
      </div>
      <div className="flex items-center justify-between text-xs">
        {dirty ? (
          <button
            onClick={handleSave}
            disabled={saving}
            className="text-cyan-400 hover:text-cyan-300 disabled:opacity-50"
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
        ) : (
          <span className="px-2 py-1 rounded-md bg-emerald-500/20 text-emerald-200">Saved</span>
        )}
        <div className="space-x-2">
          {!dirty && meta.default_name && meta.default_name !== name && (
            <button
              onClick={() => setName(meta.default_name)}
              className="text-gray-400 hover:text-gray-200"
            >
              Reset
            </button>
          )}
          <button
            onClick={onToggle}
            className={`px-2 py-1 rounded-md ${loaded ? 'bg-red-500/30 text-red-200' : 'bg-emerald-500/20 text-emerald-200'}`}
          >
            {loaded ? 'Unload' : 'Load'}
          </button>
        </div>
      </div>
    </div>
  );
}
