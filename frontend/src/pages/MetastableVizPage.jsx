import { useEffect, useRef, useState, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchMetastableStates, renameMetastableState, metastablePdbUrl, fetchSystem } from '../api/projects';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import 'molstar/build/viewer/molstar.css';

export default function MetastableVizPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();

  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  const [metastableStates, setMetastableStates] = useState([]);
  const [systemName, setSystemName] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [loadedMap, setLoadedMap] = useState({});

  const loadMetastable = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const [metaRes, sysRes] = await Promise.all([
        fetchMetastableStates(projectId, systemId),
        fetchSystem(projectId, systemId),
      ]);
      setMetastableStates(metaRes.metastable_states || []);
      setSystemName(sysRes?.name || systemId);
    } catch (err) {
      setError(err.message || 'Failed to load metastable states.');
    } finally {
      setLoading(false);
    }
  }, [projectId, systemId]);

  useEffect(() => {
    let disposed = false;
    const init = async () => {
      if (!containerRef.current || pluginRef.current) return;
      try {
        const plugin = await createPluginUI({
          target: containerRef.current,
          render: renderReact18,
        });
        if (disposed) {
          plugin.dispose?.();
          return;
        }
        pluginRef.current = plugin;
      } catch (viewerErr) {
        console.error('Failed to initialize Mol* viewer', viewerErr);
        setError('3D viewer initialization failed.');
      }
    };
    init();
    return () => {
      disposed = true;
      if (pluginRef.current) {
        try {
          pluginRef.current.dispose?.();
        } catch (err) {
          console.warn('Failed to dispose Mol* viewer', err);
        }
        pluginRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    loadMetastable();
  }, [loadMetastable]);

  const removeMetastable = async (metastableId) => {
    const plugin = pluginRef.current;
    const refs = loadedMap[metastableId];
    if (plugin && refs) {
      try {
        if (refs.trajectoryRef) {
          await plugin.state.data.remove(refs.trajectoryRef);
        } else if (refs.dataRef) {
          await plugin.state.data.remove(refs.dataRef);
        }
      } catch (err) {
        // ignore
      }
    }
    setLoadedMap((prev) => {
      const next = { ...prev };
      delete next[metastableId];
      return next;
    });
  };

  const loadStructure = async (meta) => {
    if (!meta.metastable_id) return;
    if (loadedMap[meta.metastable_id]) {
      await removeMetastable(meta.metastable_id);
      return;
    }
    setActionError(null);
    const plugin = pluginRef.current;
    if (!plugin) return;
    try {
      const url = metastablePdbUrl(projectId, systemId, meta.metastable_id);
      await plugin.dataTransaction(async () => {
        const data = await plugin.builders.data.download(
          { url: Asset.Url(url), label: meta.name || meta.metastable_id },
          { state: { isGhost: true } }
        );
        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
        setLoadedMap((prev) => ({
          ...prev,
          [meta.metastable_id]: { dataRef: data.ref, trajectoryRef: trajectory.ref },
        }));
      });
    } catch (err) {
      setActionError(err.message || 'Failed to load structure.');
    }
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
      <button
        onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)}
        className="text-cyan-400 hover:text-cyan-300 text-sm"
      >
        ← Back to system
      </button>
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
          <div
            ref={containerRef}
            className="w-full h-[520px] bg-black rounded-lg overflow-hidden border border-gray-700"
          />
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
