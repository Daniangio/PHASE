import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';

import ErrorMessage from '../components/common/ErrorMessage';
import Loader from '../components/common/Loader';
import MolstarTrajectoryViewer from '../components/visualization/MolstarTrajectoryViewer';
import { API_BASE, requestJSON } from '../api/client';
import { fetchSystem } from '../api/projects';

function stateLabel(state) {
  return state?.name || state?.state_id || 'state';
}

export default function MolstarTrajectoryTestPage() {
  const { projectId, systemId } = useParams();
  const viewerRef = useRef(null);
  const [viewerStatus, setViewerStatus] = useState('initializing');
  const [error, setError] = useState(null);
  const [sourceMode, setSourceMode] = useState(projectId && systemId ? 'state' : 'upload');
  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(Boolean(projectId && systemId));
  const [selectedStateId, setSelectedStateId] = useState('');
  const [topologyFile, setTopologyFile] = useState(null);
  const [trajectoryFile, setTrajectoryFile] = useState(null);
  const [frameHint, setFrameHint] = useState('all');

  useEffect(() => {
    if (!projectId || !systemId) return;
    let cancelled = false;
    setLoadingSystem(true);
    fetchSystem(projectId, systemId)
      .then((payload) => {
        if (cancelled) return;
        setSystem(payload);
        const values = payload?.states && !Array.isArray(payload.states) ? Object.values(payload.states) : payload?.states || [];
        const withTrajectory = values.find((s) => s?.pdb_file && s?.trajectory_file);
        const first = withTrajectory || values.find((s) => s?.pdb_file);
        if (first) setSelectedStateId(first.state_id);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || 'Failed to load system.');
      })
      .finally(() => {
        if (!cancelled) setLoadingSystem(false);
      });
    return () => { cancelled = true; };
  }, [projectId, systemId]);

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

  const canUseStateSource = Boolean(projectId && systemId);
  const canLoad = viewerStatus === 'ready';

  const loadIntoViewer = useCallback(async (payload) => {
    setError(null);
    try {
      if (payload.mode === 'structure') await viewerRef.current?.loadStructure(payload);
      else await viewerRef.current?.loadTrajectory(payload);
    } catch (err) {
      setError(err.message || 'Failed to load topology + trajectory into Mol*.');
    }
  }, []);

  const loadSelected = useCallback(async () => {
    if (!canLoad) return;
    if (sourceMode === 'state') {
      if (!projectId || !systemId || !selectedState) {
        setError('Select a state with a stored trajectory.');
        return;
      }
      if (!selectedState.trajectory_file) {
        await loadIntoViewer({
          mode: 'structure',
          structureUrl: `${API_BASE}/projects/${encodeURIComponent(projectId)}/systems/${encodeURIComponent(systemId)}/structures/${encodeURIComponent(selectedState.state_id)}`,
          structureName: selectedState.pdb_file || 'structure.pdb',
        });
        return;
      }

      const rawInfoPath = `/projects/${encodeURIComponent(projectId)}/systems/${encodeURIComponent(systemId)}/states/${encodeURIComponent(selectedState.state_id)}/trajectory/raw/info`;
      try {
        const info = await requestJSON(rawInfoPath);
        if (!info?.available) {
          setError(info?.detail || 'Stored trajectory is not reachable by the backend container.');
          return;
        }
      } catch (err) {
        setError(err.message || 'Stored trajectory is not reachable by the backend container.');
        return;
      }

      await loadIntoViewer({
        mode: 'trajectory',
        topologyUrl: `${API_BASE}/projects/${encodeURIComponent(projectId)}/systems/${encodeURIComponent(systemId)}/structures/${encodeURIComponent(selectedState.state_id)}`,
        topologyName: selectedState.pdb_file || 'structure.pdb',
        coordinatesUrl: `${API_BASE}/projects/${encodeURIComponent(projectId)}/systems/${encodeURIComponent(systemId)}/states/${encodeURIComponent(selectedState.state_id)}/trajectory/raw`,
        coordinatesName: selectedState.source_traj || selectedState.trajectory_file || 'trajectory.xtc',
      });
      return;
    }

    if (!topologyFile) {
      setError('Choose at least a topology/structure file.');
      return;
    }
    const topologyData = new Uint8Array(await topologyFile.arrayBuffer());
    if (!trajectoryFile) {
      await loadIntoViewer({
        mode: 'structure',
        structureData: topologyData,
        structureName: topologyFile.name,
      });
      return;
    }
    const coordinatesData = new Uint8Array(await trajectoryFile.arrayBuffer());
    await loadIntoViewer({
      mode: 'trajectory',
      topologyData,
      topologyName: topologyFile.name,
      coordinatesData,
      coordinatesName: trajectoryFile.name,
    });
  }, [
    canLoad,
    sourceMode,
    projectId,
    systemId,
    selectedState,
    topologyFile,
    trajectoryFile,
    loadIntoViewer,
  ]);

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Mol* Raw Trajectory Test</h1>
          <p className="text-sm text-gray-400">
            Embedded Mol* component for topology + raw coordinate trajectories. Frame playback uses Mol* native controls.
          </p>
        </div>
        {projectId && systemId && (
          <Link className="text-xs text-cyan-300 hover:text-cyan-200" to={`/projects/${projectId}/systems/${systemId}/sampling/transient_states_3d`}>
            Back to transient 3D
          </Link>
        )}
      </div>

      {error && <ErrorMessage message={error} />}
      {loadingSystem && <Loader message="Loading system states..." />}

      <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-4 min-h-0">
        <aside className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-4 h-fit">
          <div>
            <h2 className="text-sm font-semibold text-white">Trajectory Loader</h2>
            <p className="mt-1 text-xs text-gray-500">
              Use matching atom order/count between topology and coordinates. Large XTC/DCD files are streamed to the browser.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            {canUseStateSource && (
              <button
                type="button"
                onClick={() => setSourceMode('state')}
                className={`rounded-md px-3 py-1.5 text-xs border ${sourceMode === 'state' ? 'border-cyan-500 bg-cyan-950/50 text-cyan-400' : 'border-gray-700 text-gray-300 hover:bg-gray-800'}`}
              >
                Stored PHASE state
              </button>
            )}
            <button
              type="button"
              onClick={() => setSourceMode('upload')}
              className={`rounded-md px-3 py-1.5 text-xs border ${sourceMode === 'upload' ? 'border-cyan-500 bg-cyan-950/50 text-cyan-400' : 'border-gray-700 text-gray-300 hover:bg-gray-800'}`}
            >
              Local files
            </button>
          </div>

          {sourceMode === 'state' && (
            <div className="space-y-3">
              <label className="block text-xs text-gray-400">
                State with stored trajectory
                <select
                  value={selectedStateId}
                  onChange={(e) => setSelectedStateId(e.target.value)}
                  className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100"
                >
                  {states.map((s) => (
                    <option key={s.state_id} value={s.state_id}>
                      {stateLabel(s)} {s.trajectory_file ? `(trajectory, ${s.n_frames || '?'} frames)` : '(PDB only)'}
                    </option>
                  ))}
                </select>
              </label>
              {selectedState?.trajectory_file?.startsWith?.('/') && (
                <p className="text-xs text-amber-300">
                  This state points to an absolute host path. Docker can stream it only if that path is mounted inside the backend container,
                  otherwise re-upload the trajectory from the System page so it is stored under the shared PHASE data root.
                </p>
              )}
              {!selectedState?.trajectory_file && (
                <p className="text-xs text-amber-300">
                  This state has no raw trajectory; loading it will show the static PDB only.
                </p>
              )}
            </div>
          )}

          {sourceMode === 'upload' && (
            <div className="space-y-3">
              <label className="block text-xs text-gray-400">
                Topology / structure
                <input
                  type="file"
                  accept=".pdb,.ent,.cif,.mmcif,.bcif,.gro"
                  onChange={(e) => setTopologyFile(e.target.files?.[0] || null)}
                  className="mt-1 block w-full text-sm text-gray-300 bg-gray-950 border border-gray-700 rounded-md p-2"
                />
              </label>
              <label className="block text-xs text-gray-400">
                Raw trajectory coordinates
                <input
                  type="file"
                  accept=".xtc,.dcd,.trr,.nc,.nctraj,.pdb,.ent,.cif,.mmcif,.bcif,.gro"
                  onChange={(e) => setTrajectoryFile(e.target.files?.[0] || null)}
                  className="mt-1 block w-full text-sm text-gray-300 bg-gray-950 border border-gray-700 rounded-md p-2"
                />
              </label>
            </div>
          )}

          <label className="block text-xs text-gray-400">
            Frame request
            <select
              value={frameHint}
              onChange={(e) => setFrameHint(e.target.value)}
              className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100"
            >
              <option value="all">Load full trajectory, scroll frames in Mol*</option>
              <option value="range" disabled>Frame range filter (not available for raw browser load yet)</option>
            </select>
          </label>

          <button
            type="button"
            onClick={loadSelected}
            disabled={!canLoad}
            className="w-full rounded-md bg-cyan-600 px-3 py-2 text-sm font-semibold text-white hover:bg-cyan-500 disabled:opacity-50"
          >
            {viewerStatus === 'loading' ? 'Loading...' : 'Load into Mol*'}
          </button>
        </aside>

        <MolstarTrajectoryViewer
          ref={viewerRef}
          height={760}
          onStatusChange={setViewerStatus}
          onError={setError}
        />
      </div>
    </div>
  );
}
