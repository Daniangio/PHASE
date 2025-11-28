import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import {
  fetchSystem,
  downloadStructure,
  uploadStateTrajectory,
  deleteStateTrajectory,
  addSystemState,
  deleteState,
} from '../api/projects';
import { submitStaticJob, submitDynamicJob, submitQuboJob, fetchResults } from '../api/jobs';
import StaticAnalysisForm from '../components/analysis/StaticAnalysisForm';
import DynamicAnalysisForm from '../components/analysis/DynamicAnalysisForm';
import QuboAnalysisForm from '../components/analysis/QuboAnalysisForm';
import { Download } from 'lucide-react';

export default function SystemDetailPage() {
  const { projectId, systemId } = useParams();
  const [system, setSystem] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [pageError, setPageError] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [downloadError, setDownloadError] = useState(null);
  const [staticResults, setStaticResults] = useState([]);
  const [staticError, setStaticError] = useState(null);
  const [uploadingState, setUploadingState] = useState(null);
  const [addingState, setAddingState] = useState(false);
  const [actionMessage, setActionMessage] = useState(null);
  const [uploadProgress, setUploadProgress] = useState({});
  const [processingState, setProcessingState] = useState(null);
  const [statePair, setStatePair] = useState({ a: null, b: null });
  const navigate = useNavigate();

  useEffect(() => {
    const loadSystem = async () => {
      setIsLoading(true);
      setPageError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
        setActionMessage(null);
      } catch (err) {
        setPageError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    loadSystem();
  }, [projectId, systemId]);

  useEffect(() => {
    const loadStatics = async () => {
      if (!projectId || !systemId) return;
      try {
        const allResults = await fetchResults();
        const matching = allResults.filter(
          (res) => res.analysis_type === 'static' && res.system_id === systemId && res.status === 'finished'
        );
        setStaticResults(matching);
        setStaticError(null);
      } catch (err) {
        setStaticError(err.message);
      }
    };
    loadStatics();
  }, [projectId, systemId]);

  const states = useMemo(() => Object.values(system?.states || {}), [system]);
  const descriptorStates = useMemo(() => states.filter((s) => s.descriptor_file), [states]);

  useEffect(() => {
    if (!states.length) {
      setStatePair({ a: null, b: null });
      return;
    }
    const currentA = descriptorStates.find((s) => s.state_id === statePair.a);
    const currentB = descriptorStates.find((s) => s.state_id === statePair.b);
    if (descriptorStates.length >= 2) {
      if (currentA && currentB) return;
      setStatePair({ a: descriptorStates[0].state_id, b: descriptorStates[1].state_id });
      return;
    }
    if (descriptorStates.length === 1) {
      if (currentA && !statePair.b) return;
      setStatePair({ a: descriptorStates[0].state_id, b: null });
      return;
    }
    if (!statePair.a && states.length >= 2) {
      setStatePair({ a: states[0].state_id, b: states[1].state_id });
    }
  }, [states, descriptorStates, statePair.a, statePair.b]);

  const refreshSystem = async () => {
    try {
      const data = await fetchSystem(projectId, systemId);
      setSystem(data);
    } catch (err) {
      setActionError(err.message);
    }
  };

  const enqueueJob = async (runner, params) => {
    const stateA = descriptorStates.find((s) => s.state_id === statePair.a);
    const stateB = descriptorStates.find((s) => s.state_id === statePair.b);
    if (!stateA || !stateB) {
      setAnalysisError('Pick two states with built descriptors to run analysis.');
      return;
    }
    setAnalysisError(null);
    const response = await runner({
      project_id: projectId,
      system_id: systemId,
      state_a_id: stateA.state_id,
      state_b_id: stateB.state_id,
      ...params,
    });
    navigate(`/jobs/${response.job_id}`, { state: { analysis_uuid: response.analysis_uuid } });
  };

  const handleUploadTrajectory = async (stateId, file, stride) => {
    if (!file) return;
    setUploadingState(stateId);
    setActionError(null);
    try {
      const payload = new FormData();
      payload.append('trajectory', file);
      payload.append('stride', stride || 1);
      await uploadStateTrajectory(projectId, systemId, stateId, payload, {
        onUploadProgress: (percent) =>
          setUploadProgress((prev) => ({
            ...prev,
            [stateId]: percent,
          })),
        onProcessing: (processing) => setProcessingState(processing ? stateId : null),
      });
      setActionMessage('Uploaded trajectory; rebuilding descriptors...');
      await refreshSystem();
    } catch (err) {
      setActionError(err.message);
    } finally {
      setUploadingState(null);
      setProcessingState(null);
      setUploadProgress((prev) => {
        const next = { ...prev };
        delete next[stateId];
        return next;
      });
    }
  };

  const handleDeleteTrajectory = async (stateId) => {
    if (!window.confirm('Remove the trajectory and descriptors for this state?')) return;
    setActionError(null);
    try {
      await deleteStateTrajectory(projectId, systemId, stateId);
      setActionMessage('Removed trajectory and descriptors.');
      await refreshSystem();
    } catch (err) {
      setActionError(err.message);
    }
  };

  const handleDeleteState = async (stateId, name) => {
    if (!window.confirm(`Delete state "${name}" and its files?`)) return;
    setActionError(null);
    try {
      await deleteState(projectId, systemId, stateId);
      setActionMessage(`State "${name}" deleted.`);
      await refreshSystem();
    } catch (err) {
      setActionError(err.message);
    }
  };

  const handleDownloadStructure = async (stateId, name) => {
    setDownloadError(null);
    try {
      const blob = await downloadStructure(projectId, systemId, stateId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${system?.name || systemId}-${name || stateId}.pdb`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadError(err.message);
    }
  };

  const handleAddState = async ({ name, file, copyFrom }) => {
    setActionError(null);
    setAddingState(true);
    try {
      const payload = new FormData();
      payload.append('name', name);
      if (file) payload.append('pdb', file);
      if (copyFrom) payload.append('source_state_id', copyFrom);
      await addSystemState(projectId, systemId, payload);
      setActionMessage(`State "${name}" added.`);
      await refreshSystem();
    } catch (err) {
      setActionError(err.message);
    } finally {
      setAddingState(false);
    }
  };

  if (isLoading) return <Loader message="Loading system..." />;
  if (pageError) return <ErrorMessage message={pageError} />;
  if (!system) return null;

  return (
    <div className="space-y-8">
      <div>
        <button onClick={() => navigate('/projects')} className="text-cyan-400 hover:text-cyan-300 text-sm mb-2">
          ← Back to Projects
        </button>
        <h1 className="text-2xl font-bold text-white">{system.name}</h1>
        <p className="text-gray-400 text-sm">{system.description || 'No description provided.'}</p>
      </div>

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">States</h2>
          <p className="text-xs text-gray-400">Status: {system.status}</p>
        </div>
        {downloadError && <ErrorMessage message={downloadError} />}
        {actionError && <ErrorMessage message={actionError} />}
        {actionMessage && <p className="text-sm text-emerald-400">{actionMessage}</p>}
        <div className="grid md:grid-cols-3 gap-4">
          <div className="md:col-span-2 grid sm:grid-cols-2 gap-3">
            {states.length === 0 && <p className="text-sm text-gray-400">No states yet.</p>}
            {states.map((state) => (
              <StateCard
                key={state.state_id}
                state={state}
                onDownload={() => handleDownloadStructure(state.state_id, state.name)}
                onUpload={handleUploadTrajectory}
                onDeleteTrajectory={() => handleDeleteTrajectory(state.state_id)}
                onDeleteState={() => handleDeleteState(state.state_id, state.name)}
                uploading={uploadingState === state.state_id}
                progress={uploadProgress[state.state_id]}
                processing={processingState === state.state_id}
              />
            ))}
          </div>
          <AddStateForm states={states} onAdd={handleAddState} isAdding={addingState} />
        </div>
      </section>

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Run Analysis</h2>
          <p className="text-xs text-gray-400">
            Choose two descriptor-ready states (found {descriptorStates.length}).
          </p>
        </div>
        {staticError && <ErrorMessage message={`Failed to load static jobs: ${staticError}`} />}
        {analysisError && <ErrorMessage message={analysisError} />}
        <StatePairSelector
          states={descriptorStates}
          value={statePair}
          onChange={(updater) => {
            setAnalysisError(null);
            setStatePair(updater);
          }}
        />
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-md font-semibold text-white mb-2">Static Reporters</h3>
            <StaticAnalysisForm onSubmit={(params) => enqueueJob(submitStaticJob, params)} />
          </div>
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-md font-semibold text-white mb-2">QUBO</h3>
            <QuboAnalysisForm staticOptions={staticResults} onSubmit={(params) => enqueueJob(submitQuboJob, params)} />
          </div>
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-md font-semibold text-white mb-2">Dynamic TE</h3>
            <DynamicAnalysisForm onSubmit={(params) => enqueueJob(submitDynamicJob, params)} />
          </div>
        </div>
      </section>
    </div>
  );
}

function StateCard({ state, onDownload, onUpload, onDeleteTrajectory, onDeleteState, uploading, progress, processing }) {
  const [file, setFile] = useState(null);
  const [stride, setStride] = useState(state?.stride || 1);

  const descriptorLabel = state?.descriptor_file ? 'Ready' : 'Not built';
  const trajectoryLabel = state?.source_traj || '—';

  return (
    <article className="bg-gray-900 rounded-lg p-4 border border-gray-700 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-md font-semibold text-white">{state.name}</h3>
          <p className="text-xs text-gray-500">{state.state_id?.slice(0, 8)}</p>
        </div>
        <button
          onClick={onDownload}
          disabled={!state?.pdb_file}
          className="flex items-center space-x-1 text-sm text-cyan-400 hover:text-cyan-300 disabled:opacity-50"
        >
          <Download className="h-4 w-4" />
          <span>PDB</span>
        </button>
      </div>

      <dl className="space-y-1 text-sm text-gray-300">
        <div className="flex justify-between">
          <dt>Trajectory</dt>
          <dd className="truncate text-gray-200">{trajectoryLabel}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Frames</dt>
          <dd>{state?.n_frames ?? 0}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Stride</dt>
          <dd>{state?.stride ?? 1}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Descriptors</dt>
          <dd className="truncate text-gray-200">{descriptorLabel}</dd>
        </div>
      </dl>

      {(uploading || progress !== undefined || processing) && (
        <div className="space-y-1 text-xs text-gray-300">
          <div className="flex items-center justify-between">
            <span>{processing ? 'Processing descriptors' : 'Uploading trajectory'}</span>
            <span>{progress !== undefined ? `${progress}%` : ''}</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full ${processing ? 'bg-amber-400 animate-pulse' : 'bg-cyan-500'}`}
              style={{ width: progress !== undefined ? `${progress}%` : '33%' }}
            />
          </div>
        </div>
      )}

      <div className="space-y-2">
        <div>
          <label className="block text-sm text-gray-300 mb-1">Upload trajectory</label>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="w-full text-sm text-gray-300"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">Stride</label>
          <input
            type="number"
            min={1}
            value={stride}
            onChange={(e) => setStride(Number(e.target.value) || 1)}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
          />
        </div>
        <button
          onClick={() => onUpload(state.state_id, file, stride)}
          disabled={uploading || !file}
          className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
        >
          {uploading ? 'Uploading...' : 'Upload & Build'}
        </button>
      </div>

      <div className="flex items-center justify-between text-xs">
        <button
          onClick={onDeleteTrajectory}
          className="text-red-300 hover:text-red-200 disabled:opacity-50"
          disabled={!state?.trajectory_file && !state?.descriptor_file}
        >
          Remove trajectory
        </button>
        <button onClick={onDeleteState} className="text-gray-400 hover:text-red-300">
          Delete state
        </button>
      </div>
    </article>
  );
}

function StatePairSelector({ states, value, onChange }) {
  if (states.length < 2) {
    return <p className="text-sm text-gray-400">Upload trajectories for at least two states to run analyses.</p>;
  }
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <p className="text-sm text-gray-200 mb-2">Select two states to compare</p>
      <div className="grid md:grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-gray-400 mb-1">State A</label>
          <select
            value={value.a || ''}
            onChange={(e) => onChange((prev) => ({ ...prev, a: e.target.value || null }))}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          >
            <option value="">Choose state</option>
            {states.map((state) => (
              <option key={state.state_id} value={state.state_id}>
                {state.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">State B</label>
          <select
            value={value.b || ''}
            onChange={(e) => onChange((prev) => ({ ...prev, b: e.target.value || null }))}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          >
            <option value="">Choose state</option>
            {states.map((state) => (
              <option key={state.state_id} value={state.state_id}>
                {state.name}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}

function AddStateForm({ states, onAdd, isAdding }) {
  const [name, setName] = useState('');
  const [file, setFile] = useState(null);
  const [copyFrom, setCopyFrom] = useState('');
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    if (!file && !copyFrom) return;
    await onAdd({ name: name.trim(), file, copyFrom });
    setName('');
    setFile(null);
    setCopyFrom('');
  };
  return (
    <form onSubmit={handleSubmit} className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-3">
      <h3 className="text-md font-semibold text-white">Add State</h3>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          required
        />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Upload PDB</label>
        <input
          type="file"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="w-full text-sm text-gray-300"
        />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Or copy existing PDB</label>
        <select
          value={copyFrom}
          onChange={(e) => setCopyFrom(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        >
          <option value="">—</option>
          {states.map((state) => (
            <option key={state.state_id} value={state.state_id}>
              {state.name}
            </option>
          ))}
        </select>
      </div>
      <p className="text-xs text-gray-500">
        Provide a new PDB or select an existing state to duplicate its structure.
      </p>
      <button
        type="submit"
        disabled={isAdding || (!file && !copyFrom)}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isAdding ? 'Adding…' : 'Add State'}
      </button>
    </form>
  );
}
