import { useEffect, useMemo, useRef, useState } from 'react';
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
  renameState, // Import the new renameState function
  fetchMetastableStates,
  recomputeMetastableStates,
  renameMetastableState,
  submitMetastableClusterJob,
  renameMetastableCluster,
  confirmMacroStates,
  confirmMetastableStates,
  clearMetastableStates,
  downloadSavedCluster,
  deleteSavedCluster,
} from '../api/projects';
import { submitStaticJob, submitSimulationJob, fetchJobStatus } from '../api/jobs';
import StaticAnalysisForm from '../components/analysis/StaticAnalysisForm';
import SimulationAnalysisForm from '../components/analysis/SimulationAnalysisForm';
import { Download, Eye, Info, Plus, X, Pencil } from 'lucide-react';

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

function MetastableCard({ meta, onRename }) {
  const [name, setName] = useState(meta.name || meta.default_name || `Meta ${meta.metastable_index}`);
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    if (!meta.metastable_id) return;
    setIsSaving(true);
    try {
      await onRename(meta.metastable_id, name);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-400">{meta.macro_state}</p>
        <span className="text-xs text-gray-500">Frames: {meta.n_frames ?? '—'}</span>
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Name</label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white text-sm"
        />
      </div>
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span>ID: {meta.metastable_id || meta.metastable_index}</span>
        <button
          onClick={handleSave}
          disabled={isSaving || !meta.metastable_id}
          className="text-cyan-400 hover:text-cyan-300 disabled:opacity-50"
        >
          {isSaving ? 'Saving…' : 'Save'}
        </button>
      </div>
      {meta.representative_pdb && (
        <p className="text-xs text-cyan-400 break-all">
          Representative PDB: {meta.representative_pdb}
        </p>
      )}
    </div>
  );
}

const DOC_FILES = {
  metastable_states: '/docs/metastable_states.md',
  vamp_tica: '/docs/vamp_tica.md',
  msm: '/docs/msm.md',
  pcca: '/docs/pcca.md',
  potts_overview: '/docs/potts_overview.md',
  potts_model: '/docs/potts_model.md',
  potts_gibbs: '/docs/potts_gibbs.md',
  potts_sa_qubo: '/docs/potts_sa_qubo.md',
  potts_beta_eff: '/docs/potts_beta_eff.md',
};

function getClusterDisplayName(run) {
  if (!run) return 'Cluster';
  return run.name || run.path?.split('/').pop() || run.cluster_id || 'Cluster';
}

function formatClusterAlgorithm(run) {
  if (!run) return '';
  const algo = (run.cluster_algorithm || '').toLowerCase();
  const params = run.algorithm_params || {};
  if (algo === 'dbscan') {
    return `dbscan (eps=${params.eps ?? '—'}, min_samples=${params.min_samples ?? '—'})`;
  }
  if (algo === 'hierarchical') {
    return `hierarchical (n_clusters=${params.n_clusters ?? '—'}, linkage=${params.linkage || 'ward'})`;
  }
  if (algo === 'tomato') {
    return `tomato (k=${params.k_neighbors ?? '—'}, tau=${params.tau ?? '—'}, k_max=${params.k_max ?? '—'})`;
  }
  if (algo === 'density_peaks' || algo === 'kmeans') {
    return `${algo} (max_clusters=${run.max_clusters_per_residue ?? '—'})`;
  }
  return algo ? `${algo}` : 'cluster';
}

function DocOverlay({ docId, onClose, onNavigate }) {
  const cacheRef = useRef({});
  const [docState, setDocState] = useState({
    title: 'Documentation',
    blocks: [],
    loading: true,
    error: null,
  });

  useEffect(() => {
    let isMounted = true;
    const targetId = DOC_FILES[docId] ? docId : 'metastable_states';
    const cached = cacheRef.current[targetId];
    if (cached) {
      setDocState({ ...cached, loading: false, error: null });
      return () => {};
    }
    setDocState((prev) => ({ ...prev, loading: true, error: null }));
    fetch(DOC_FILES[targetId])
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load ${targetId} documentation.`);
        return res.text();
      })
      .then((text) => {
        const parsed = parseMarkdown(text);
        if (!isMounted) return;
        cacheRef.current[targetId] = parsed;
        setDocState({ ...parsed, loading: false, error: null });
      })
      .catch((err) => {
        if (!isMounted) return;
        setDocState((prev) => ({
          ...prev,
          loading: false,
          error: err.message || 'Failed to load documentation.',
        }));
      });
    return () => {
      isMounted = false;
    };
  }, [docId]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
      <div className="w-full max-w-3xl bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-lg font-semibold text-white">{docState.title}</h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-5 max-h-[75vh] overflow-y-auto space-y-4 text-sm text-gray-200">
          {docState.loading && <p className="text-gray-400">Loading documentation...</p>}
          {docState.error && <ErrorMessage message={docState.error} />}
          {!docState.loading &&
            !docState.error &&
            docState.blocks.map((block, idx) => renderDocBlock(block, idx, onNavigate))}
        </div>
      </div>
    </div>
  );
}

export default function SystemDetailPage() {
  const { projectId, systemId } = useParams();
  const [system, setSystem] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [pageError, setPageError] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [downloadError, setDownloadError] = useState(null);
  const [uploadingState, setUploadingState] = useState(null);
  const [addingState, setAddingState] = useState(false);
  const [actionMessage, setActionMessage] = useState(null);
  const [uploadProgress, setUploadProgress] = useState({});
  const [processingState, setProcessingState] = useState(null);
  const [statePair, setStatePair] = useState({ a: null, b: null });
  const [metastable, setMetastable] = useState({ states: [], model_dir: null });
  const [metaLoading, setMetaLoading] = useState(false);
  const [metaError, setMetaError] = useState(null);
  const [metaActionError, setMetaActionError] = useState(null);
  const [metaParamsOpen, setMetaParamsOpen] = useState(false);
  const [metaParams, setMetaParams] = useState({
    n_microstates: 20,
    k_meta_min: 1,
    k_meta_max: 4,
    tica_lag_frames: 5,
    tica_dim: 5,
    random_state: 0,
  });
  const [selectedMetastableIds, setSelectedMetastableIds] = useState([]);
  const [clusterError, setClusterError] = useState(null);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [clusterPanelOpen, setClusterPanelOpen] = useState(false);
  const [clusterName, setClusterName] = useState('');
  const [clusterDetailState, setClusterDetailState] = useState(null);
  const [clusterJobStatus, setClusterJobStatus] = useState({});
  const [maxClustersPerResidue, setMaxClustersPerResidue] = useState(6);
  const [maxClusterFrames, setMaxClusterFrames] = useState(0);
  const [contactMode, setContactMode] = useState('CA');
  const [contactCutoff, setContactCutoff] = useState(10);
  const [clusterAlgorithm, setClusterAlgorithm] = useState('DENSITY_PEAKS');
  const [dbscanEps, setDbscanEps] = useState(0.5);
  const [dbscanMinSamples, setDbscanMinSamples] = useState(5);
  const [hierClusters, setHierClusters] = useState(4);
  const [hierLinkage, setHierLinkage] = useState('ward');
  const [tomatoK, setTomatoK] = useState(15);
  const [tomatoTauMode, setTomatoTauMode] = useState('auto');
  const [tomatoTauValue, setTomatoTauValue] = useState(0.5);
  const [tomatoKMax, setTomatoKMax] = useState(6);
  const [densityZMode, setDensityZMode] = useState('auto');
  const [densityZValue, setDensityZValue] = useState(1.65);
  const [densityMaxk, setDensityMaxk] = useState(100);
  const [metaChoice, setMetaChoice] = useState(null);
  const [showMetastableInfo, setShowMetastableInfo] = useState(false);
  const [docId, setDocId] = useState('metastable_states');
  const [macroChoiceLoading, setMacroChoiceLoading] = useState(false);
  const [infoOverlayState, setInfoOverlayState] = useState(null); // New state for info overlay
  const navigate = useNavigate();

  useEffect(() => {
    const loadSystem = async () => {
      setIsLoading(true);
      setPageError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
        setActionMessage(null);
        setMetastable({
          states: data.metastable_states || [],
          model_dir: data.metastable_model_dir || null,
        });
      } catch (err) {
        setPageError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    loadSystem();
  }, [projectId, systemId]);

  useEffect(() => {
    loadMetastable();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, systemId]);


  const states = useMemo(() => Object.values(system?.states || {}), [system]);
  const descriptorStates = useMemo(() => states.filter((s) => s.descriptor_file), [states]);
  const metastableStates = useMemo(() => metastable.states || [], [metastable.states]);
  const macroLocked = system?.macro_locked;
  const metastableLocked = system?.metastable_locked;
  const analysisMode = system?.analysis_mode || null;
  const clusterRuns = useMemo(() => system?.metastable_clusters || [], [system]);
  const readyClusterRuns = useMemo(
    () => clusterRuns.filter((run) => run.path && run.status !== 'failed'),
    [clusterRuns]
  );
  const descriptorsReady = useMemo(
    () => states.length > 0 && states.every((state) => state.descriptor_file),
    [states]
  );
  const macroReady = Boolean(macroLocked && descriptorsReady);
  const showSidebar = macroReady;
  const effectiveChoice = analysisMode || metaChoice;
  const clustersUnlocked = macroReady && (analysisMode === 'macro' || metastableLocked);
  const analysisUnlocked = macroReady && (metastableLocked || effectiveChoice === 'macro');
  const showPottsPanel = metastableLocked || effectiveChoice === 'macro';
  const analysisNote =
    metastableLocked || effectiveChoice === 'macro'
      ? 'Static needs two descriptor-ready states; Potts uses saved cluster NPZ files.'
      : 'Static needs two descriptor-ready states. Potts requires metastable discovery and saved cluster NPZ files.';
  const showMacroChoicePanel =
    macroReady && !metastableLocked && effectiveChoice !== 'macro' && analysisMode !== 'metastable';
  const showMetastablePanel = macroReady && effectiveChoice !== 'macro' && !metastableLocked;
  const clusterSelectableStates = useMemo(() => {
    if (metastableLocked) return metastableStates;
    if (analysisMode === 'macro') {
      return states.map((state) => ({
        metastable_id: state.state_id,
        metastable_index: 0,
        macro_state_id: state.state_id,
        macro_state: state.name,
        name: state.name,
        default_name: state.name,
        n_frames: state.n_frames ?? 0,
      }));
    }
    return metastableStates;
  }, [analysisMode, metastableLocked, metastableStates, states]);
  const selectedClusterFrameCount = useMemo(() => {
    const framesById = new Map();
    clusterSelectableStates.forEach((m) => {
      const metaId = m.metastable_id || `${m.macro_state}-${m.metastable_index}`;
      const count = Number(m.n_frames ?? 0);
      framesById.set(metaId, Number.isFinite(count) ? count : 0);
    });
    return selectedMetastableIds.reduce((sum, id) => sum + (framesById.get(id) || 0), 0);
  }, [clusterSelectableStates, selectedMetastableIds]);
  const clusterFrameCap = useMemo(() => {
    if (!selectedClusterFrameCount) return 0;
    if (maxClusterFrames > 0) return Math.min(maxClusterFrames, selectedClusterFrameCount);
    return selectedClusterFrameCount;
  }, [maxClusterFrames, selectedClusterFrameCount]);
  const activeClusterJobs = useMemo(
    () =>
      clusterRuns.filter(
        (run) =>
          run.job_id &&
          !run.path &&
          run.status !== 'failed' &&
          run.status !== 'finished'
      ),
    [clusterRuns]
  );

  useEffect(() => {
    setSelectedMetastableIds((prev) =>
      prev.filter((id) =>
        clusterSelectableStates.some((m) => {
          const metaId = m.metastable_id || `${m.macro_state}-${m.metastable_index}`;
          return metaId === id;
        })
      )
    );
  }, [clusterSelectableStates]);

  useEffect(() => {
    if (!activeClusterJobs.length) return;
    let cancelled = false;
    const poll = async () => {
      let shouldRefresh = false;
      const updates = {};
      for (const run of activeClusterJobs) {
        if (!run.job_id) continue;
        try {
          const status = await fetchJobStatus(run.job_id);
          updates[run.cluster_id] = status;
          if (status.status === 'finished' || status.status === 'failed') {
            shouldRefresh = true;
          }
        } catch (err) {
          updates[run.cluster_id] = { status: 'failed', result: { error: err.message } };
          shouldRefresh = true;
        }
      }
      if (!cancelled && Object.keys(updates).length) {
        setClusterJobStatus((prev) => ({ ...prev, ...updates }));
      }
      if (!cancelled && shouldRefresh) {
        await refreshSystem();
      }
    };
    poll();
    const timer = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeClusterJobs]);

  useEffect(() => {
    if (!macroReady) {
      setMetaChoice(null);
      return;
    }
    if (analysisMode) {
      setMetaChoice(analysisMode);
      return;
    }
    if (!metaChoice && metastableStates.length > 0) {
      setMetaChoice('metastable');
    }
  }, [analysisMode, macroReady, metaChoice, metastableStates.length]);

  const openDoc = (nextDocId = 'metastable_states') => {
    setDocId(nextDocId);
    setShowMetastableInfo(true);
  };

  const analysisStateOptions = useMemo(() => {
    const macroOpts = descriptorStates.map((s) => ({
      kind: 'macro',
      id: s.state_id,
      macroId: s.state_id,
      label: s.name,
    }));
    if (!metastableLocked) {
      return macroOpts;
    }
    const metaOpts = metastableStates
      .filter((m) => m.macro_state_id)
      .map((m) => ({
        kind: 'meta',
        id: m.metastable_id,
        macroId: m.macro_state_id,
        label: `[Meta] ${m.name || m.default_name || m.metastable_id} (${m.macro_state})`,
      }));
    return [...macroOpts, ...metaOpts];
  }, [descriptorStates, metastableStates, metastableLocked]);

  useEffect(() => {
    const macroOpts = analysisStateOptions.filter((o) => o.kind === 'macro');
    if (!macroOpts.length) {
      setStatePair({ a: null, b: null });
      return;
    }
    if (statePair.a && statePair.b) return;
    const a = macroOpts[0] || null;
    const b = macroOpts[1] || null;
    setStatePair({ a, b });
  }, [analysisStateOptions, statePair.a, statePair.b]);


  const refreshSystem = async () => {
    try {
      const data = await fetchSystem(projectId, systemId);
      setSystem(data);
      setMetastable({
        states: data.metastable_states || [],
        model_dir: data.metastable_model_dir || null,
      });
    } catch (err) {
      setActionError(err.message);
    }
  };

  const loadMetastable = async () => {
    setMetaLoading(true);
    setMetaError(null);
    try {
      const data = await fetchMetastableStates(projectId, systemId);
      setMetastable({ states: data.metastable_states || [], model_dir: data.model_dir || null });
    } catch (err) {
      setMetaError(err.message);
    } finally {
      setMetaLoading(false);
    }
  };

  const enqueueStaticJob = async (runner, params) => {
    const stateAId = statePair.a?.macroId;
    const stateBId = statePair.b?.macroId;
    const stateA = descriptorStates.find((s) => s.state_id === stateAId);
    const stateB = descriptorStates.find((s) => s.state_id === stateBId);
    if (!stateA || !stateB) {
      setAnalysisError('Pick two states with built descriptors to run analysis.');
      return;
    }
    const extraParams = { ...params };
    if (statePair.a?.kind === 'meta') extraParams.metastable_a_id = statePair.a.id;
    if (statePair.b?.kind === 'meta') extraParams.metastable_b_id = statePair.b.id;
    setAnalysisError(null);
    const response = await runner({
      project_id: projectId,
      system_id: systemId,
      state_a_id: stateA.state_id,
      state_b_id: stateB.state_id,
      ...extraParams,
    });
    navigate(`/jobs/${response.job_id}`, { state: { analysis_uuid: response.analysis_uuid } });
  };

  const enqueueSimulationJob = async (params) => {
    setAnalysisError(null);
    const response = await submitSimulationJob({
      project_id: projectId,
      system_id: systemId,
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

  const handleRunMetastable = async () => {
    setMetaActionError(null);
    setMetaLoading(true);
    try {
      const payload = {
        ...metaParams,
        k_meta_max: Math.max(metaParams.k_meta_min, metaParams.k_meta_max),
      };
      await recomputeMetastableStates(projectId, systemId, payload);
      await refreshSystem();
      await loadMetastable();
    } catch (err) {
      setMetaActionError(err.message);
    } finally {
      setMetaLoading(false);
    }
  };

  const handleRenameMacroState = async (stateId, newName) => {
    if (!newName.trim()) return;
    setActionError(null);
    try {
      await renameState(projectId, systemId, stateId, newName.trim());
      await refreshSystem();
      setInfoOverlayState(null); // Close overlay after rename
    } catch (err) {
      setActionError(err.message);
    }
  };

  const handleRenameMetastable = async (metastableId, name) => {
    if (!name.trim()) return;
    setMetaActionError(null);
    try {
      await renameMetastableState(projectId, systemId, metastableId, name.trim());
      await loadMetastable();
      setInfoOverlayState(null); // Close overlay after rename
    } catch (err) {
      setMetaActionError(err.message);
    }
  };

  const handleConfirmMacro = async () => {
    setActionError(null);
    try {
      await confirmMacroStates(projectId, systemId);
      await refreshSystem();
    } catch (err) {
      setActionError(err.message);
    }
  };

  const handleConfirmMetastable = async () => {
    setMetaActionError(null);
    try {
      await confirmMetastableStates(projectId, systemId);
      await refreshSystem();
      await loadMetastable();
    } catch (err) {
      setMetaActionError(err.message);
    }
  };

  const handleConfirmMacroOnly = async () => {
    setMetaActionError(null);
    setMacroChoiceLoading(true);
    try {
      await clearMetastableStates(projectId, systemId);
      setMetaChoice('macro');
      await refreshSystem();
      await loadMetastable();
    } catch (err) {
      setMetaActionError(err.message);
    } finally {
      setMacroChoiceLoading(false);
    }
  };

  const toggleMetastableSelection = (metastableId) => {
    setClusterError(null);
    setSelectedMetastableIds((prev) =>
      prev.includes(metastableId) ? prev.filter((id) => id !== metastableId) : [...prev, metastableId]
    );
  };

  const handleDownloadClusters = async () => {
    if (!selectedMetastableIds.length) {
      setClusterError(
        analysisMode === 'macro'
          ? 'Select at least one macro-state to cluster.'
          : 'Select at least one metastable state to cluster.'
      );
      return;
    }
    setClusterError(null);
    setClusterLoading(true);
    try {
      const trimmedName = clusterName.trim();
      const algo = (clusterAlgorithm || 'DENSITY_PEAKS').toLowerCase();
      const algorithmParams = {};
      if (algo === 'dbscan') {
        algorithmParams.eps = dbscanEps;
        algorithmParams.min_samples = dbscanMinSamples;
      } else if (algo === 'tomato') {
        algorithmParams.k_neighbors = tomatoK;
        algorithmParams.tau = tomatoTauMode === 'auto' ? 'auto' : tomatoTauValue;
        algorithmParams.k_max = tomatoKMax;
      } else if (algo === 'density_peaks') {
        if (densityZMode === 'manual') {
          algorithmParams.Z = densityZValue;
        } else {
          algorithmParams.Z = 'auto';
        }
        algorithmParams.maxk = densityMaxk;
        } else if (algo === 'hierarchical') {
          algorithmParams.n_clusters = hierClusters;
          algorithmParams.linkage = hierLinkage;
        }
      await submitMetastableClusterJob(projectId, systemId, selectedMetastableIds, {
        cluster_name: trimmedName || undefined,
        max_clusters_per_residue: maxClustersPerResidue,
        max_cluster_frames: maxClusterFrames > 0 ? maxClusterFrames : undefined,
        contact_atom_mode: contactMode,
        contact_cutoff: contactCutoff,
        cluster_algorithm: algo,
        algorithm_params: algorithmParams,
        dbscan_eps: dbscanEps,
        dbscan_min_samples: dbscanMinSamples,
      });
      setClusterPanelOpen(false);
      setClusterName('');
      await refreshSystem();
    } catch (err) {
      setClusterError(err.message);
    } finally {
      setClusterLoading(false);
    }
  };

  const handleRenameCluster = async (clusterId, name) => {
    if (!name.trim()) return;
    setClusterError(null);
    try {
      await renameMetastableCluster(projectId, systemId, clusterId, name.trim());
      setClusterDetailState((prev) =>
        prev && prev.cluster_id === clusterId ? { ...prev, name: name.trim() } : prev
      );
      await refreshSystem();
    } catch (err) {
      setClusterError(err.message);
    }
  };

  const handleDownloadSavedCluster = async (clusterId, filename) => {
    setClusterError(null);
    try {
      const blob = await downloadSavedCluster(projectId, systemId, clusterId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || `metastable_clusters_${clusterId}.npz`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setClusterError(err.message);
    }
  };

  const handleDeleteSavedCluster = async (clusterId) => {
    if (!window.confirm('Delete this cluster NPZ?')) return;
    setClusterError(null);
    try {
      await deleteSavedCluster(projectId, systemId, clusterId);
      await refreshSystem();
      setClusterDetailState((prev) => (prev && prev.cluster_id === clusterId ? null : prev));
    } catch (err) {
      setClusterError(err.message);
    }
  };

  const handleCloseInfoOverlay = () => {
    setInfoOverlayState(null);
  };

  return (
    <div className="space-y-8">
      <div>
        <button onClick={() => navigate('/projects')} className="text-cyan-400 hover:text-cyan-300 text-sm mb-2">
          ← Back to Projects
        </button>
        <h1 className="text-2xl font-bold text-white">{system.name}</h1>
        <p className="text-gray-400 text-sm">{system.description || 'No description provided.'}</p>
      </div>

      <div className={showSidebar ? 'lg:grid lg:grid-cols-[260px_1fr] gap-6' : ''}>
        {showSidebar && (
          <aside className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4 lg:sticky lg:top-6 h-fit">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-gray-500">System state</p>
              <h2 className="text-base font-semibold text-white mt-2">Macro-states</h2>
              <p className="text-xs text-gray-400">
                {metastableLocked ? 'Metastable locked' : 'Metastable pending'}
              </p>
            </div>
            <button
              type="button"
              onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/descriptors/visualize`)}
              className="w-full text-xs px-3 py-2 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10 inline-flex items-center justify-center gap-2"
            >
              <Eye className="h-4 w-4" />
              Visualize descriptors
            </button>
            <div className="space-y-2">
              {states.map((state) => {
                const metaForState = metastableStates.filter((m) => m.macro_state_id === state.state_id);
                return (
                  <div key={state.state_id} className="rounded-md border border-gray-700 bg-gray-900/60 px-3 py-2">
                    {metastableLocked && metaForState.length > 0 ? (
                      <details className="group">
                        <summary className="flex items-center justify-between cursor-pointer text-sm text-gray-200">
                          <span className="flex items-center gap-2">
                            {state.name}
                            <button
                              type="button"
                              onClick={(event) => {
                                event.preventDefault();
                                event.stopPropagation();
                                setInfoOverlayState({ type: 'macro', data: state });
                              }}
                              className="text-gray-400 hover:text-cyan-300"
                              aria-label={`Show info for ${state.name}`}
                            >
                              <Info className="h-4 w-4" />
                            </button>
                          </span>
                          <span className="text-xs text-gray-400">{metaForState.length} meta</span>
                        </summary>
                        <div className="mt-2 space-y-1 text-xs text-gray-300">
                          {metaForState.map((m) => (
                            <div key={m.metastable_id} className="flex justify-between items-center">
                              <span>{m.name || m.default_name || m.metastable_id}</span>
                              <div className="flex items-center gap-2">
                                <span className="text-gray-500">#{m.metastable_index}</span>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.preventDefault();
                                    event.stopPropagation();
                                    setInfoOverlayState({ type: 'meta', data: m });
                                  }}
                                  className="text-gray-400 hover:text-cyan-300"
                                  aria-label={`Show info for ${m.name || m.default_name || m.metastable_id}`}
                                >
                                  <Info className="h-4 w-4" />
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </details>
                    ) : (
                      <div className="flex items-center justify-between text-sm text-gray-200">
                        <span className="flex items-center gap-2">
                          {state.name}
                        </span>
                        <span className="text-xs text-gray-500">
                          {state.descriptor_file ? 'Descriptors ready' : 'Descriptors missing'}
                        </span>
                        <button
                          type="button"
                          onClick={() => setInfoOverlayState({ type: 'macro', data: state })}
                          className="text-gray-400 hover:text-cyan-300"
                          aria-label={`Show info for ${state.name}`}
                        >
                          <Info className="h-4 w-4" />
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {!clustersUnlocked && (
              <p className="text-xs text-gray-500">Confirm how to proceed to unlock clustering and analysis.</p>
            )}
            {clustersUnlocked && (
              <div className="border-t border-gray-700 pt-3 space-y-2">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-white">Cluster NPZ</h3>
                    <p className="text-xs text-gray-400">Residue cluster runs for Potts sampling.</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setClusterError(null);
                      setClusterPanelOpen(true);
                    }}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-gray-700 text-gray-200 hover:border-cyan-500 hover:text-cyan-300 disabled:opacity-50"
                    disabled={clusterLoading}
                    aria-label="Create new cluster run"
                  >
                    <Plus className="h-4 w-4" />
                  </button>
                </div>
                {analysisMode === 'macro' && (
                  <p className="text-[11px] text-cyan-300">
                    Macro-states are treated as metastable sets when clustering.
                  </p>
                )}
                {clusterError && <ErrorMessage message={clusterError} />}
                {clusterRuns.length === 0 && <p className="text-xs text-gray-400">No cluster runs yet.</p>}
                {clusterRuns.length > 0 && (
                  <div className="space-y-2">
                    {clusterRuns
                      .slice()
                      .reverse()
                      .map((run) => {
                        const name = getClusterDisplayName(run);
                        const jobSnapshot = clusterJobStatus[run.cluster_id];
                        const jobMeta = jobSnapshot?.meta || {};
                        const jobStatus = jobSnapshot?.status || run.status;
                        const statusLabel = jobMeta.status || run.status_message || run.status || 'queued';
                        const progress = jobMeta.progress ?? run.progress;
                        const isReady = Boolean(run.path) && (run.status === 'finished' || !run.status);
                        const isFailed = jobStatus === 'failed' || run.status === 'failed';
                        const isRunning = !isReady && !isFailed;
                        if (isReady) {
                          return (
                            <button
                              key={run.cluster_id}
                              type="button"
                              onClick={() => setClusterDetailState(run)}
                              className="w-full text-left rounded-md border px-3 py-2 border-gray-700 bg-gray-900/60 hover:border-cyan-500"
                            >
                              <div className="flex items-center justify-between text-xs">
                                <span className="text-gray-200">{name}</span>
                                <span className="text-[10px] text-gray-500">
                                  {run.cluster_algorithm ? run.cluster_algorithm.toUpperCase() : 'CLUSTER'}
                                </span>
                              </div>
                              <p className="text-[11px] text-gray-400 mt-1">
                                {analysisMode === 'macro' ? 'States' : 'Metastable'}:{' '}
                                {Array.isArray(run.metastable_ids) ? run.metastable_ids.length : '—'} |{' '}
                                Max clusters: {run.max_clusters_per_residue ?? '—'} | Max frames:{' '}
                                {run.max_cluster_frames ?? 'all'}
                              </p>
                            </button>
                          );
                        }

                        return (
                          <div
                            key={run.cluster_id}
                            className="w-full text-left rounded-md border px-3 py-2 border-gray-800 bg-gray-900/40"
                          >
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-gray-200">{name}</span>
                              <span className="text-[10px] text-gray-500">
                                {run.cluster_algorithm ? run.cluster_algorithm.toUpperCase() : 'CLUSTER'}
                              </span>
                            </div>
                            <p className="text-[11px] text-gray-400 mt-1">
                              {analysisMode === 'macro' ? 'States' : 'Metastable'}:{' '}
                              {Array.isArray(run.metastable_ids) ? run.metastable_ids.length : '—'} |{' '}
                              Max clusters: {run.max_clusters_per_residue ?? '—'} | Max frames:{' '}
                              {run.max_cluster_frames ?? 'all'}
                            </p>
                            {isRunning && (
                              <div className="mt-2 space-y-1 text-[11px] text-gray-400">
                                <div className="flex items-center justify-between">
                                  <span>{statusLabel}</span>
                                  <span>{Number.isFinite(progress) ? `${progress}%` : '—'}</span>
                                </div>
                                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-cyan-500 animate-pulse"
                                    style={{ width: Number.isFinite(progress) ? `${progress}%` : '40%' }}
                                  />
                                </div>
                              </div>
                            )}
                            {isFailed && (
                              <div className="mt-2 flex items-center justify-between text-[11px] text-red-300">
                                <span>{run.error || jobSnapshot?.result?.error || 'Clustering failed.'}</span>
                                <button
                                  type="button"
                                  onClick={(event) => {
                                    event.preventDefault();
                                    event.stopPropagation();
                                    handleDeleteSavedCluster(run.cluster_id);
                                  }}
                                  className="text-[11px] text-red-200 hover:text-red-100"
                                >
                                  Delete
                                </button>
                              </div>
                            )}
                          </div>
                        );
                      })}
                  </div>
                )}
              </div>
            )}
          </aside>
        )}

        <div className="space-y-8">
      {macroLocked && !showSidebar && (
        <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">System Snapshot</h2>
            {descriptorStates.length > 0 && (
              <button
                onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/descriptors/visualize`)}
                className="text-xs px-3 py-1 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10"
              >
                Visualize descriptors
              </button>
            )}
          </div>
          <div className="grid md:grid-cols-2 gap-3">
            <div className="bg-gray-900 border border-gray-700 rounded-md p-3">
              <p className="text-xs text-gray-400 mb-1">Macro-states (locked)</p>
              <ul className="space-y-1 text-sm text-gray-200">
                {states.map((s) => (
                  <li key={s.state_id} className="flex justify-between border-b border-gray-800 pb-1">
                    <span>{s.name}</span>
                    <span className="text-xs text-gray-400">
                      {s.descriptor_file ? 'Descriptors ready' : 'No descriptors'}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-gray-900 border border-gray-700 rounded-md p-3 space-y-1">
              <div className="flex items-center justify-between">
                <p className="text-xs text-gray-400">Metastable states</p>
                {metastableLocked ? (
                  <span className="text-emerald-300 text-xs">Locked</span>
                ) : (
                  <span className="text-amber-300 text-xs">Editable</span>
                )}
              </div>
              {metastableStates.length === 0 && (
                <p className="text-xs text-gray-500">Not computed yet.</p>
              )}
              {metastableStates.length > 0 && (
                <ul className="space-y-1 text-sm text-gray-200">
                  {metastableStates.map((m) => (
                    <li key={m.metastable_id || `${m.macro_state}-${m.metastable_index}`} className="flex justify-between border-b border-gray-800 pb-1">
                      <span>{m.name || m.default_name || m.metastable_id}</span>
                      <span className="text-xs text-gray-400">{m.macro_state}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </section>
      )}

      {!macroLocked && (
        <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">States</h2>
            <div className="flex items-center space-x-3">
              <p className="text-xs text-gray-400">Status: {system.status}</p>
              <button
                onClick={handleConfirmMacro}
                disabled={states.length === 0 || !descriptorsReady}
                className="text-xs px-3 py-1 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50"
              >
                Confirm states
              </button>
            </div>
          </div>
          {downloadError && <ErrorMessage message={downloadError} />}
          {actionError && <ErrorMessage message={actionError} />}
          {actionMessage && <p className="text-sm text-emerald-400">{actionMessage}</p>}
          {!descriptorsReady && states.length > 0 && (
            <p className="text-xs text-amber-300">
              Upload trajectories and build descriptors for every state before confirming.
            </p>
          )}
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
      )}

      {showMacroChoicePanel && (
        <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-white">Use macro-states only (optional)</h2>
              <p className="text-sm text-gray-400 mt-1">
                Your macro-states are confirmed. You can proceed with static analysis using only macro-states, or run
                the metastable algorithm manually below.
              </p>
            </div>
            <InfoTooltip
              ariaLabel="Metastable algorithm info"
              text="Metastable discovery uses per-residue dihedral descriptors, TICA projection, and k-means clustering."
              onClick={() => openDoc('metastable_states')}
            />
          </div>
          <div className="rounded-lg border border-gray-700 bg-gray-900/60 p-4 space-y-3">
            <p className="text-xs text-gray-400">
              If you prefer to skip metastable discovery, confirm macro-only. Any existing metastable results will be
              discarded.
            </p>
            <button
              type="button"
              onClick={handleConfirmMacroOnly}
              disabled={macroChoiceLoading || metaChoice === 'macro'}
              className="text-xs px-3 py-2 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-60"
            >
              {macroChoiceLoading
                ? 'Confirming…'
                : metaChoice === 'macro'
                ? 'Macro-only selected'
                : 'Use macro-states only'}
            </button>
            {metaActionError && <ErrorMessage message={metaActionError} />}
          </div>
        </section>
      )}

      {showMetastablePanel && (
        <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold text-white">Metastable States (TICA)</h2>
              <InfoTooltip
                ariaLabel="Metastable states info"
                text="Open detailed documentation for the metastable pipeline and related methods."
                onClick={() => openDoc('metastable_states')}
              />
            </div>
            <div className="flex items-center space-x-2">
              {!metastableLocked && (
                <button
                  onClick={() => setMetaParamsOpen((prev) => !prev)}
                  className="text-xs px-3 py-1 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
                >
                  Hyperparams
                </button>
              )}
              <button
                onClick={handleRunMetastable}
                disabled={metaLoading || metastableLocked}
                className="text-xs px-3 py-1 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10 disabled:opacity-50"
              >
                {metastableLocked ? 'Locked' : metaLoading ? 'Running…' : 'Recompute'}
              </button>
              <button
                onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/metastable/visualize`)}
                className="text-xs px-3 py-1 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10"
              >
                Visualize
              </button>
              {!metastableLocked && (
                <button
                  onClick={handleConfirmMetastable}
                  disabled={metaLoading || metastableStates.length === 0}
                  className="text-xs px-3 py-1 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50"
                >
                  Confirm metastable
                </button>
              )}
            </div>
          </div>
          {metaParamsOpen && !metastableLocked && (
            <div className="bg-gray-900 border border-gray-700 rounded-md p-3 space-y-3 text-sm">
              <div className="grid md:grid-cols-3 gap-3">
                {[
                  {
                    key: 'n_microstates',
                    label: 'Microstates (k-means)',
                    min: 2,
                    info: 'Number of k-means clusters in TICA space before coarse-graining.',
                  },
                  {
                    key: 'k_meta_min',
                    label: 'Metastable min k',
                    min: 1,
                    info: 'Minimum metastable state count to test via spectral gap.',
                  },
                  {
                    key: 'k_meta_max',
                    label: 'Metastable max k',
                    min: 1,
                    info: 'Maximum metastable state count to test via spectral gap.',
                  },
                  {
                    key: 'tica_lag_frames',
                    label: 'TICA lag (frames)',
                    min: 1,
                    info: 'Lag time in frames for TICA projection.',
                  },
                  {
                    key: 'tica_dim',
                    label: 'TICA dims',
                    min: 1,
                    info: 'Number of TICA components retained for clustering.',
                  },
                  {
                    key: 'random_state',
                    label: 'Random seed',
                    min: 0,
                    info: 'Seed for k-means and MSM initialization.',
                  },
                ].map((field) => (
                  <label key={field.key} className="space-y-1">
                    <span className="flex items-center gap-2 text-xs text-gray-400">
                      {field.label}
                      <InfoTooltip ariaLabel={`${field.label} info`} text={field.info} />
                    </span>
                    <input
                      type="number"
                      min={field.min}
                      value={metaParams[field.key]}
                      onChange={(e) =>
                        setMetaParams((prev) => ({
                          ...prev,
                          [field.key]: Number(e.target.value),
                        }))
                      }
                      className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                    />
                  </label>
                ))}
              </div>
            </div>
          )}
          {metaError && <ErrorMessage message={`Failed to load metastable states: ${metaError}`} />}
          {metaActionError && <ErrorMessage message={metaActionError} />}
          {metaLoading && <Loader message="Computing metastable states..." />}
          {!metaLoading && metastable.states.length === 0 && (
            <p className="text-sm text-gray-400">
              Metastable analysis is manual. Click Recompute after uploading trajectories and building descriptors.
            </p>
          )}
          {!metaLoading && metastableStates.length > 0 && (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
              {metastableStates.map((m) => (
                <MetastableCard key={m.metastable_id || `${m.macro_state}-${m.metastable_index}`} meta={m} onRename={handleRenameMetastable} />
              ))}
            </div>
          )}
        </section>
      )}

      {analysisUnlocked && (
        <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Run Analysis</h2>
            <p className="text-xs text-gray-400">{analysisNote}</p>
          </div>
          {analysisError && <ErrorMessage message={analysisError} />}
          <div className={showPottsPanel ? 'grid lg:grid-cols-2 gap-6' : ''}>
            <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
              <div>
                <h3 className="text-md font-semibold text-white">Static Reporters</h3>
                <p className="text-xs text-gray-500 mt-1">Select two descriptor-ready states to compare.</p>
              </div>
              <StatePairSelector
                options={analysisStateOptions}
                value={statePair}
                onChange={(updater) => {
                  setAnalysisError(null);
                  setStatePair(updater);
                }}
              />
              <StaticAnalysisForm onSubmit={(params) => enqueueStaticJob(submitStaticJob, params)} />
            </div>
            {showPottsPanel && (
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <h3 className="text-md font-semibold text-white">Potts Sampling</h3>
                  <InfoTooltip
                    ariaLabel="Potts analysis documentation"
                    text="Open documentation for the Potts model, sampling, and diagnostics."
                    onClick={() => openDoc('potts_overview')}
                  />
                </div>
                <SimulationAnalysisForm clusterRuns={readyClusterRuns} onSubmit={enqueueSimulationJob} />
              </div>
            )}
          </div>
        </section>
      )}
      {clusterPanelOpen && (
        <ClusterBuildOverlay
          metastableStates={clusterSelectableStates}
          selectedMetastableIds={selectedMetastableIds}
          onToggleMetastable={toggleMetastableSelection}
          clusterName={clusterName}
          setClusterName={setClusterName}
          analysisMode={analysisMode}
          clusterAlgorithm={clusterAlgorithm}
          setClusterAlgorithm={setClusterAlgorithm}
          tomatoK={tomatoK}
          setTomatoK={setTomatoK}
          tomatoTauMode={tomatoTauMode}
          setTomatoTauMode={setTomatoTauMode}
          tomatoTauValue={tomatoTauValue}
          setTomatoTauValue={setTomatoTauValue}
          tomatoKMax={tomatoKMax}
          setTomatoKMax={setTomatoKMax}
          densityZMode={densityZMode}
          setDensityZMode={setDensityZMode}
          densityZValue={densityZValue}
          setDensityZValue={setDensityZValue}
          densityMaxk={densityMaxk}
          setDensityMaxk={setDensityMaxk}
          maxClustersPerResidue={maxClustersPerResidue}
          setMaxClustersPerResidue={setMaxClustersPerResidue}
          maxClusterFrames={maxClusterFrames}
          setMaxClusterFrames={setMaxClusterFrames}
          selectedClusterFrameCount={selectedClusterFrameCount}
          clusterFrameCap={clusterFrameCap}
          dbscanEps={dbscanEps}
          setDbscanEps={setDbscanEps}
          dbscanMinSamples={dbscanMinSamples}
          setDbscanMinSamples={setDbscanMinSamples}
          hierClusters={hierClusters}
          setHierClusters={setHierClusters}
          hierLinkage={hierLinkage}
          setHierLinkage={setHierLinkage}
          contactMode={contactMode}
          setContactMode={setContactMode}
          contactCutoff={contactCutoff}
          setContactCutoff={setContactCutoff}
          clusterError={clusterError}
          clusterLoading={clusterLoading}
          onClose={() => setClusterPanelOpen(false)}
          onSubmit={handleDownloadClusters}
        />
      )}
      {clusterDetailState && (
        <ClusterDetailOverlay
          cluster={clusterDetailState}
          analysisMode={analysisMode}
          onClose={() => setClusterDetailState(null)}
          onRename={handleRenameCluster}
          onDownload={handleDownloadSavedCluster}
          onDelete={handleDeleteSavedCluster}
          onVisualize={(clusterId) =>
            navigate(
              `/projects/${projectId}/systems/${systemId}/descriptors/visualize?cluster_id=${encodeURIComponent(
                clusterId
              )}`
            )
          }
        />
      )}
      {showMetastableInfo && (
        <DocOverlay docId={docId} onClose={() => setShowMetastableInfo(false)} onNavigate={setDocId} />
      )}
      {infoOverlayState && (
        <InfoOverlay
          state={infoOverlayState.data}
          type={infoOverlayState.type}
          onClose={handleCloseInfoOverlay}
          onRenameMacro={handleRenameMacroState}
          onRenameMeta={handleRenameMetastable}
        />
      )}
        </div>
      </div>
    </div>
  );
}

function InfoTooltip({ text, ariaLabel, onClick }) {
  return (
    <span className="relative inline-flex group">
      <button
        type="button"
        className="inline-flex items-center justify-center text-gray-500 hover:text-gray-300 focus:outline-none"
        aria-label={ariaLabel}
        aria-haspopup={onClick ? 'dialog' : undefined}
        onClick={onClick}
      >
        <Info className="h-4 w-4" />
      </button>
      <span className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 w-72 -translate-x-1/2 rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-xs text-gray-200 opacity-0 shadow-lg transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
        {text}
      </span>
    </span>
  );
}

function ClusterBuildOverlay({
  metastableStates,
  selectedMetastableIds,
  onToggleMetastable,
  clusterName,
  setClusterName,
  analysisMode,
  clusterAlgorithm,
  setClusterAlgorithm,
  tomatoK,
  setTomatoK,
  tomatoTauMode,
  setTomatoTauMode,
  tomatoTauValue,
  setTomatoTauValue,
  tomatoKMax,
  setTomatoKMax,
  densityZMode,
  setDensityZMode,
  densityZValue,
  setDensityZValue,
  densityMaxk,
  setDensityMaxk,
  maxClustersPerResidue,
  setMaxClustersPerResidue,
  maxClusterFrames,
  setMaxClusterFrames,
  selectedClusterFrameCount,
  clusterFrameCap,
  dbscanEps,
  setDbscanEps,
  dbscanMinSamples,
  setDbscanMinSamples,
  hierClusters,
  setHierClusters,
  hierLinkage,
  setHierLinkage,
  contactMode,
  setContactMode,
  contactCutoff,
  setContactCutoff,
  clusterError,
  clusterLoading,
  onClose,
  onSubmit,
}) {
  const hasMetastable = metastableStates.length > 0;
  const canSubmit = selectedMetastableIds.length > 0 && !clusterLoading;
  const macroOnly = analysisMode === 'macro';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
      <div className="w-full max-w-5xl bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-lg font-semibold text-white">New Cluster NPZ</h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-5 max-h-[75vh] overflow-y-auto space-y-4 text-sm text-gray-200">
          <label className="space-y-1">
            <span className="block text-xs text-gray-400">Cluster name (optional)</span>
            <input
              type="text"
              value={clusterName}
              onChange={(e) => setClusterName(e.target.value)}
              placeholder="e.g. beta sweep cluster"
              className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white"
            />
          </label>

          <div className="space-y-2">
            <p className="text-xs text-gray-400">
              {macroOnly
                ? 'Select macro-states to cluster per-residue angles. Each macro-state contributes all its frames.'
                : 'Select metastable states to cluster per-residue angles. Generated NPZ files are saved for Potts sampling.'}
            </p>
            {!hasMetastable && (
              <p className="text-xs text-amber-300">
                {macroOnly ? 'No macro-states available.' : 'No metastable states available.'}
              </p>
            )}
            {hasMetastable && (
              <div className="flex flex-wrap gap-2">
                {metastableStates.map((m) => {
                  const metaId = m.metastable_id || `${m.macro_state}-${m.metastable_index}`;
                  const label = m.name || m.default_name || metaId;
                  const checked = selectedMetastableIds.includes(metaId);
                  return (
                    <label
                      key={metaId}
                      className={`flex items-center space-x-2 px-3 py-2 rounded-md border cursor-pointer ${
                        checked ? 'border-emerald-400 bg-emerald-500/10' : 'border-gray-700 bg-gray-800/60'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => onToggleMetastable(metaId)}
                        className="accent-emerald-400"
                      />
                      <span className="text-sm text-gray-200">{label}</span>
                    </label>
                  );
                })}
              </div>
            )}
          </div>

          <div className="grid md:grid-cols-4 gap-3 text-xs text-gray-300">
            <label className="space-y-1">
              <span className="block text-gray-400">Algorithm</span>
              <select
                value={clusterAlgorithm}
                onChange={(e) => setClusterAlgorithm(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
              >
                <option value="TOMATO">ToMATo (periodic)</option>
                <option value="DENSITY_PEAKS">Density Peaks (periodic)</option>
                <option value="DBSCAN">DBSCAN (periodic)</option>
                <option value="KMEANS">Periodic K-Means</option>
                <option value="HIERARCHICAL">Hierarchical (periodic)</option>
              </select>
            </label>
            <label className="space-y-1 md:col-span-2">
              <span className="block text-gray-400">Max frames to cluster (0 = all)</span>
              <input
                type="number"
                min={0}
                value={maxClusterFrames}
                onChange={(e) => setMaxClusterFrames(Math.max(0, Number(e.target.value) || 0))}
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
              />
              <p className="text-[11px] text-gray-400 leading-snug">
                Selected frames: {selectedClusterFrameCount || 0}. Clustering uses up to {clusterFrameCap || 0} frames
                per dataset (uniform sample). Remaining frames are assigned by nearest-neighbor majority.
              </p>
            </label>
            {clusterAlgorithm === 'TOMATO' && (
              <>
                <label className="space-y-1">
                  <span className="block text-gray-400">k neighbors</span>
                  <input
                    type="number"
                    min={1}
                    value={tomatoK}
                    onChange={(e) => setTomatoK(Math.max(1, Number(e.target.value) || 1))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  />
                </label>
                <label className="space-y-1">
                  <span className="block text-gray-400">Tau (persistence)</span>
                  <div className="space-y-2">
                    <select
                      value={tomatoTauMode}
                      onChange={(e) => setTomatoTauMode(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                    >
                      <option value="auto">Auto (gap heuristic)</option>
                      <option value="manual">Manual</option>
                    </select>
                    {tomatoTauMode === 'manual' ? (
                      <input
                        type="number"
                        min={0}
                        step="0.1"
                        value={tomatoTauValue}
                        onChange={(e) => setTomatoTauValue(Number(e.target.value) || 0)}
                        className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                      />
                    ) : (
                      <p className="text-[11px] text-gray-400 leading-snug">
                        Automatically picks tau from the largest persistence gap in the residue trajectory.
                      </p>
                    )}
                  </div>
                </label>
                <label className="space-y-1">
                  <span className="block text-gray-400">Max clusters / residue</span>
                  <input
                    type="number"
                    min={1}
                    max={12}
                    value={tomatoKMax}
                    onChange={(e) => setTomatoKMax(Math.max(1, Number(e.target.value) || 1))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  />
                </label>
              </>
            )}
            {clusterAlgorithm === 'DENSITY_PEAKS' && (
              <>
                <label className="space-y-1">
                  <span className="block text-gray-400">DADApy Z (merge factor)</span>
                  <div className="space-y-2">
                    <select
                      value={densityZMode}
                      onChange={(e) => setDensityZMode(e.target.value)}
                      className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                    >
                      <option value="auto">Auto (package default)</option>
                      <option value="manual">Manual</option>
                    </select>
                    {densityZMode === 'manual' ? (
                      <input
                        type="number"
                        step="0.05"
                        min={0}
                        value={densityZValue}
                        onChange={(e) => setDensityZValue(Number(e.target.value) || 0)}
                        className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                      />
                    ) : (
                      <p className="text-[11px] text-gray-400 leading-snug">Use DADApy's default Z if left on auto.</p>
                    )}
                  </div>
                </label>
                <label className="space-y-1">
                  <span className="block text-gray-400">Max neighbors (maxk)</span>
                  <input
                    type="number"
                    min={1}
                    max={1000}
                    value={densityMaxk}
                    onChange={(e) => setDensityMaxk(Math.max(1, Number(e.target.value) || 1))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  />
                  <p className="text-[11px] text-gray-400 leading-snug">Caps neighbor search; not a cluster count.</p>
                </label>
              </>
            )}
            {clusterAlgorithm === 'KMEANS' && (
              <label className="space-y-1">
                <span className="block text-gray-400">Max clusters / residue</span>
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={maxClustersPerResidue}
                  onChange={(e) => setMaxClustersPerResidue(Math.max(1, Number(e.target.value) || 1))}
                  className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                />
              </label>
            )}
            {clusterAlgorithm === 'DBSCAN' && (
              <>
                <label className="space-y-1">
                  <span className="block text-gray-400">DBSCAN eps</span>
                  <input
                    type="number"
                    min={0.01}
                    step="0.05"
                    value={dbscanEps}
                    onChange={(e) => setDbscanEps(Math.max(0.01, Number(e.target.value) || 0.01))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  />
                </label>
                <label className="space-y-1">
                  <span className="block text-gray-400">DBSCAN min samples</span>
                  <input
                    type="number"
                    min={1}
                    value={dbscanMinSamples}
                    onChange={(e) => setDbscanMinSamples(Math.max(1, Number(e.target.value) || 1))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  />
                </label>
              </>
            )}
            {clusterAlgorithm === 'HIERARCHICAL' && (
              <>
                <label className="space-y-1">
                  <span className="block text-gray-400">Clusters / residue</span>
                  <input
                    type="number"
                    min={1}
                    max={12}
                    value={hierClusters}
                    onChange={(e) => setHierClusters(Math.max(1, Number(e.target.value) || 1))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  />
                </label>
                <label className="space-y-1">
                  <span className="block text-gray-400">Linkage</span>
                  <select
                    value={hierLinkage}
                    onChange={(e) => setHierLinkage(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  >
                    <option value="ward">Ward</option>
                    <option value="complete">Complete</option>
                    <option value="average">Average</option>
                    <option value="single">Single</option>
                  </select>
                </label>
              </>
            )}
            <label className="space-y-1">
              <span className="block text-gray-400">Contact mode</span>
              <select
                value={contactMode}
                onChange={(e) => setContactMode(e.target.value)}
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
              >
                <option value="CA">CA</option>
                <option value="CM">Residue CM</option>
              </select>
            </label>
            <label className="space-y-1">
              <span className="block text-gray-400">Contact cutoff (A)</span>
              <input
                type="number"
                min={1}
                step="0.5"
                value={contactCutoff}
                onChange={(e) => setContactCutoff(Math.max(0.1, Number(e.target.value) || 0))}
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
              />
            </label>
            <div className="flex items-center">
              <p className="text-gray-300">
                Selected: <span className="text-white font-semibold">{selectedMetastableIds.length}</span> /{' '}
                {metastableStates.length}
              </p>
            </div>
          </div>

          <p className="text-gray-400 text-xs">
            NPZ includes merged cluster vectors, contact map edge_index (pyg format), and metadata JSON.
          </p>
          {clusterError && <ErrorMessage message={clusterError} />}
        </div>
        <div className="flex items-center justify-end gap-2 border-t border-gray-800 px-4 py-3">
          <button
            type="button"
            onClick={onClose}
            className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onSubmit}
            disabled={!canSubmit}
            className="text-xs px-3 py-2 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50"
          >
            {clusterLoading ? 'Generating...' : 'Run clustering'}
          </button>
        </div>
      </div>
    </div>
  );
}

function ClusterDetailOverlay({ cluster, analysisMode, onClose, onRename, onDownload, onDelete, onVisualize }) {
  const [name, setName] = useState(getClusterDisplayName(cluster));
  const [isSaving, setIsSaving] = useState(false);
  const stateLabel = analysisMode === 'macro' ? 'States' : 'Metastable';

  useEffect(() => {
    setName(getClusterDisplayName(cluster));
  }, [cluster]);

  const handleSave = async () => {
    if (!name.trim()) return;
    setIsSaving(true);
    try {
      await onRename(cluster.cluster_id, name.trim());
    } finally {
      setIsSaving(false);
    }
  };

  const algoSummary = formatClusterAlgorithm(cluster);
  const filename = cluster.path?.split('/').pop();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
      <div className="w-full max-w-2xl bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-lg font-semibold text-white">Cluster NPZ Details</h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-5 space-y-4 text-sm text-gray-200">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Cluster name</label>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="flex-1 bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white"
              />
              <button
                type="button"
                onClick={handleSave}
                disabled={isSaving || !name.trim()}
                className="text-xs px-3 py-2 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10 disabled:opacity-50"
              >
                {isSaving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>

          <div className="space-y-1 text-xs text-gray-400">
            <p>
              <span className="text-gray-300 font-semibold">ID:</span> {cluster.cluster_id}
            </p>
            <p>
              <span className="text-gray-300 font-semibold">Algorithm:</span> {algoSummary || '—'}
            </p>
            <p>
              <span className="text-gray-300 font-semibold">{stateLabel}:</span>{' '}
              {Array.isArray(cluster.metastable_ids) ? cluster.metastable_ids.join(', ') : '—'}
            </p>
            <p>
              <span className="text-gray-300 font-semibold">Max clusters:</span> {cluster.max_clusters_per_residue ?? '—'}
            </p>
            <p>
              <span className="text-gray-300 font-semibold">Max frames:</span> {cluster.max_cluster_frames ?? 'all'}
            </p>
            <p>
              <span className="text-gray-300 font-semibold">Contact:</span>{' '}
              {cluster.contact_atom_mode || cluster.contact_mode || 'CA'} @ {cluster.contact_cutoff ?? 10} A
            </p>
            <p>
              <span className="text-gray-300 font-semibold">Generated:</span> {cluster.generated_at || '—'}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2 border-t border-gray-800 px-4 py-3">
          <button
            type="button"
            onClick={() => onDownload(cluster.cluster_id, filename)}
            className="text-xs px-3 py-2 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10"
          >
            Download
          </button>
          <button
            type="button"
            onClick={() => onVisualize(cluster.cluster_id)}
            className="text-xs px-3 py-2 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10"
          >
            Visualize
          </button>
          <button
            type="button"
            onClick={() => onDelete(cluster.cluster_id)}
            className="text-xs px-3 py-2 rounded-md border border-red-500 text-red-300 hover:bg-red-500/10"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

// New InfoOverlay component
function InfoOverlay({ state, type, onClose, onRenameMacro, onRenameMeta }) {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(state.name || state.default_name || '');
  const isMetastable = type === 'meta';

  const handleSave = async () => {
    if (isMetastable) {
      await onRenameMeta(state.metastable_id, name);
    } else {
      await onRenameMacro(state.state_id, name);
    }
    setIsEditing(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
      <div className="w-full max-w-xl bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-lg font-semibold text-white">
            {isMetastable ? 'Metastable State Info' : 'Macro-state Info'}
          </h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-5 max-h-[75vh] overflow-y-auto space-y-4 text-sm text-gray-200">
          <div className="flex items-center gap-2">
            <h4 className="text-base font-semibold text-white">Name:</h4>
            <RenameableText
              value={name}
              onSave={setName}
              isEditing={isEditing}
              setIsEditing={setIsEditing}
              handleSave={handleSave}
            />
          </div>

          <p>
            <span className="font-semibold">ID:</span>{' '}
            {isMetastable ? state.metastable_id : state.state_id}
          </p>
          {!isMetastable && state.pdb_file && (
            <p>
              <span className="font-semibold">PDB File:</span> {state.pdb_file}
            </p>
          )}
          {!isMetastable && state.trajectory_file && (
            <p>
              <span className="font-semibold">Trajectory File:</span> {state.trajectory_file}
            </p>
          )}
          {isMetastable && state.macro_state && (
            <p>
              <span className="font-semibold">Macro State:</span> {state.macro_state}
            </p>
          )}
          {isMetastable && (
            <p>
              <span className="font-semibold">Metastable Index:</span> {state.metastable_index}
            </p>
          )}
          <p>
            <span className="font-semibold">Number of Frames:</span> {state.n_frames ?? 'N/A'}
          </p>
          <p>
            <span className="font-semibold">Stride:</span> {state.stride ?? 'N/A'}
          </p>
          <p>
            <span className="font-semibold">Descriptors Ready:</span>{' '}
            {state.descriptor_file ? 'Yes' : 'No'}
          </p>
          {isMetastable && state.representative_pdb && (
            <p>
              <span className="font-semibold">Representative PDB:</span>{' '}
              <a href={state.representative_pdb} target="_blank" rel="noreferrer" className="text-cyan-400 hover:underline break-all">
                {state.representative_pdb}
              </a>
            </p>
          )}
          {/* Add more relevant info as needed */}
        </div>
      </div>
    </div>
  );
}

function RenameableText({ value, onSave, isEditing, setIsEditing, handleSave }) {
  const [currentValue, setCurrentValue] = useState(value);
  const inputRef = useRef(null);

  useEffect(() => {
    if (isEditing) {
      inputRef.current?.focus();
    }
  }, [isEditing]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSave();
    }
    if (e.key === 'Escape') {
      setIsEditing(false);
      setCurrentValue(value); // Reset to original value on escape
    }
  };

  return (
    <div className="flex items-center gap-2">
      {isEditing ? (
        <input
          ref={inputRef}
          type="text"
          value={currentValue}
          onChange={(e) => setCurrentValue(e.target.value)}
          onBlur={handleSave}
          onKeyDown={handleKeyDown}
          className="bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white text-sm"
        />
      ) : (
        <span className="text-gray-200">{value}</span>
      )}
      <button
        type="button"
        onClick={() => (isEditing ? handleSave() : setIsEditing(true))}
        className="text-gray-400 hover:text-cyan-300"
        aria-label={isEditing ? 'Save name' : 'Edit name'}
      >
        <Pencil className="h-4 w-4" />
      </button>
    </div>
  );
}

function parseMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const blocks = [];
  let title = 'Documentation';
  let paragraph = [];
  let list = null;
  let listType = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ type: 'p', text: paragraph.join(' ') });
      paragraph = [];
    }
  };

  const flushList = () => {
    if (list && list.length) {
      blocks.push({ type: listType, items: list });
    }
    list = null;
    listType = null;
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = headingMatch[1].length;
      const text = headingMatch[2].trim();
      if (level === 1 && title === 'Documentation') {
        title = text;
      } else {
        blocks.push({ type: `h${level}`, text });
      }
      continue;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.*)$/);
    const olMatch = trimmed.match(/^\d+[.)]\s+(.*)$/);
    if (ulMatch || olMatch) {
      flushParagraph();
      const nextType = olMatch ? 'ol' : 'ul';
      if (listType && listType !== nextType) {
        flushList();
      }
      listType = nextType;
      if (!list) list = [];
      list.push((ulMatch ? ulMatch[1] : olMatch[1]).trim());
      continue;
    }

    if (listType) {
      flushList();
    }
    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();

  return { title, blocks };
}

function renderDocBlock(block, idx, onNavigate) {
  if (block.type === 'p') {
    return (
      <p key={`p-${idx}`} className="text-gray-300 leading-relaxed">
        {renderInline(block.text, onNavigate)}
      </p>
    );
  }
  if (block.type === 'ul' || block.type === 'ol') {
    const Tag = block.type === 'ol' ? 'ol' : 'ul';
    return (
      <Tag key={`${block.type}-${idx}`} className={`pl-5 space-y-1 text-gray-300 ${block.type === 'ol' ? 'list-decimal' : 'list-disc'}`}>
        {block.items.map((item, itemIdx) => (
          <li key={`${block.type}-${idx}-${itemIdx}`}>{renderInline(item, onNavigate)}</li>
        ))}
      </Tag>
    );
  }
  if (block.type === 'h2' || block.type === 'h3') {
    return (
      <h4 key={`${block.type}-${idx}`} className="text-base font-semibold text-white">
        {block.text}
      </h4>
    );
  }
  return null;
}

function renderInline(text, onNavigate) {
  const parts = [];
  const regex = /\[([^\]]+)\]\(([^)]+)\)/g;
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', value: text.slice(lastIndex, match.index) });
    }
    parts.push({ type: 'link', label: match[1], href: match[2] });
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push({ type: 'text', value: text.slice(lastIndex) });
  }

  return parts.map((part, index) => {
    if (part.type === 'text') {
      const segments = part.value.split('`');
      return (
        <span key={`text-${index}`}>
          {segments.map((segment, segIdx) =>
            segIdx % 2 === 1 ? (
              <code key={`code-${index}-${segIdx}`} className="rounded bg-gray-800 px-1 text-xs text-gray-100">
                {segment}
              </code>
            ) : (
              <span key={`seg-${index}-${segIdx}`}>{segment}</span>
            )
          )}
        </span>
      );
    }
    if (part.href.startsWith('doc:')) {
      const target = part.href.replace('doc:', '').trim();
      return (
        <button
          key={`doc-${index}`}
          type="button"
          onClick={() => onNavigate(target)}
          className="text-cyan-300 hover:text-cyan-200 underline underline-offset-2"
        >
          {part.label}
        </button>
      );
    }
    return (
      <a
        key={`link-${index}`}
        href={part.href}
        target="_blank"
        rel="noreferrer"
        className="text-cyan-300 hover:text-cyan-200 underline underline-offset-2"
      >
        {part.label}
      </a>
    );
  });
}

function StatePairSelector({ options, value, onChange }) {
  if (options.length < 2) {
    return <p className="text-sm text-gray-400">Upload trajectories for at least two states to run analyses.</p>;
  }

  const toValue = (opt) => (opt ? `${opt.kind}:${opt.id}` : '');

  const handleChange = (key, raw) => {
    const next = options.find((o) => `${o.kind}:${o.id}` === raw) || null;
    onChange((prev) => ({ ...prev, [key]: next }));
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <p className="text-sm text-gray-200 mb-2">Select two states or metastable states to compare</p>
      <div className="grid md:grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-gray-400 mb-1">State A</label>
          <select
            value={toValue(value.a)}
            onChange={(e) => handleChange('a', e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          >
            <option value="">Choose</option>
            {options.map((opt) => (
              <option key={`${opt.kind}-${opt.id}`} value={`${opt.kind}:${opt.id}`}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">State B</label>
          <select
            value={toValue(value.b)}
            onChange={(e) => handleChange('b', e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          >
            <option value="">Choose</option>
            {options.map((opt) => (
              <option key={`${opt.kind}-${opt.id}`} value={`${opt.kind}:${opt.id}`}>
                {opt.label}
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
