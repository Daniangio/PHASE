import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { StateCard, MetastableCard } from '../components/system/StateCards';
import {
  ClusterBuildOverlay,
  ClusterDetailOverlay,
  DocOverlay,
  InfoOverlay,
} from '../components/system/SystemDetailOverlays';
import {
  AddStateForm,
  AnalysisResultsList,
  InfoTooltip,
  StatePairSelector,
} from '../components/system/SystemDetailWidgets';
import { getClusterDisplayName } from '../components/system/systemDetailUtils';
import {
  fetchSystem,
  downloadStructure,
  downloadStateDescriptors,
  uploadStateTrajectory,
  deleteStateTrajectory,
  addSystemState,
  deleteState,
  renameState, // Import the new renameState function
  fetchMetastableStates,
  recomputeMetastableStates,
  renameMetastableState,
  submitMetastableClusterJob,
  uploadMetastableClusterNp,
  renameMetastableCluster,
  confirmMacroStates,
  confirmMetastableStates,
  clearMetastableStates,
  downloadSavedCluster,
  downloadBackmappingCluster,
  submitBackmappingClusterJob,
  downloadPottsModel,
  uploadPottsModel,
  renamePottsModel,
  deletePottsModel,
  deleteSavedCluster,
} from '../api/projects';
import {
  fetchResults,
  submitPottsFitJob,
  submitStaticJob,
  submitSimulationJob,
  fetchJobStatus,
  uploadSimulationResults,
} from '../api/jobs';
import { confirmAndDeleteResult } from '../utils/results';
import StaticAnalysisForm from '../components/analysis/StaticAnalysisForm';
import SimulationAnalysisForm from '../components/analysis/SimulationAnalysisForm';
import SimulationUploadForm from '../components/analysis/SimulationUploadForm';
import { Eye, Info, Plus, SlidersHorizontal } from 'lucide-react';

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
  const [clusterPanelMode, setClusterPanelMode] = useState('generate');
  const [clusterName, setClusterName] = useState('');
  const [uploadClusterName, setUploadClusterName] = useState('');
  const [uploadClusterFile, setUploadClusterFile] = useState(null);
  const [uploadClusterStateIds, setUploadClusterStateIds] = useState([]);
  const [uploadClusterError, setUploadClusterError] = useState(null);
  const [uploadClusterLoading, setUploadClusterLoading] = useState(false);
  const [clusterDetailState, setClusterDetailState] = useState(null);
  const [clusterJobStatus, setClusterJobStatus] = useState({});
  const [backmappingDownloadProgress, setBackmappingDownloadProgress] = useState({});
  const [backmappingJobStatus, setBackmappingJobStatus] = useState({});
  const [maxClusterFrames, setMaxClusterFrames] = useState(0);
  const [densityZMode, setDensityZMode] = useState('auto');
  const [densityZValue, setDensityZValue] = useState(1.65);
  const [densityMaxk, setDensityMaxk] = useState(100);
  const [metaChoice, setMetaChoice] = useState(null);
  const [analysisFocus, setAnalysisFocus] = useState('');
  const [showMetastableInfo, setShowMetastableInfo] = useState(false);
  const [docId, setDocId] = useState('metastable_states');
  const [macroChoiceLoading, setMacroChoiceLoading] = useState(false);
  const [infoOverlayState, setInfoOverlayState] = useState(null); // New state for info overlay
  const [resultsList, setResultsList] = useState([]);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [resultsError, setResultsError] = useState(null);
  const [pottsFitClusterId, setPottsFitClusterId] = useState('');
  const [pottsFitMethod, setPottsFitMethod] = useState('pmi+plm');
  const [pottsFitContactMode, setPottsFitContactMode] = useState('CA');
  const [pottsFitContactCutoff, setPottsFitContactCutoff] = useState(10);
  const [pottsFitAdvanced, setPottsFitAdvanced] = useState(false);
  const [pottsFitMode, setPottsFitMode] = useState('run');
  const [pottsModelName, setPottsModelName] = useState('');
  const [pottsFitParams, setPottsFitParams] = useState({
    plm_epochs: '',
    plm_lr: '',
    plm_lr_min: '',
    plm_lr_schedule: 'cosine',
    plm_l2: '',
    plm_batch_size: '',
    plm_progress_every: '',
  });
  const [samplingMode, setSamplingMode] = useState('run');
  const [samplingUploadBusy, setSamplingUploadBusy] = useState(false);
  const [samplingUploadProgress, setSamplingUploadProgress] = useState(null);
  const [pottsFitError, setPottsFitError] = useState(null);
  const [pottsFitSubmitting, setPottsFitSubmitting] = useState(false);
  const [pottsUploadFile, setPottsUploadFile] = useState(null);
  const [pottsUploadName, setPottsUploadName] = useState('');
  const [pottsUploadError, setPottsUploadError] = useState(null);
  const [pottsUploadBusy, setPottsUploadBusy] = useState(false);
  const [pottsUploadProgress, setPottsUploadProgress] = useState(null);
  const [pottsRenameValues, setPottsRenameValues] = useState({});
  const [pottsRenameBusy, setPottsRenameBusy] = useState({});
  const [pottsRenameError, setPottsRenameError] = useState(null);
  const [pottsDeleteBusy, setPottsDeleteBusy] = useState({});
  const [pottsDeleteError, setPottsDeleteError] = useState(null);
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
  const clusterNameById = useMemo(() => {
    const mapping = new Map();
    clusterRuns.forEach((run) => {
      const name = run.name || run.path?.split('/').pop() || run.cluster_id;
      if (run.cluster_id) {
        mapping.set(run.cluster_id, name);
      }
    });
    return mapping;
  }, [clusterRuns]);
  const readyClusterRuns = useMemo(
    () => clusterRuns.filter((run) => run.path && run.status !== 'failed'),
    [clusterRuns]
  );
  const pottsModels = useMemo(
    () => readyClusterRuns.filter((run) => run.potts_model_path),
    [readyClusterRuns]
  );
  const selectedClusterName = useMemo(() => {
    if (!pottsFitClusterId) return null;
    return clusterNameById.get(pottsFitClusterId) || pottsFitClusterId;
  }, [pottsFitClusterId, clusterNameById]);
  const systemResults = useMemo(
    () => resultsList.filter((result) => result.system_id === systemId && result.project_id === projectId),
    [resultsList, projectId, systemId]
  );
  const staticResults = useMemo(
    () => systemResults.filter((result) => result.analysis_type === 'static'),
    [systemResults]
  );
  const simulationResults = useMemo(
    () => systemResults.filter((result) => result.analysis_type === 'simulation'),
    [systemResults]
  );
  const pottsFitResults = useMemo(
    () => systemResults.filter((result) => result.analysis_type === 'potts_fit'),
    [systemResults]
  );
  const pottsFitResultsWithClusters = useMemo(
    () =>
      pottsFitResults.map((result) => {
        if (result.cluster_name) {
          return { ...result, cluster_label: result.cluster_name };
        }
        const clusterId = result.cluster_id;
        const label = clusterId ? clusterNameById.get(clusterId) : null;
        return label ? { ...result, cluster_label: label } : result;
      }),
    [pottsFitResults, clusterNameById]
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
  const analysisNote =
    metastableLocked || effectiveChoice === 'macro'
      ? 'Static needs two descriptor-ready states; Potts uses saved cluster NPZ files.'
      : 'Static needs two descriptor-ready states. Potts requires metastable discovery and saved cluster NPZ files.';
  const showMacroChoicePanel =
    macroReady && !metastableLocked && effectiveChoice !== 'macro' && analysisMode !== 'metastable';
  const showMetastablePanel = macroReady && effectiveChoice !== 'macro' && !metastableLocked;
  const clusterSelectableStates = useMemo(() => {
    if (Array.isArray(system?.analysis_states) && system.analysis_states.length > 0) {
      return system.analysis_states;
    }
    const macroStates = states.map((state) => ({
      state_id: state.state_id,
      name: state.name,
      kind: 'macro',
      n_frames: state.n_frames ?? 0,
    }));
    const metaStates = (metastableStates || []).map((meta) => ({
      state_id: meta.metastable_id,
      name: meta.name || meta.default_name || meta.metastable_id,
      kind: 'metastable',
      metastable_id: meta.metastable_id,
      metastable_index: meta.metastable_index,
      macro_state: meta.macro_state,
      macro_state_id: meta.macro_state_id,
      n_frames: meta.n_frames ?? 0,
    }));
    const merged = [...macroStates, ...metaStates];
    const seen = new Set();
    return merged.filter((item) => {
      const key = item.state_id || item.metastable_id;
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [system, states, metastableStates]);
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
          const metaId = m.state_id || m.metastable_id || `${m.macro_state}-${m.metastable_index}`;
          return metaId === id;
        })
      )
    );
  }, [clusterSelectableStates]);

  useEffect(() => {
    if (!readyClusterRuns.length) {
      setPottsFitClusterId('');
      return;
    }
    const exists = readyClusterRuns.some((run) => run.cluster_id === pottsFitClusterId);
    if (!pottsFitClusterId || !exists) {
      setPottsFitClusterId(readyClusterRuns[readyClusterRuns.length - 1].cluster_id);
    }
  }, [readyClusterRuns, pottsFitClusterId]);

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

  const activeBackmappingJobs = useMemo(
    () =>
      Object.entries(backmappingJobStatus)
        .map(([clusterId, status]) => ({ clusterId, ...status }))
        .filter((job) => job.job_id && !['finished', 'failed'].includes(job.status)),
    [backmappingJobStatus]
  );

  useEffect(() => {
    if (!activeBackmappingJobs.length) return;
    let cancelled = false;
    const poll = async () => {
      const updates = {};
      for (const job of activeBackmappingJobs) {
        if (!job.job_id) continue;
        try {
          const status = await fetchJobStatus(job.job_id);
          updates[job.clusterId] = status;
        } catch (err) {
          updates[job.clusterId] = { status: 'failed', result: { error: err.message } };
        }
      }
      if (!cancelled && Object.keys(updates).length) {
        setBackmappingJobStatus((prev) => ({ ...prev, ...updates }));
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [activeBackmappingJobs]);

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

  const loadResults = async () => {
    setResultsLoading(true);
    setResultsError(null);
    try {
      const data = await fetchResults();
      setResultsList(Array.isArray(data) ? data : []);
    } catch (err) {
      setResultsError(err.message || 'Failed to load results.');
    } finally {
      setResultsLoading(false);
    }
  };

  useEffect(() => {
    loadResults();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, systemId]);

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
    loadResults();
    navigate(`/jobs/${response.job_id}`, { state: { analysis_uuid: response.analysis_uuid } });
  };

  const enqueueSimulationJob = async (params) => {
    setAnalysisError(null);
    const response = await submitSimulationJob({
      project_id: projectId,
      system_id: systemId,
      ...params,
    });
    loadResults();
    navigate(`/jobs/${response.job_id}`, { state: { analysis_uuid: response.analysis_uuid } });
  };

  const handleUploadSimulationResults = async ({ cluster_id, compare_cluster_ids, summaryFile, modelFile }) => {
    setSamplingUploadBusy(true);
    setSamplingUploadProgress(0);
    try {
      const response = await uploadSimulationResults(
        projectId,
        systemId,
        cluster_id,
        compare_cluster_ids,
        summaryFile,
        modelFile,
        { onUploadProgress: (percent) => setSamplingUploadProgress(percent) }
      );
      setSamplingUploadProgress(null);
      await refreshSystem();
      await loadResults();
      if (response?.job_id) {
        navigate(`/simulation/${response.job_id}`);
      }
    } catch (err) {
      throw err;
    } finally {
      setSamplingUploadBusy(false);
      setSamplingUploadProgress(null);
    }
  };

  const enqueuePottsFitJob = async () => {
    if (!pottsFitClusterId) {
      setPottsFitError('Select a cluster NPZ to fit.');
      return;
    }
    setPottsFitSubmitting(true);
    setPottsFitError(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: pottsFitClusterId,
        fit_method: pottsFitMethod,
        contact_atom_mode: pottsFitContactMode,
        contact_cutoff: pottsFitContactCutoff,
      };
      if (pottsModelName.trim()) payload.model_name = pottsModelName.trim();
      const numericKeys = new Set([
        'plm_epochs',
        'plm_lr',
        'plm_lr_min',
        'plm_l2',
        'plm_batch_size',
        'plm_progress_every',
      ]);
      Object.entries(pottsFitParams).forEach(([key, value]) => {
        if (value === '' || value === null || value === undefined) return;
        if (numericKeys.has(key)) {
          const num = Number(value);
          if (Number.isFinite(num)) payload[key] = num;
          return;
        }
        payload[key] = value;
      });
      const response = await submitPottsFitJob(payload);
      loadResults();
      navigate(`/jobs/${response.job_id}`, { state: { analysis_uuid: response.analysis_uuid } });
    } catch (err) {
      setPottsFitError(err.message || 'Failed to submit Potts fit job.');
    } finally {
      setPottsFitSubmitting(false);
    }
  };

  const handleUploadPottsModel = async () => {
    if (!pottsFitClusterId || !pottsUploadFile) {
      setPottsUploadError('Select a cluster and choose a model file to upload.');
      return;
    }
    setPottsUploadBusy(true);
    setPottsUploadError(null);
    setPottsUploadProgress(0);
    try {
      await uploadPottsModel(projectId, systemId, pottsFitClusterId, pottsUploadFile, {
        onUploadProgress: (percent) => setPottsUploadProgress(percent),
      });
      if (pottsUploadName.trim()) {
        await renamePottsModel(projectId, systemId, pottsFitClusterId, pottsUploadName.trim());
      }
      setPottsUploadFile(null);
      setPottsUploadName('');
      await refreshSystem();
    } catch (err) {
      setPottsUploadError(err.message || 'Failed to upload Potts model.');
    } finally {
      setPottsUploadBusy(false);
      setPottsUploadProgress(null);
    }
  };

  const formatPottsModelName = (run) => {
    const raw =
      run.potts_model_name ||
      (run.potts_model_path ? run.potts_model_path.split('/').pop() : '') ||
      'Potts model';
    return raw.replace(/\.npz$/i, '');
  };

  const handleRenamePottsModel = async (clusterId) => {
    const name = (pottsRenameValues[clusterId] || '').trim();
    if (!name) {
      setPottsRenameError('Provide a model name.');
      return;
    }
    setPottsRenameError(null);
    setPottsRenameBusy((prev) => ({ ...prev, [clusterId]: true }));
    try {
      await renamePottsModel(projectId, systemId, clusterId, name);
      await refreshSystem();
    } catch (err) {
      setPottsRenameError(err.message || 'Failed to rename Potts model.');
    } finally {
      setPottsRenameBusy((prev) => ({ ...prev, [clusterId]: false }));
    }
  };

  const handleDeletePottsModel = async (clusterId) => {
    if (!window.confirm('Delete this fitted Potts model?')) {
      return;
    }
    setPottsDeleteError(null);
    setPottsDeleteBusy((prev) => ({ ...prev, [clusterId]: true }));
    try {
      await deletePottsModel(projectId, systemId, clusterId);
      await refreshSystem();
    } catch (err) {
      setPottsDeleteError(err.message || 'Failed to delete Potts model.');
    } finally {
      setPottsDeleteBusy((prev) => ({ ...prev, [clusterId]: false }));
    }
  };

  const handleUploadTrajectory = async (stateId, file, sliceSpec, residueSelection) => {
    if (!file) return;
    setUploadingState(stateId);
    setActionError(null);
    try {
      const payload = new FormData();
      payload.append('trajectory', file);
      if (sliceSpec && sliceSpec.trim()) {
        payload.append('slice_spec', sliceSpec.trim());
      } else {
        payload.append('stride', 1);
      }
      if (residueSelection && residueSelection.trim()) {
        payload.append('residue_selection', residueSelection.trim());
      }
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

  const handleDownloadMacroStateNpz = async (stateId, name) => {
    setDownloadError(null);
    try {
      const blob = await downloadStateDescriptors(projectId, systemId, stateId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${system?.name || systemId}-${name || stateId}_descriptors.npz`;
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
        'Select at least one state to cluster.'
      );
      return;
    }
    if (!clusterName.trim()) {
      setClusterError('Provide a cluster name.');
      return;
    }
    setClusterError(null);
    setClusterLoading(true);
    try {
      const trimmedName = clusterName.trim();
      const densityZ = densityZMode === 'manual' ? densityZValue : 'auto';
      await submitMetastableClusterJob(projectId, systemId, selectedMetastableIds, {
        cluster_name: trimmedName || undefined,
        max_cluster_frames: maxClusterFrames > 0 ? maxClusterFrames : undefined,
        density_maxk: densityMaxk,
        density_z: densityZ,
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

  const uploadStateOptions = useMemo(() => {
    const options = clusterSelectableStates.map((state) => {
      const id = state.state_id || state.metastable_id;
      if (!id) return null;
      const kind = state.kind || (state.metastable_id ? 'metastable' : 'macro');
      const label = state.name || state.default_name || state.macro_state || id;
      return { id, label: `[${kind === 'metastable' ? 'Meta' : 'Macro'}] ${label}` };
    });
    return options.filter(Boolean);
  }, [clusterSelectableStates]);

  const toggleUploadState = (stateId) => {
    setUploadClusterError(null);
    setUploadClusterStateIds((prev) =>
      prev.includes(stateId) ? prev.filter((id) => id !== stateId) : [...prev, stateId]
    );
  };

  const handleUploadClusterNp = async () => {
    setUploadClusterError(null);
    if (!uploadClusterFile) {
      setUploadClusterError('Select a cluster NPZ file to upload.');
      return;
    }
    if (!uploadClusterName.trim()) {
      setUploadClusterError('Provide a cluster name.');
      return;
    }
    if (!uploadClusterStateIds.length) {
      setUploadClusterError('Select the states used to build the local cluster.');
      return;
    }
    setUploadClusterLoading(true);
    try {
      const payload = new FormData();
      payload.append('cluster_npz', uploadClusterFile);
      payload.append('state_ids', uploadClusterStateIds.join(','));
      if (uploadClusterName.trim()) payload.append('name', uploadClusterName.trim());
      await uploadMetastableClusterNp(projectId, systemId, payload);
      setClusterPanelOpen(false);
      setClusterPanelMode('generate');
      setUploadClusterName('');
      setUploadClusterFile(null);
      setUploadClusterStateIds([]);
      await refreshSystem();
    } catch (err) {
      setUploadClusterError(err.message);
    } finally {
      setUploadClusterLoading(false);
    }
  };

  if (isLoading) return <Loader message="Loading system..." />;
  if (pageError) return <ErrorMessage message={pageError} />;
  if (!system) return null;

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

  const handleDownloadBackmappingCluster = async (clusterId, filename) => {
    setClusterError(null);
    try {
      setBackmappingDownloadProgress((prev) => ({ ...prev, [clusterId]: 0 }));
      const blob = await downloadBackmappingCluster(projectId, systemId, clusterId, {
        onProgress: (percent) =>
          setBackmappingDownloadProgress((prev) => ({ ...prev, [clusterId]: percent })),
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || `backmapping_${clusterId}.npz`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setClusterError(err.message);
    } finally {
      setBackmappingDownloadProgress((prev) => ({ ...prev, [clusterId]: null }));
    }
  };

  const handleBackmappingAction = async (clusterId, filename) => {
    const job = backmappingJobStatus[clusterId];
    if (job?.status === 'finished') {
      await handleDownloadBackmappingCluster(clusterId, filename);
      return;
    }
    if (job?.status && job.status !== 'failed') return;
    setClusterError(null);
    try {
      const response = await submitBackmappingClusterJob(projectId, systemId, clusterId);
      if (response?.job_id) {
        setBackmappingJobStatus((prev) => ({
          ...prev,
          [clusterId]: { status: 'queued', job_id: response.job_id, meta: { progress: 0 } },
        }));
      }
    } catch (err) {
      setClusterError(err.message);
    }
  };

  const handleDownloadPottsModel = async (clusterId, filename) => {
    setPottsRenameError(null);
    try {
      const blob = await downloadPottsModel(projectId, systemId, clusterId);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename || `potts_model_${clusterId}.npz`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setPottsRenameError(err.message || 'Failed to download Potts model.');
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

  const handleDeleteResult = async (result) => {
    if (!result?.job_id) return;
    await confirmAndDeleteResult(result.job_id, {
      onSuccess: loadResults,
      onError: (err) => setResultsError(err.message || 'Failed to delete result.'),
    });
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
                    Macro and metastable states can both be selected when clustering.
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
                                {Array.isArray(run.state_ids || run.metastable_ids)
                                  ? (run.state_ids || run.metastable_ids).length
                                  : '—'}{' '}
                                |{' '}
                                Max frames: {run.max_cluster_frames ?? 'all'}
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
                              {Array.isArray(run.state_ids || run.metastable_ids)
                                ? (run.state_ids || run.metastable_ids).length
                                : '—'}{' '}
                              |{' '}
                              Max frames: {run.max_cluster_frames ?? 'all'}
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
                <div className="border-t border-gray-800 pt-3">
                  <h4 className="text-xs font-semibold text-gray-300">Potts models</h4>
                  {pottsModels.length === 0 && <p className="text-xs text-gray-500 mt-1">No Potts models yet.</p>}
                  {pottsModels.length > 0 && (
                    <div className="space-y-2 mt-2">
                      {pottsModels.map((run) => (
                        <div
                          key={run.cluster_id}
                          className="rounded-md border border-gray-800 bg-gray-900/50 px-3 py-2 text-xs"
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-gray-200">{formatPottsModelName(run)}</span>
                            <span className="text-[10px] text-gray-500">
                              {run.cluster_id?.slice(0, 8)}
                            </span>
                          </div>
                          <p className="text-[11px] text-gray-500 mt-1">
                            Cluster: {run.name || run.cluster_id}
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
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
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-white">Run Analysis</h2>
              <p className="text-xs text-gray-400">{analysisNote}</p>
            </div>
            {analysisFocus && (
              <button
                type="button"
                onClick={() => setAnalysisFocus('')}
                className="text-xs px-3 py-1 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
              >
                Back to analyses
              </button>
            )}
          </div>
          {analysisError && <ErrorMessage message={analysisError} />}
          {!analysisFocus && (
            <div className="grid lg:grid-cols-2 gap-6">
              <button
                type="button"
                onClick={() => setAnalysisFocus('static')}
                className="bg-gray-900 border border-gray-700 rounded-lg p-4 text-left hover:border-cyan-500/60 transition-colors"
              >
                <h3 className="text-md font-semibold text-white">Static Reporters</h3>
                <p className="text-xs text-gray-500 mt-1">Compare two descriptor-ready states.</p>
                <p className="text-xs text-gray-400 mt-3">
                  Results: {staticResults.length || 0}
                </p>
              </button>
              <button
                type="button"
                onClick={() => setAnalysisFocus('potts')}
                className="bg-gray-900 border border-gray-700 rounded-lg p-4 text-left hover:border-cyan-500/60 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <h3 className="text-md font-semibold text-white">Potts Modeling</h3>
                  <InfoTooltip
                    ariaLabel="Potts analysis documentation"
                    text="Open documentation for the Potts model, sampling, and diagnostics."
                    onClick={() => openDoc('potts_overview')}
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">Fit a model or reuse an uploaded one.</p>
                <p className="text-xs text-gray-400 mt-3">
                  Sampling runs: {simulationResults.length || 0}
                </p>
              </button>
            </div>
          )}
          {analysisFocus === 'static' && (
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
              <div className="border-t border-gray-700 pt-4 space-y-2">
                <h4 className="text-sm font-semibold text-white">Previous static results</h4>
                {resultsLoading && <p className="text-xs text-gray-500">Loading results…</p>}
                {resultsError && <ErrorMessage message={resultsError} />}
                {!resultsLoading && !resultsError && (
                  <AnalysisResultsList
                    results={staticResults}
                    emptyLabel="No static results for this system yet."
                    onOpen={(result) => navigate(`/results/${result.job_id}`)}
                    onDelete={handleDeleteResult}
                  />
                )}
              </div>
            </div>
          )}
          {analysisFocus === 'potts' && (
            <div className="space-y-4">
              <div className="grid lg:grid-cols-2 gap-6">
                <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="text-md font-semibold text-white">Potts Model Fitting</h3>
                      <p className="text-xs text-gray-500 mt-1">
                        Fit on the webserver or download the cluster NPZ for offline fitting.
                      </p>
                    </div>
                    <InfoTooltip
                      ariaLabel="Potts model fitting documentation"
                      text="Fit a Potts model once and reuse it for sampling."
                      onClick={() => openDoc('potts_model')}
                    />
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <button
                      type="button"
                      onClick={() => setPottsFitMode('run')}
                      className={`px-3 py-1 rounded-full border ${
                        pottsFitMode === 'run'
                          ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10'
                          : 'border-gray-700 text-gray-400 hover:border-gray-500'
                      }`}
                    >
                      Run on server
                    </button>
                    <button
                      type="button"
                      onClick={() => setPottsFitMode('upload')}
                      className={`px-3 py-1 rounded-full border ${
                        pottsFitMode === 'upload'
                          ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10'
                          : 'border-gray-700 text-gray-400 hover:border-gray-500'
                      }`}
                    >
                      Upload results
                    </button>
                  </div>
                  {pottsFitMode === 'run' && (
                    <>
                      <div>
                        <label className="block text-sm text-gray-300 mb-1">Cluster NPZ</label>
                        <select
                          value={pottsFitClusterId}
                          onChange={(event) => setPottsFitClusterId(event.target.value)}
                          disabled={!readyClusterRuns.length}
                          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
                        >
                          {!readyClusterRuns.length && <option value="">No saved clusters</option>}
                          {readyClusterRuns.map((run) => {
                            const name = run.name || run.path?.split('/').pop() || run.cluster_id;
                            return (
                              <option key={run.cluster_id} value={run.cluster_id}>
                                {name}
                              </option>
                            );
                          })}
                        </select>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-300 mb-1">Potts model name</label>
                        <input
                          type="text"
                          value={pottsModelName}
                          onChange={(event) => setPottsModelName(event.target.value)}
                          placeholder="e.g. Active+Inactive Potts"
                          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                        />
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            if (!pottsFitClusterId) return;
                            const entry = readyClusterRuns.find((run) => run.cluster_id === pottsFitClusterId);
                            const filename = entry?.path?.split('/').pop();
                            handleDownloadSavedCluster(pottsFitClusterId, filename);
                          }}
                          disabled={!pottsFitClusterId}
                          className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50"
                        >
                          Download cluster NPZ
                        </button>
                      </div>
                      <div>
                        <label className="block text-sm text-gray-300 mb-1">Fit method</label>
                        <select
                          value={pottsFitMethod}
                          onChange={(event) => setPottsFitMethod(event.target.value)}
                          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                        >
                          <option value="pmi+plm">PMI + PLM</option>
                          <option value="plm">PLM only</option>
                          <option value="pmi">PMI only</option>
                        </select>
                      </div>
                      <div className="grid md:grid-cols-2 gap-3 text-sm">
                        <label className="space-y-1">
                          <span className="text-xs text-gray-400">Contact mode</span>
                          <select
                            value={pottsFitContactMode}
                            onChange={(event) => setPottsFitContactMode(event.target.value)}
                            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                          >
                            <option value="CA">CA</option>
                            <option value="CM">Residue CM</option>
                          </select>
                        </label>
                        <label className="space-y-1">
                          <span className="text-xs text-gray-400">Contact cutoff (A)</span>
                          <input
                            type="number"
                            min={1}
                            step="0.5"
                            value={pottsFitContactCutoff}
                            onChange={(event) =>
                              setPottsFitContactCutoff(Math.max(0.1, Number(event.target.value) || 0))
                            }
                            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                          />
                        </label>
                      </div>
                      <button
                        type="button"
                        onClick={() => setPottsFitAdvanced((prev) => !prev)}
                        className="flex items-center gap-2 text-xs text-cyan-300 hover:text-cyan-200"
                      >
                        <SlidersHorizontal className="h-4 w-4" />
                        {pottsFitAdvanced ? 'Hide' : 'Show'} fit hyperparams
                      </button>
                      {pottsFitAdvanced && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                          {[
                            { key: 'plm_epochs', label: 'PLM epochs', placeholder: '200' },
                            { key: 'plm_lr', label: 'PLM lr', placeholder: '1e-2' },
                            { key: 'plm_lr_min', label: 'PLM lr min', placeholder: '1e-3' },
                            { key: 'plm_l2', label: 'PLM L2', placeholder: '1e-5' },
                            { key: 'plm_batch_size', label: 'Batch size', placeholder: '512' },
                            { key: 'plm_progress_every', label: 'Progress every', placeholder: '10' },
                          ].map((field) => (
                            <label key={field.key} className="space-y-1">
                              <span className="text-xs text-gray-400">{field.label}</span>
                              <input
                                type="text"
                                placeholder={field.placeholder}
                                value={pottsFitParams[field.key]}
                                onChange={(event) =>
                                  setPottsFitParams((prev) => ({
                                    ...prev,
                                    [field.key]: event.target.value,
                                  }))
                                }
                                className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                              />
                            </label>
                          ))}
                          <label className="space-y-1">
                            <span className="text-xs text-gray-400">LR schedule</span>
                            <select
                              value={pottsFitParams.plm_lr_schedule}
                              onChange={(event) =>
                                setPottsFitParams((prev) => ({
                                  ...prev,
                                  plm_lr_schedule: event.target.value,
                                }))
                              }
                              className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                            >
                              <option value="cosine">Cosine</option>
                              <option value="none">None</option>
                            </select>
                          </label>
                        </div>
                      )}
                      {pottsFitError && <ErrorMessage message={pottsFitError} />}
                      <button
                        type="button"
                        onClick={enqueuePottsFitJob}
                        disabled={pottsFitSubmitting || !pottsFitClusterId}
                        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
                      >
                        {pottsFitSubmitting ? 'Submitting…' : 'Run Potts fit'}
                      </button>
                    </>
                  )}
                  {pottsFitMode === 'upload' && (
                    <div className="border border-gray-700 rounded-md p-3 space-y-2">
                      <p className="text-xs text-gray-400">
                        Upload a fitted model for the selected cluster.
                      </p>
                      {selectedClusterName && (
                        <p className="text-[11px] text-gray-500">Selected cluster: {selectedClusterName}</p>
                      )}
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Model name</label>
                        <input
                          type="text"
                          value={pottsUploadName}
                          onChange={(event) => setPottsUploadName(event.target.value)}
                          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                          placeholder="e.g. Active+Inactive Potts"
                        />
                      </div>
                      <input
                        type="file"
                        accept=".npz"
                        onChange={(event) => setPottsUploadFile(event.target.files?.[0] || null)}
                        className="w-full text-xs text-gray-200"
                      />
                      {pottsUploadProgress !== null && (
                        <div>
                          <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                            <span>Uploading model</span>
                            <span>{pottsUploadProgress}%</span>
                          </div>
                          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-cyan-500 transition-all duration-200"
                              style={{ width: `${pottsUploadProgress}%` }}
                            />
                          </div>
                        </div>
                      )}
                      {pottsUploadError && <ErrorMessage message={pottsUploadError} />}
                      <button
                        type="button"
                        onClick={handleUploadPottsModel}
                        disabled={!pottsUploadFile || !pottsFitClusterId || pottsUploadBusy}
                        className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50"
                      >
                        {pottsUploadBusy ? 'Uploading…' : 'Upload model'}
                      </button>
                    </div>
                  )}
                  {pottsModels.length > 0 && (
                    <div className="border border-gray-700 rounded-md p-3 space-y-3">
                      <p className="text-xs text-gray-400">Uploaded models</p>
                      {pottsModels.map((run) => {
                        const displayName = formatPottsModelName(run);
                        const value = pottsRenameValues[run.cluster_id] ?? displayName;
                        const isBusy = pottsRenameBusy[run.cluster_id];
                        const isDeleting = pottsDeleteBusy[run.cluster_id];
                        return (
                          <div key={run.cluster_id} className="flex items-center gap-2">
                            <div className="flex-1">
                              <p className="text-[11px] text-gray-500">
                                {run.name || run.cluster_id}
                              </p>
                              <input
                                type="text"
                                value={value}
                                onChange={(event) =>
                                  setPottsRenameValues((prev) => ({
                                    ...prev,
                                    [run.cluster_id]: event.target.value,
                                  }))
                                }
                                className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-xs text-white focus:ring-cyan-500"
                              />
                            </div>
                            <button
                              type="button"
                              onClick={() =>
                                handleDownloadPottsModel(
                                  run.cluster_id,
                                  run.potts_model_path?.split('/').pop()
                                )
                              }
                              disabled={isBusy || isDeleting}
                              className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50"
                            >
                              Download
                            </button>
                            <button
                              type="button"
                              onClick={() => handleRenamePottsModel(run.cluster_id)}
                              disabled={isBusy || isDeleting}
                              className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50"
                            >
                              {isBusy ? 'Saving…' : 'Rename'}
                            </button>
                            <button
                              type="button"
                              onClick={() => handleDeletePottsModel(run.cluster_id)}
                              disabled={isDeleting || isBusy}
                              className="text-xs px-3 py-2 rounded-md border border-red-500/40 text-red-200 hover:bg-red-500/10 disabled:opacity-50"
                            >
                              {isDeleting ? 'Deleting…' : 'Delete'}
                            </button>
                          </div>
                        );
                      })}
                      {pottsRenameError && <ErrorMessage message={pottsRenameError} />}
                      {pottsDeleteError && <ErrorMessage message={pottsDeleteError} />}
                    </div>
                  )}
                  {pottsFitResults.length > 0 && (
                    <div className="border-t border-gray-800 pt-3 space-y-2">
                      <h4 className="text-xs font-semibold text-gray-300">Recent fit jobs</h4>
                      <AnalysisResultsList
                        results={pottsFitResultsWithClusters}
                        emptyLabel="No fit results yet."
                        onOpen={(result) => navigate(`/results/${result.job_id}`)}
                        onDelete={handleDeleteResult}
                      />
                    </div>
                  )}
                </div>
                <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2">
                      <h3 className="text-md font-semibold text-white">Potts Sampling</h3>
                      <InfoTooltip
                        ariaLabel="Potts analysis documentation"
                        text="Run sampling with a selected fitted model."
                        onClick={() => openDoc('potts_overview')}
                      />
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <button
                        type="button"
                        onClick={() => setSamplingMode('run')}
                        className={`px-2 py-1 rounded-md border ${
                          samplingMode === 'run'
                            ? 'border-cyan-400 text-cyan-200'
                            : 'border-gray-700 text-gray-400 hover:text-gray-200'
                        }`}
                      >
                        Run on server
                      </button>
                      <button
                        type="button"
                        onClick={() => setSamplingMode('upload')}
                        className={`px-2 py-1 rounded-md border ${
                          samplingMode === 'upload'
                            ? 'border-cyan-400 text-cyan-200'
                            : 'border-gray-700 text-gray-400 hover:text-gray-200'
                        }`}
                      >
                        Upload results
                      </button>
                    </div>
                  </div>
                  {samplingMode === 'run' ? (
                    <SimulationAnalysisForm clusterRuns={readyClusterRuns} onSubmit={enqueueSimulationJob} />
                  ) : (
                    <div className="space-y-3">
                      {samplingUploadProgress !== null && (
                        <p className="text-xs text-gray-500">Uploading... {samplingUploadProgress}%</p>
                      )}
                      <SimulationUploadForm
                        clusterRuns={readyClusterRuns}
                        onSubmit={handleUploadSimulationResults}
                        isBusy={samplingUploadBusy}
                      />
                    </div>
                  )}
                </div>
              </div>
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-2">
                <h4 className="text-sm font-semibold text-white">Sampling results</h4>
                {resultsLoading && <p className="text-xs text-gray-500">Loading results…</p>}
                {resultsError && <ErrorMessage message={resultsError} />}
                {!resultsLoading && !resultsError && (
                  <AnalysisResultsList
                    results={simulationResults}
                    emptyLabel="No Potts sampling results for this system yet."
                    onOpen={(result) => navigate(`/results/${result.job_id}`)}
                    onOpenSimulation={(result) => navigate(`/simulation/${result.job_id}`)}
                    onDelete={handleDeleteResult}
                  />
                )}
              </div>
            </div>
          )}
        </section>
      )}
      {clusterPanelOpen && (
        <ClusterBuildOverlay
          metastableStates={clusterSelectableStates}
          selectedMetastableIds={selectedMetastableIds}
          onToggleMetastable={toggleMetastableSelection}
          clusterName={clusterName}
          setClusterName={setClusterName}
          clusterMode={clusterPanelMode}
          setClusterMode={setClusterPanelMode}
          uploadClusterName={uploadClusterName}
          setUploadClusterName={setUploadClusterName}
          uploadClusterFile={uploadClusterFile}
          setUploadClusterFile={setUploadClusterFile}
          uploadClusterStateIds={uploadClusterStateIds}
          uploadStateOptions={uploadStateOptions}
          onToggleUploadState={toggleUploadState}
          uploadClusterError={uploadClusterError}
          uploadClusterLoading={uploadClusterLoading}
          densityZMode={densityZMode}
          setDensityZMode={setDensityZMode}
          densityZValue={densityZValue}
          setDensityZValue={setDensityZValue}
          densityMaxk={densityMaxk}
          setDensityMaxk={setDensityMaxk}
          maxClusterFrames={maxClusterFrames}
          setMaxClusterFrames={setMaxClusterFrames}
          clusterError={clusterError}
          clusterLoading={clusterLoading}
          onClose={() => setClusterPanelOpen(false)}
          onSubmit={handleDownloadClusters}
          onUpload={handleUploadClusterNp}
        />
      )}
      {clusterDetailState && (
        <ClusterDetailOverlay
          cluster={clusterDetailState}
          analysisMode={analysisMode}
          onClose={() => setClusterDetailState(null)}
          onRename={handleRenameCluster}
          onDownload={handleDownloadSavedCluster}
          onDownloadBackmapping={handleBackmappingAction}
          backmappingProgress={backmappingDownloadProgress[clusterDetailState.cluster_id] ?? null}
          backmappingJob={backmappingJobStatus[clusterDetailState.cluster_id] ?? null}
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
          onDownloadMacroNpz={handleDownloadMacroStateNpz}
        />
      )}
        </div>
      </div>
    </div>
  );
}
