import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import SystemDetailSidebar from '../components/system/SystemDetailSidebar';
import SystemDetailMacroPanel from '../components/system/SystemDetailMacroPanel';
import SystemDetailMetastablePanel from '../components/system/SystemDetailMetastablePanel';
import SystemDetailPottsSection from '../components/system/SystemDetailPottsSection';
import { ClusterBuildOverlay, DocOverlay, InfoOverlay } from '../components/system/SystemDetailOverlays';
import {
  AnalysisResultsList,
  InfoTooltip,
  StatePairSelector,
} from '../components/system/SystemDetailWidgets';
import { formatPottsModelName } from './SystemDetailPage.utils';
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
  confirmMacroStates,
  confirmMetastableStates,
  clearMetastableStates,
  downloadSavedCluster,
  downloadBackmappingCluster,
  submitBackmappingClusterJob,
  uploadBackmappingTrajectories,
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
  const selectedCluster = useMemo(
    () => clusterRuns.find((run) => run.cluster_id === pottsFitClusterId) || null,
    [clusterRuns, pottsFitClusterId]
  );
  const pottsModels = useMemo(() => {
    if (!selectedCluster) return [];
    const models = selectedCluster.potts_models || [];
    return models.map((model) => ({ ...model, cluster_id: selectedCluster.cluster_id }));
  }, [selectedCluster]);
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
  const metastableById = useMemo(() => {
    const mapping = new Map();
    (system?.metastable_states || []).forEach((m) => {
      if (m.metastable_id) {
        mapping.set(m.metastable_id, m);
      }
    });
    return mapping;
  }, [system]);
  const sampleEntries = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);
  const mdSamples = useMemo(
    () => sampleEntries.filter((s) => s.type === 'md_eval'),
    [sampleEntries]
  );
  const clusterSimulationResults = useMemo(
    () => simulationResults.filter((result) => result.cluster_id === pottsFitClusterId),
    [simulationResults, pottsFitClusterId]
  );
  const gibbsSamples = useMemo(
    () =>
      clusterSimulationResults.filter((result) => {
        const params = result.params || {};
        return (
          params.gibbs_samples ||
          params.rex_samples ||
          params.gibbs_method ||
          params.rex_betas ||
          params.rex_beta_min ||
          params.rex_beta_max
        );
      }),
    [clusterSimulationResults]
  );
  const saSamples = useMemo(
    () =>
      clusterSimulationResults.filter((result) => {
        const params = result.params || {};
        return (
          params.sa_reads ||
          params.sa_sweeps ||
          params.sa_beta_hot ||
          params.sa_beta_cold ||
          (Array.isArray(params.sa_beta_schedules) && params.sa_beta_schedules.length > 0)
        );
      }),
    [clusterSimulationResults]
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
      const response = await uploadPottsModel(projectId, systemId, pottsFitClusterId, pottsUploadFile, {
        onUploadProgress: (percent) => setPottsUploadProgress(percent),
      });
      if (pottsUploadName.trim() && response?.model_id) {
        await renamePottsModel(projectId, systemId, pottsFitClusterId, response.model_id, pottsUploadName.trim());
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

  const handleRenamePottsModel = async (clusterId, modelId) => {
    const name = (pottsRenameValues[modelId] || '').trim();
    if (!name) {
      setPottsRenameError('Provide a model name.');
      return;
    }
    setPottsRenameError(null);
    setPottsRenameBusy((prev) => ({ ...prev, [modelId]: true }));
    try {
      await renamePottsModel(projectId, systemId, clusterId, modelId, name);
      await refreshSystem();
    } catch (err) {
      setPottsRenameError(err.message || 'Failed to rename Potts model.');
    } finally {
      setPottsRenameBusy((prev) => ({ ...prev, [modelId]: false }));
    }
  };

  const handleDeletePottsModel = async (clusterId, modelId) => {
    if (!window.confirm('Delete this fitted Potts model?')) {
      return;
    }
    setPottsDeleteError(null);
    setPottsDeleteBusy((prev) => ({ ...prev, [modelId]: true }));
    try {
      await deletePottsModel(projectId, systemId, clusterId, modelId);
      await refreshSystem();
    } catch (err) {
      setPottsDeleteError(err.message || 'Failed to delete Potts model.');
    } finally {
      setPottsDeleteBusy((prev) => ({ ...prev, [modelId]: false }));
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

  const openDescriptorExplorer = useCallback(
    ({ clusterId, stateId, metastableId }) => {
      const params = new URLSearchParams();
      if (clusterId) params.set('cluster_id', clusterId);
      if (stateId) params.append('state_id', stateId);
      if (metastableId) params.set('metastable_ids', metastableId);
      if (clusterId && clusterId !== pottsFitClusterId) {
        setPottsFitClusterId(clusterId);
      }
      navigate(`/projects/${projectId}/systems/${systemId}/descriptors/visualize?${params.toString()}`);
    },
    [navigate, projectId, systemId, pottsFitClusterId, setPottsFitClusterId]
  );

  if (isLoading) return <Loader message="Loading system..." />;
  if (pageError) return <ErrorMessage message={pageError} />;
  if (!system) return null;

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

  const handleUploadBackmappingTrajectories = async (clusterId, filesByState) => {
    const stateIds = Object.keys(filesByState || {}).filter((key) => filesByState[key]);
    if (!stateIds.length) {
      throw new Error('Select at least one trajectory to upload.');
    }
    const payload = new FormData();
    payload.append('state_ids', stateIds.join(','));
    stateIds.forEach((stateId) => payload.append('trajectories', filesByState[stateId]));
    const blob = await uploadBackmappingTrajectories(projectId, systemId, clusterId, payload);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `backmapping_${clusterId}.npz`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  const handleDownloadPottsModel = async (clusterId, modelId, filename) => {
    setPottsRenameError(null);
    try {
      const blob = await downloadPottsModel(projectId, systemId, clusterId, modelId);
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
          <SystemDetailSidebar
            states={states}
            metastableStates={metastableStates}
            metastableLocked={metastableLocked}
            clustersUnlocked={clustersUnlocked}
            clusterError={clusterError}
            clusterLoading={clusterLoading}
            clusterRuns={clusterRuns}
            analysisMode={analysisMode}
            clusterJobStatus={clusterJobStatus}
            setInfoOverlayState={setInfoOverlayState}
            setPottsFitClusterId={setPottsFitClusterId}
            setAnalysisFocus={setAnalysisFocus}
            setClusterPanelOpen={setClusterPanelOpen}
            setClusterError={setClusterError}
            handleDeleteSavedCluster={handleDeleteSavedCluster}
            selectedClusterId={pottsFitClusterId}
            openDescriptorExplorer={openDescriptorExplorer}
            openDoc={openDoc}
            navigate={navigate}
            projectId={projectId}
            systemId={systemId}
          />
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
            <SystemDetailMacroPanel
              states={states}
              systemStatus={system.status}
              descriptorsReady={descriptorsReady}
              handleConfirmMacro={handleConfirmMacro}
              downloadError={downloadError}
              actionError={actionError}
              actionMessage={actionMessage}
              handleDownloadStructure={handleDownloadStructure}
              handleUploadTrajectory={handleUploadTrajectory}
              handleDeleteTrajectory={handleDeleteTrajectory}
              handleDeleteState={handleDeleteState}
              uploadingState={uploadingState}
              uploadProgress={uploadProgress}
              processingState={processingState}
              handleAddState={handleAddState}
              addingState={addingState}
            />
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
            <SystemDetailMetastablePanel
              metastableLocked={metastableLocked}
              metaLoading={metaLoading}
              metaParamsOpen={metaParamsOpen}
              setMetaParamsOpen={setMetaParamsOpen}
              metaParams={metaParams}
              setMetaParams={setMetaParams}
              metaError={metaError}
              metaActionError={metaActionError}
              metastableStates={metastableStates}
              handleRunMetastable={handleRunMetastable}
              handleConfirmMetastable={handleConfirmMetastable}
              handleRenameMetastable={handleRenameMetastable}
              openDoc={openDoc}
              navigate={navigate}
              projectId={projectId}
              systemId={systemId}
            />
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
            <SystemDetailPottsSection
              mdSamples={mdSamples}
              gibbsSamples={gibbsSamples}
              saSamples={saSamples}
              metastableById={metastableById}
              states={states}
              openDescriptorExplorer={openDescriptorExplorer}
              pottsFitClusterId={pottsFitClusterId}
              pottsFitMode={pottsFitMode}
              setPottsFitMode={setPottsFitMode}
              setPottsFitClusterId={setPottsFitClusterId}
              readyClusterRuns={readyClusterRuns}
              pottsModelName={pottsModelName}
              setPottsModelName={setPottsModelName}
              handleDownloadSavedCluster={handleDownloadSavedCluster}
              handleBackmappingAction={handleBackmappingAction}
              backmappingJobStatus={backmappingJobStatus}
              backmappingProgressById={backmappingDownloadProgress}
              handleDeleteSavedCluster={handleDeleteSavedCluster}
              handleUploadBackmappingTrajectories={handleUploadBackmappingTrajectories}
              pottsFitMethod={pottsFitMethod}
              setPottsFitMethod={setPottsFitMethod}
              pottsFitContactMode={pottsFitContactMode}
              setPottsFitContactMode={setPottsFitContactMode}
              pottsFitContactCutoff={pottsFitContactCutoff}
              setPottsFitContactCutoff={setPottsFitContactCutoff}
              pottsFitAdvanced={pottsFitAdvanced}
              setPottsFitAdvanced={setPottsFitAdvanced}
              pottsFitParams={pottsFitParams}
              setPottsFitParams={setPottsFitParams}
              pottsFitError={pottsFitError}
              enqueuePottsFitJob={enqueuePottsFitJob}
              pottsFitSubmitting={pottsFitSubmitting}
              pottsFitResults={pottsFitResults}
              pottsFitResultsWithClusters={pottsFitResultsWithClusters}
              handleDeleteResult={handleDeleteResult}
              openDoc={openDoc}
              selectedCluster={selectedCluster}
              selectedClusterName={selectedClusterName}
              pottsUploadName={pottsUploadName}
              setPottsUploadName={setPottsUploadName}
              setPottsUploadFile={setPottsUploadFile}
              pottsUploadFile={pottsUploadFile}
              pottsUploadProgress={pottsUploadProgress}
              pottsUploadError={pottsUploadError}
              pottsUploadBusy={pottsUploadBusy}
              handleUploadPottsModel={handleUploadPottsModel}
              pottsModels={pottsModels}
              formatPottsModelName={formatPottsModelName}
              pottsRenameValues={pottsRenameValues}
              setPottsRenameValues={setPottsRenameValues}
              pottsRenameBusy={pottsRenameBusy}
              pottsDeleteBusy={pottsDeleteBusy}
              handleDownloadPottsModel={handleDownloadPottsModel}
              handleRenamePottsModel={handleRenamePottsModel}
              handleDeletePottsModel={handleDeletePottsModel}
              pottsRenameError={pottsRenameError}
              pottsDeleteError={pottsDeleteError}
              samplingMode={samplingMode}
              setSamplingMode={setSamplingMode}
              samplingUploadProgress={samplingUploadProgress}
              handleUploadSimulationResults={handleUploadSimulationResults}
              samplingUploadBusy={samplingUploadBusy}
              enqueueSimulationJob={enqueueSimulationJob}
              simulationResults={simulationResults}
              resultsLoading={resultsLoading}
              resultsError={resultsError}
              navigate={navigate}
            />
          )}
        </section>
          )}
        </div>
      </div>
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
  );
}
