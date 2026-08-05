import { useEffect, useMemo, useState } from 'react';
import { Check, Download, Eye, Info, Pencil, Plus, SlidersHorizontal, Trash2, UploadCloud, X } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import ErrorMessage from '../common/ErrorMessage';
import { AnalysisResultsList, InfoTooltip } from './SystemDetailWidgets';
import SimulationAnalysisForm from '../analysis/SimulationAnalysisForm';
import SimulationUploadForm from '../analysis/SimulationUploadForm';
import { createLambdaPottsModel, fetchSampleStats } from '../../api/projects';

export default function SystemDetailPottsSection(props) {
  const {
    allSamples: allSampleEntries = [],
    mdSamples,
    gibbsSamples,
    saSamples,
    metastableById,
    states,
    openDescriptorExplorer,
    pottsFitClusterId,
    pottsFitMode,
    setPottsFitMode,
    pottsFitKind,
    setPottsFitKind,
    pottsFitStartMode,
    setPottsFitStartMode,
    pottsFitBaseModelId,
    setPottsFitBaseModelId,
    pottsFitSampleIds,
    setPottsFitSampleIds,
    pottsFitExistingMode,
    setPottsFitExistingMode,
    pottsDeltaBaseModelId,
    setPottsDeltaBaseModelId,
    pottsDeltaStateIds,
    setPottsDeltaStateIds,
    setPottsFitClusterId,
    readyClusterRuns,
    pottsModelName,
    setPottsModelName,
    handleDownloadSavedCluster,
    handleDownloadSampleBackmapping,
    handleSubmitSampleBackmappingDataset,
    sampleBackmappingJobStatus,
    handleDeleteSavedCluster,
    pottsFitMethod,
    setPottsFitMethod,
    pottsFitContactMode,
    setPottsFitContactMode,
    pottsFitContactCutoff,
    setPottsFitContactCutoff,
    pottsFitAdvanced,
    setPottsFitAdvanced,
    pottsFitParams,
    setPottsFitParams,
    pottsDeltaParams,
    setPottsDeltaParams,
    pottsFitError,
    enqueuePottsFitJob,
    pottsFitSubmitting,
    pottsFitResults,
    pottsFitResultsWithClusters,
    handleDeleteResult,
    openDoc,
    selectedCluster,
    selectedClusterName,
    pottsUploadName,
    setPottsUploadName,
    setPottsUploadFile,
    pottsUploadFile,
    pottsUploadProgress,
    pottsUploadError,
    pottsUploadBusy,
    handleUploadPottsModel,
    pottsModels,
    formatPottsModelName,
    pottsRenameValues,
    setPottsRenameValues,
    pottsRenameBusy,
    pottsDeleteBusy,
    handleDownloadPottsModel,
    handleRenamePottsModel,
    handleDeletePottsModel,
    pottsRenameError,
    pottsDeleteError,
    samplingMode,
    setSamplingMode,
    samplingUploadProgress,
    handleUploadSimulationResults,
    samplingUploadBusy,
    enqueueSimulationJob,
    enqueueMdSamplesRefreshJob,
    navigate,
    projectId,
    systemId,
    handleDeleteSample,
  } = props;
  const allSamples = useMemo(
    () => allSampleEntries.length ? allSampleEntries : [...mdSamples, ...gibbsSamples, ...saSamples],
    [allSampleEntries, mdSamples, gibbsSamples, saSamples]
  );
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = searchParams.get('tab') === 'samples' ? 'samples' : 'models';
  const linkedModelIds = useMemo(
    () => new Set((searchParams.get('model_ids') || '').split(',').map((value) => value.trim()).filter(Boolean)),
    [searchParams]
  );
  const linkedSampleIds = useMemo(
    () => new Set((searchParams.get('sample_ids') || '').split(',').map((value) => value.trim()).filter(Boolean)),
    [searchParams]
  );
  const [modelSearch, setModelSearch] = useState('');
  const [modelKindFilter, setModelKindFilter] = useState('all');
  const [sampleSearch, setSampleSearch] = useState('');
  const [sampleMethodFilter, setSampleMethodFilter] = useState('all');
  const [sampleModelFilter, setSampleModelFilter] = useState('all');

  const setWorkspaceTab = (tab, extra = {}) => {
    const next = new URLSearchParams(searchParams);
    next.set('tab', tab);
    Object.entries(extra).forEach(([key, value]) => {
      if (value) next.set(key, value);
      else next.delete(key);
    });
    setSearchParams(next);
  };

  const sampleModelIds = (sample) => {
    const values = [
      sample?.model_id,
      ...(Array.isArray(sample?.model_ids) ? sample.model_ids : []),
      sample?.params?.model_id,
      ...(Array.isArray(sample?.params?.model_ids) ? sample.params.model_ids : []),
    ];
    return Array.from(new Set(values.map((value) => String(value || '').trim()).filter(Boolean)));
  };

  const visiblePottsModels = useMemo(() => {
    const query = modelSearch.trim().toLowerCase();
    return (pottsModels || []).filter((model) => {
      if (linkedModelIds.size && !linkedModelIds.has(String(model.model_id || ''))) return false;
      if (linkedSampleIds.size) {
        const linked = allSamples
          .filter((sample) => linkedSampleIds.has(String(sample.sample_id || '')))
          .some((sample) => sampleModelIds(sample).includes(String(model.model_id || '')));
        if (!linked) return false;
      }
      const kind = model?.params?.delta_kind || (model?.params?.fit_mode === 'delta' ? 'delta' : 'standard');
      if (modelKindFilter !== 'all' && kind !== modelKindFilter) return false;
      if (!query) return true;
      return `${formatPottsModelName(model)} ${model.model_id || ''}`.toLowerCase().includes(query);
    });
  }, [allSamples, formatPottsModelName, linkedModelIds, linkedSampleIds, modelKindFilter, modelSearch, pottsModels]);

  const sampleMatchesFilters = (sample) => {
    const query = sampleSearch.trim().toLowerCase();
    const ids = sampleModelIds(sample);
    if (linkedModelIds.size && !ids.some((id) => linkedModelIds.has(id))) return false;
    if (linkedSampleIds.size && !linkedSampleIds.has(String(sample?.sample_id || ''))) return false;
    if (sampleModelFilter !== 'all' && !ids.includes(sampleModelFilter)) return false;
    const method = String(sample?.method || (sample?.type === 'md_eval' ? 'md' : 'other')).toLowerCase();
    if (sampleMethodFilter === 'other' && ['md', 'gibbs', 'sa'].includes(method)) return false;
    if (sampleMethodFilter !== 'all' && sampleMethodFilter !== 'other' && method !== sampleMethodFilter) return false;
    if (!query) return true;
    return `${sample?.name || ''} ${sample?.sample_id || ''} ${method} ${ids.join(' ')}`.toLowerCase().includes(query);
  };
  const visibleMdSamples = allSamples.filter((sample) => sample?.type === 'md_eval' && sampleMatchesFilters(sample));
  const visibleGibbsSamples = allSamples.filter((sample) => sample?.type !== 'md_eval' && String(sample?.method || '').toLowerCase() === 'gibbs' && sampleMatchesFilters(sample));
  const visibleSaSamples = allSamples.filter((sample) => sample?.type !== 'md_eval' && String(sample?.method || '').toLowerCase() === 'sa' && sampleMatchesFilters(sample));
  const visibleOtherSamples = allSamples.filter((sample) => {
    const method = String(sample?.method || '').toLowerCase();
    return sample?.type !== 'md_eval' && method !== 'gibbs' && method !== 'sa' && sampleMatchesFilters(sample);
  });

  const [fitOverlayOpen, setFitOverlayOpen] = useState(false);
  const [samplingOverlayOpen, setSamplingOverlayOpen] = useState(false);
  const [sampleBackmappingOpen, setSampleBackmappingOpen] = useState(false);
  const [sampleBackmappingSampleId, setSampleBackmappingSampleId] = useState('');
  const [sampleBackmappingFile, setSampleBackmappingFile] = useState(null);
  const [sampleBackmappingError, setSampleBackmappingError] = useState(null);
  const [sampleBackmappingBusy, setSampleBackmappingBusy] = useState(false);
  const [sampleBackmappingUploadProgress, setSampleBackmappingUploadProgress] = useState(null);
  const [renameEditingId, setRenameEditingId] = useState(null);
  const [infoModel, setInfoModel] = useState(null);
  const [infoSampleId, setInfoSampleId] = useState(null);
  const [lambdaCreateBusy, setLambdaCreateBusy] = useState(false);
  const [lambdaCreateError, setLambdaCreateError] = useState(null);
  const [lambdaModelAId, setLambdaModelAId] = useState('');
  const [lambdaModelBId, setLambdaModelBId] = useState('');
  const [lambdaValue, setLambdaValue] = useState(0.5);
  const [lambdaModelName, setLambdaModelName] = useState('');
  const [assignBusy, setAssignBusy] = useState(false);
  const [assignError, setAssignError] = useState(null);
  const [assignStateIds, setAssignStateIds] = useState([]);

  const clusterLabel =
    selectedClusterName || selectedCluster?.name || selectedCluster?.cluster_id || '';
  const clusterFileName = selectedCluster?.path?.split('/').pop();
  const analysisStateOptions = useMemo(() => {
    const options = [];
    (states || []).forEach((state) => {
      if (!state?.state_id) return;
      options.push({
        value: state.state_id,
        label: `[macro] ${state.name || state.state_id}`,
      });
    });
    if (metastableById instanceof Map) {
      metastableById.forEach((meta, key) => {
        const id = meta?.metastable_id || meta?.id || key;
        if (!id) return;
        options.push({
          value: String(id),
          label: `[metastable] ${meta?.name || meta?.default_name || id}`,
        });
      });
    }
    return options;
  }, [states, metastableById]);
  const baseNameById = useMemo(() => {
    const map = new Map();
    (pottsModels || []).forEach((model) => {
      if (model?.model_id) {
        map.set(model.model_id, formatPottsModelName(model));
      }
    });
    return map;
  }, [pottsModels, formatPottsModelName]);

  const infoSample = useMemo(
    () => allSamples.find((sample) => sample.sample_id === infoSampleId) || null,
    [allSamples, infoSampleId]
  );
  const infoModelBestLoss = useMemo(() => {
    if (!infoModel) return null;
    const params = infoModel?.params || {};
    const summary = infoModel?.summary || {};
    const candidates = [
      params.delta_best_loss,
      params.plm_best_loss,
      params.best_loss,
      summary.delta_best_loss,
      summary.plm_best_loss,
      summary.best_loss,
      infoModel.delta_best_loss,
      infoModel.plm_best_loss,
      infoModel.best_loss,
    ];
    for (const value of candidates) {
      const num = Number(value);
      if (Number.isFinite(num)) return num;
    }
    return null;
  }, [infoModel]);
  useEffect(() => {
    if (!infoModel) return;
    const currentId = infoModel.model_id || infoModel.id || infoModel.path || '';
    const matched = (pottsModels || []).find((model) => {
      const id = model?.model_id || model?.id || model?.path || '';
      return String(id) === String(currentId);
    });
    if (!matched) setInfoModel(null);
    else if (matched !== infoModel) setInfoModel(matched);
  }, [pottsModels, infoModel]);
  const [infoSampleStats, setInfoSampleStats] = useState(null);
  const [infoSampleStatsError, setInfoSampleStatsError] = useState(null);
  const sampleBackmappingStatusById = useMemo(() => {
    const statusMap = {};
    mdSamples.forEach((sample) => {
      const persisted = sample?.backmapping_dataset || {};
      const transient = sampleBackmappingJobStatus?.[sample.sample_id] || {};
      statusMap[sample.sample_id] = {
        ...persisted,
        ...transient,
        status: transient.status || persisted.status || (persisted.path ? 'finished' : ''),
      };
    });
    return statusMap;
  }, [mdSamples, sampleBackmappingJobStatus]);
  const selectedBackmappingSample = useMemo(
    () => mdSamples.find((sample) => sample.sample_id === sampleBackmappingSampleId) || null,
    [mdSamples, sampleBackmappingSampleId]
  );

  const renderInlineSampleInfo = (sample) => {
    if (!sample || !infoSample || sample.sample_id !== infoSample.sample_id) return null;
    return (
      <div className="rounded-md border border-gray-800 bg-gray-950/60 p-2 text-[11px] text-gray-300 space-y-2">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="text-xs font-semibold text-white">{infoSample.name || infoSample.sample_id}</p>
            <p className="text-[10px] text-gray-500">Sample info</p>
          </div>
          <button
            type="button"
            onClick={() => setInfoSampleId(null)}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close sample info"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
        <div className="space-y-1">
          <div><span className="text-gray-400">id:</span> {infoSample.sample_id}</div>
          {infoSample.created_at && <div><span className="text-gray-400">created:</span> {infoSample.created_at}</div>}
          {infoSample.method && <div><span className="text-gray-400">method:</span> {infoSample.method}</div>}
          {infoSample.model_names && infoSample.model_names.length > 0 && (
            <div><span className="text-gray-400">models:</span> {infoSample.model_names.join(', ')}</div>
          )}
          {infoSample.path && <div><span className="text-gray-400">path:</span> {infoSample.path}</div>}
          {infoSampleStats && (
            <>
              <div><span className="text-gray-400">frames:</span> {infoSampleStats.n_frames}</div>
              <div><span className="text-gray-400">residues:</span> {infoSampleStats.n_residues}</div>
              <div>
                <span className="text-gray-400">invalid:</span>{' '}
                {infoSampleStats.invalid_count} ({(infoSampleStats.invalid_fraction * 100).toFixed(2)}%)
              </div>
            </>
          )}
          {infoSampleStatsError && (
            <div className="text-red-300">{infoSampleStatsError}</div>
          )}
        </div>
        {(infoSample.summary || infoSample.params) && (
          <details className="text-[11px] text-gray-300">
            <summary className="cursor-pointer text-gray-200">Run details</summary>
            <pre className="mt-2 max-h-56 overflow-auto rounded bg-gray-900 p-2 text-[10px] text-gray-300">
              {JSON.stringify(infoSample.summary || infoSample.params, null, 2)}
            </pre>
          </details>
        )}
      </div>
    );
  };

  useEffect(() => {
    const load = async () => {
      setInfoSampleStats(null);
      setInfoSampleStatsError(null);
      if (!infoSampleId || !pottsFitClusterId) return;
      try {
        const stats = await fetchSampleStats(projectId, systemId, pottsFitClusterId, infoSampleId);
        setInfoSampleStats(stats);
      } catch (err) {
        setInfoSampleStatsError(err.message || 'Failed to load sample stats.');
      }
    };
    load();
  }, [infoSampleId, pottsFitClusterId, projectId, systemId]);

  useEffect(() => {
    if (!fitOverlayOpen) return;
    if (pottsFitMode !== 'lambda') return;
    if (!pottsModels?.length) return;
    if (!lambdaModelBId) setLambdaModelBId(pottsModels[0]?.model_id || '');
    if (!lambdaModelAId) {
      const fallback = pottsModels[Math.min(1, pottsModels.length - 1)]?.model_id || pottsModels[0]?.model_id || '';
      setLambdaModelAId(fallback);
    }
  }, [fitOverlayOpen, pottsFitMode, pottsModels, lambdaModelAId, lambdaModelBId]);

  useEffect(() => {
    if (!samplingOverlayOpen) return;
    if (samplingMode !== 'assign') return;
    const defaults = (states || [])
      .filter((state) => Boolean(state?.descriptor_file))
      .map((state) => state.state_id)
      .filter(Boolean);
    setAssignStateIds(defaults);
  }, [samplingOverlayOpen, samplingMode, states]);

  const handleCreateLambdaModel = async () => {
    setLambdaCreateError(null);
    if (!pottsFitClusterId) {
      setLambdaCreateError('Select a cluster first.');
      return;
    }
    if (!lambdaModelAId || !lambdaModelBId) {
      setLambdaCreateError('Select two endpoint models.');
      return;
    }
    if (lambdaModelAId === lambdaModelBId) {
      setLambdaCreateError('Endpoint models must be different.');
      return;
    }
    const lam = Number(lambdaValue);
    if (!Number.isFinite(lam) || lam < 0 || lam > 1) {
      setLambdaCreateError('Lambda must be in [0,1].');
      return;
    }
    setLambdaCreateBusy(true);
    try {
      await createLambdaPottsModel(projectId, systemId, pottsFitClusterId, {
        model_a_id: lambdaModelAId,
        model_b_id: lambdaModelBId,
        lam,
        name: lambdaModelName?.trim() ? lambdaModelName.trim() : undefined,
        zero_sum_gauge: true,
      });
      if (typeof props.refreshSystem === 'function') {
        await props.refreshSystem();
      }
      setLambdaModelName('');
      setFitOverlayOpen(false);
    } catch (err) {
      setLambdaCreateError(err.message || 'Failed to create lambda model.');
    } finally {
      setLambdaCreateBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 rounded-lg border border-gray-700 bg-gray-900/70 p-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex items-center gap-1">
          <button type="button" onClick={() => setWorkspaceTab('models')} className={`rounded-md px-4 py-2 text-sm ${activeTab === 'models' ? 'bg-cyan-600 text-white' : 'text-gray-300 hover:bg-gray-800'}`}>Models</button>
          <button type="button" onClick={() => setWorkspaceTab('samples')} className={`rounded-md px-4 py-2 text-sm ${activeTab === 'samples' ? 'bg-cyan-600 text-white' : 'text-gray-300 hover:bg-gray-800'}`}>Samples</button>
        </div>
        {activeTab === 'models' ? (
          <div className="flex flex-wrap items-center gap-2">
            <input value={modelSearch} onChange={(event) => setModelSearch(event.target.value)} placeholder="Filter models by name or ID" className="min-w-64 rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-xs text-white" />
            <select value={modelKindFilter} onChange={(event) => setModelKindFilter(event.target.value)} className="rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-xs text-white">
              <option value="all">All model types</option>
              <option value="standard">Standard</option>
              <option value="delta">Delta</option>
            </select>
            {(linkedModelIds.size > 0 || linkedSampleIds.size > 0) && <button type="button" onClick={() => setWorkspaceTab('models', { model_ids: '', sample_ids: '' })} className="text-xs text-cyan-300 hover:text-cyan-200">Clear linked filter</button>}
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            <input value={sampleSearch} onChange={(event) => setSampleSearch(event.target.value)} placeholder="Filter samples" className="min-w-64 rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-xs text-white" />
            <select value={sampleMethodFilter} onChange={(event) => setSampleMethodFilter(event.target.value)} className="rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-xs text-white">
              <option value="all">All methods</option>
              <option value="md">MD</option>
              <option value="gibbs">Gibbs</option>
              <option value="sa">SA</option>
              <option value="other">Other</option>
            </select>
            <select value={sampleModelFilter} onChange={(event) => setSampleModelFilter(event.target.value)} className="max-w-64 rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-xs text-white">
              <option value="all">All source models</option>
              {pottsModels.map((model) => <option key={model.model_id} value={model.model_id}>{formatPottsModelName(model)}</option>)}
            </select>
            {(linkedModelIds.size > 0 || linkedSampleIds.size > 0) && <button type="button" onClick={() => setWorkspaceTab('samples', { model_ids: '', sample_ids: '' })} className="text-xs text-cyan-300 hover:text-cyan-200">Clear linked filter</button>}
          </div>
        )}
      </div>
      <div className="grid xl:grid-cols-[320px_minmax(0,1fr)] gap-3">
        <section className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-md font-semibold text-white">Selected Cluster</h3>
              <p className="text-xs text-gray-500 mt-1">
                {pottsFitClusterId
                  ? 'Cluster actions and diagnostics.'
                  : 'Select a cluster from the left panel to see details here.'}
              </p>
            </div>
            <InfoTooltip
              ariaLabel="Potts analysis documentation"
              text="Cluster-specific Potts models and sampling are scoped to the selected NPZ."
              onClick={() => openDoc('potts_overview')}
            />
          </div>
          <select
            value={pottsFitClusterId}
            onChange={(event) => setPottsFitClusterId(event.target.value)}
            className="w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-xs text-white"
          >
            {!readyClusterRuns.length && <option value="">No clusters available</option>}
            {readyClusterRuns.map((run) => (
              <option key={run.cluster_id} value={run.cluster_id}>{run.name || run.cluster_name || run.cluster_id}</option>
            ))}
          </select>
          {!pottsFitClusterId && (
            <div className="rounded-md border border-dashed border-gray-700 bg-gray-950/40 p-4 text-sm text-gray-400">
              No cluster selected yet.
            </div>
          )}
          {pottsFitClusterId && (
            <>
              <div className="grid md:grid-cols-2 gap-3 text-sm">
                <div className="rounded-md border border-gray-800 bg-gray-950/50 p-3 space-y-1">
                  <p className="text-xs text-gray-500">Cluster</p>
                  <p className="text-sm text-white">{clusterLabel}</p>
                  <p className="text-[11px] text-gray-500">{selectedCluster?.cluster_id}</p>
                </div>
                <div className="rounded-md border border-gray-800 bg-gray-950/50 p-3 space-y-1">
                  <p className="text-xs text-gray-500">Algorithm</p>
                  <p className="text-sm text-white">
                    {selectedCluster?.cluster_algorithm || 'density_peaks'}
                  </p>
                  <p className="text-[11px] text-gray-500">
                    Max frames: {selectedCluster?.max_cluster_frames ?? 'all'}
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => handleDownloadSavedCluster(pottsFitClusterId, clusterFileName)}
                  className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
                >
                  <Download className="h-4 w-4" />
                  {'Download cluster NPZ'}
                </button>
                <button
                  type="button"
                  onClick={() => handleDeleteSavedCluster(pottsFitClusterId)}
                  className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-red-500/40 text-red-200 hover:bg-red-500/10"
                >
                  <Trash2 className="h-4 w-4" />
                  Delete cluster
                </button>
              </div>
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
            </>
          )}
        </section>

        {activeTab === 'models' && <aside className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-3 min-w-[320px]">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-white">Potts models</h3>
              <p className="text-[11px] text-gray-500">Models fitted for this cluster.</p>
            </div>
            <button
              type="button"
              onClick={() => setFitOverlayOpen(true)}
              className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md border border-cyan-500 text-cyan-200 hover:bg-cyan-500/10"
            >
              <Plus className="h-4 w-4" />
              New
            </button>
          </div>
          {!pottsFitClusterId && (
            <p className="text-[11px] text-gray-500">Select a cluster to view its Potts models.</p>
          )}
          {pottsFitClusterId && visiblePottsModels.length === 0 && (
            <p className="text-[11px] text-gray-500">No Potts models yet.</p>
          )}
          {visiblePottsModels.length > 0 && (
            <div className="space-y-3">
              {visiblePottsModels.map((run) => {
                const displayName = formatPottsModelName(run);
                const isBusy = pottsRenameBusy[run.model_id];
                const isDeleting = pottsDeleteBusy[run.model_id];
                const deltaKind = run?.params?.delta_kind || (run?.params?.fit_mode === 'delta' ? 'delta' : null);
                const baseModelId = run?.params?.base_model_id;
                const baseLabel = baseModelId ? baseNameById.get(baseModelId) : null;
                const isEditing = renameEditingId === run.model_id;
                const draftValue = pottsRenameValues[run.model_id] ?? displayName;
                return (
                  <div key={run.model_id} className="rounded-md border border-gray-800 bg-gray-950/50 p-2 space-y-2">
                    <div className="flex items-center gap-2">
                      {!isEditing ? (
                        <p className="flex-1 min-w-0 text-sm text-white truncate">{displayName}</p>
                      ) : (
                        <input
                          type="text"
                          value={draftValue}
                          onChange={(event) =>
                            setPottsRenameValues((prev) => ({
                              ...prev,
                              [run.model_id]: event.target.value,
                            }))
                          }
                          className="flex-1 min-w-0 bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-xs text-white focus:ring-cyan-500"
                        />
                      )}
                      <div className="flex items-center gap-2 shrink-0">
                        <button
                          type="button"
                          onClick={() => setInfoModel(run)}
                          disabled={isBusy || isDeleting}
                          className="inline-flex items-center justify-center rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50 p-1.5"
                          aria-label="Potts model metadata"
                        >
                          <Info className="h-4 w-4" />
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            handleDownloadPottsModel(run.cluster_id, run.model_id, run.path?.split('/').pop())
                          }
                          disabled={isBusy || isDeleting}
                          className="inline-flex items-center justify-center rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50 p-1.5"
                          aria-label="Download Potts model"
                        >
                          <Download className="h-4 w-4" />
                        </button>
                        {!isEditing && (
                          <button
                            type="button"
                            onClick={() => {
                              setPottsRenameValues((prev) => ({
                                ...prev,
                                [run.model_id]: displayName,
                              }));
                              setRenameEditingId(run.model_id);
                            }}
                            disabled={isBusy || isDeleting}
                            className="inline-flex items-center justify-center rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50 p-1.5"
                            aria-label="Rename Potts model"
                          >
                            <Pencil className="h-4 w-4" />
                          </button>
                        )}
                        {isEditing && (
                          <>
                            <button
                              type="button"
                              onClick={async () => {
                                const name = (pottsRenameValues[run.model_id] || '').trim();
                                if (!name) {
                                  await handleRenamePottsModel(run.cluster_id, run.model_id);
                                  return;
                                }
                                await handleRenamePottsModel(run.cluster_id, run.model_id);
                                setRenameEditingId(null);
                              }}
                              disabled={isBusy || isDeleting}
                              className="inline-flex items-center justify-center rounded-md border border-cyan-500 text-cyan-200 hover:bg-cyan-500/10 disabled:opacity-50 p-1.5"
                              aria-label="Confirm rename"
                            >
                              <Check className="h-4 w-4" />
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setRenameEditingId(null);
                                setPottsRenameValues((prev) => {
                                  const next = { ...prev };
                                  delete next[run.model_id];
                                  return next;
                                });
                              }}
                              disabled={isBusy || isDeleting}
                              className="inline-flex items-center justify-center rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50 p-1.5"
                              aria-label="Cancel rename"
                            >
                              <X className="h-4 w-4" />
                            </button>
                          </>
                        )}
                        <button
                          type="button"
                          onClick={() => handleDeletePottsModel(run.cluster_id, run.model_id)}
                          disabled={isDeleting || isBusy}
                          className="inline-flex items-center justify-center rounded-md border border-red-500/40 text-red-200 hover:bg-red-500/10 disabled:opacity-50 p-1.5"
                          aria-label="Delete Potts model"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                    {deltaKind && (
                      <p className="text-[10px] text-cyan-300">
                        Δ patch{baseLabel ? ` · base: ${baseLabel}` : ''}
                      </p>
                    )}
                    <button
                      type="button"
                      onClick={() => setWorkspaceTab('samples', { model_ids: run.model_id, sample_ids: '' })}
                      className="text-[11px] text-cyan-300 hover:text-cyan-200"
                    >
                      Show samples generated with this model
                    </button>
                    {infoModel && String(infoModel.model_id || '') === String(run.model_id || '') && (
                      <div className="mt-2 rounded-md border border-gray-800 bg-gray-950/60 p-2 text-[11px] text-gray-300 space-y-2">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <p className="text-xs font-semibold text-white">{formatPottsModelName(infoModel)}</p>
                            <p className="text-[10px] text-gray-500">Potts model metadata</p>
                          </div>
                          <button
                            type="button"
                            onClick={() => setInfoModel(null)}
                            className="text-gray-400 hover:text-gray-200"
                            aria-label="Close model info"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                        <div className="space-y-1">
                          <div><span className="text-gray-400">id:</span> {infoModel.model_id}</div>
                          {infoModel.created_at && <div><span className="text-gray-400">created:</span> {infoModel.created_at}</div>}
                          {infoModel.source && <div><span className="text-gray-400">source:</span> {infoModel.source}</div>}
                          {infoModel.path && <div><span className="text-gray-400">path:</span> {infoModel.path}</div>}
                          {infoModelBestLoss != null && (
                            <div><span className="text-gray-400">best loss:</span> {infoModelBestLoss.toFixed(6)}</div>
                          )}
                        </div>
                        {(infoModel.summary || infoModel.params) && (
                          <details className="text-[11px] text-gray-300">
                            <summary className="cursor-pointer text-gray-200">Details</summary>
                            <pre className="mt-2 max-h-56 overflow-auto rounded bg-gray-900 p-2 text-[10px] text-gray-300">
                              {JSON.stringify(infoModel.summary || infoModel.params, null, 2)}
                            </pre>
                          </details>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
              {pottsRenameError && <ErrorMessage message={pottsRenameError} />}
              {pottsDeleteError && <ErrorMessage message={pottsDeleteError} />}
            </div>
          )}
        </aside>}

        {activeTab === 'samples' && <aside className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-white">Samples</h3>
              <p className="text-[11px] text-gray-500">MD, Gibbs, and SA outputs.</p>
            </div>
            <button
              type="button"
              onClick={() => setSamplingOverlayOpen(true)}
              className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md border border-cyan-500 text-cyan-200 hover:bg-cyan-500/10"
            >
              <Plus className="h-4 w-4" />
              New
            </button>
          </div>
          {!pottsFitClusterId && (
            <p className="text-[11px] text-gray-500">Select a cluster to see its samples.</p>
          )}
          <div className="space-y-3">
            <div>
              <p className="text-xs font-semibold text-gray-300">From MD</p>
              {visibleMdSamples.length === 0 && <p className="text-[11px] text-gray-500 mt-1">No matching MD samples.</p>}
              {visibleMdSamples.length > 0 && (
                <div className="space-y-2 mt-2">
                  {visibleMdSamples.map((sample) => {
                    const meta = sample.metastable_id ? metastableById.get(sample.metastable_id) : null;
                    const stateId = sample.state_id || meta?.macro_state_id;
                    const stateName =
                      states.find((s) => s.state_id === stateId)?.name || stateId || 'Unknown state';
                    const label = meta
                      ? `${meta.name || meta.default_name || meta.metastable_id} (${stateName})`
                      : stateName;
                    const backmapping = sampleBackmappingStatusById[sample.sample_id] || {};
                    const backmappingStatus = backmapping.status || '';
                    const canBuildBackmapping = Boolean(sample.state_id);
                    const isBackmappingActive = backmapping.job_id && !['finished', 'failed'].includes(backmappingStatus);
                    const backmappingStatusLabel = backmapping.path
                      ? backmappingStatus === 'failed'
                        ? 'dataset ready; last rebuild failed'
                        : isBackmappingActive
                          ? `dataset ${backmappingStatus}`
                          : 'dataset ready'
                      : backmappingStatus
                        ? `dataset ${backmappingStatus}`
                        : 'no dataset';
                    return (
                      <div key={sample.sample_id || sample.path} className="space-y-2">
                        <div className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/50 px-2 py-1">
                          <div className="min-w-0">
                            <p className="text-[11px] text-gray-300 truncate">{label}</p>
                            <p className="text-[10px] text-gray-500 truncate">
                              {backmappingStatusLabel}
                              {isBackmappingActive && backmapping.meta?.progress !== undefined
                                ? ` • ${backmapping.meta.progress}%`
                                : ''}
                            </p>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              onClick={() => setInfoSampleId(sample.sample_id)}
                              className="text-gray-400 hover:text-gray-200"
                              aria-label={`Show info for ${label}`}
                            >
                              <Info className="h-4 w-4" />
                            </button>
                            {stateId && (
                              <button
                                type="button"
                                onClick={() =>
                                  openDescriptorExplorer({
                                    clusterId: pottsFitClusterId,
                                    stateId,
                                    metastableId: sample.metastable_id || null,
                                  })
                                }
                                className="text-gray-400 hover:text-cyan-300"
                                aria-label={`View ${label} in Descriptor Explorer`}
                              >
                                <Eye className="h-4 w-4" />
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={() => {
                                setSampleBackmappingSampleId(sample.sample_id || '');
                                setSampleBackmappingFile(null);
                                setSampleBackmappingError(null);
                                setSampleBackmappingUploadProgress(null);
                                setSampleBackmappingOpen(true);
                              }}
                              className={`${
                                canBuildBackmapping ? 'text-gray-400 hover:text-cyan-300' : 'text-gray-700 cursor-not-allowed'
                              }`}
                              aria-label={`Build backmapping dataset for ${label}`}
                              disabled={!canBuildBackmapping}
                              title={
                                canBuildBackmapping
                                  ? 'Build or rebuild backmapping dataset'
                                  : 'Backmapping dataset currently requires a state-based MD sample'
                              }
                            >
                              <UploadCloud className="h-4 w-4" />
                            </button>
                            {backmapping.path && (
                              <button
                                type="button"
                                onClick={() => handleDownloadSampleBackmapping(pottsFitClusterId, sample)}
                                className="text-gray-400 hover:text-cyan-300"
                                aria-label={`Download backmapping dataset for ${label}`}
                              >
                                <Download className="h-4 w-4" />
                              </button>
                            )}
                          </div>
                        </div>
                        {renderInlineSampleInfo(sample)}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-300">From Gibbs</p>
              {visibleGibbsSamples.length === 0 && <p className="text-[11px] text-gray-500 mt-1">No matching Gibbs samples.</p>}
              {visibleGibbsSamples.length > 0 && (
                <div className="space-y-1 mt-2">
                  {visibleGibbsSamples.map((sample) => (
                    <div key={sample.sample_id || sample.path} className="space-y-2">
                      <div className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/40 px-2 py-1 text-[11px] text-gray-300">
                        <span className="truncate">{sample.name || 'Gibbs sample'} • {sample.created_at || ''}</span>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => setInfoSampleId(sample.sample_id)}
                            className="text-gray-400 hover:text-gray-200"
                            aria-label={`Show info for ${sample.name || 'Gibbs sample'}`}
                          >
                            <Info className="h-4 w-4" />
                          </button>
                          <button
                            type="button"
                            onClick={() =>
                              navigate(
                                `/projects/${projectId}/systems/${systemId}/sampling/visualize?cluster_id=${pottsFitClusterId}&sample_id=${sample.sample_id}`
                              )
                            }
                            className="text-gray-400 hover:text-cyan-300"
                            aria-label={`View report for ${sample.name || 'Gibbs sample'}`}
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDeleteSample(sample.sample_id)}
                            className="text-gray-400 hover:text-red-300"
                            aria-label={`Delete ${sample.name || 'Gibbs sample'}`}
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                      {renderInlineSampleInfo(sample)}
                      {!!sampleModelIds(sample).length && (
                        <button type="button" onClick={() => setWorkspaceTab('models', { model_ids: sampleModelIds(sample).join(','), sample_ids: sample.sample_id })} className="text-[11px] text-cyan-300 hover:text-cyan-200">Show source model</button>
                      )}
                    </div>
                    ))}
                </div>
              )}
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-300">From SA</p>
              {visibleSaSamples.length === 0 && <p className="text-[11px] text-gray-500 mt-1">No matching SA samples.</p>}
              {visibleSaSamples.length > 0 && (
                <div className="space-y-1 mt-2">
                  {visibleSaSamples.map((sample) => (
                    <div key={sample.sample_id || sample.path} className="space-y-2">
                      <div className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/40 px-2 py-1 text-[11px] text-gray-300">
                        <span className="truncate">{sample.name || 'SA sample'} • {sample.created_at || ''}</span>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => setInfoSampleId(sample.sample_id)}
                            className="text-gray-400 hover:text-gray-200"
                            aria-label={`Show info for ${sample.name || 'SA sample'}`}
                          >
                            <Info className="h-4 w-4" />
                          </button>
                          <button
                            type="button"
                            onClick={() =>
                              navigate(
                                `/projects/${projectId}/systems/${systemId}/sampling/visualize?cluster_id=${pottsFitClusterId}&sample_id=${sample.sample_id}`
                              )
                            }
                            className="text-gray-400 hover:text-cyan-300"
                            aria-label={`View report for ${sample.name || 'SA sample'}`}
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleDeleteSample(sample.sample_id)}
                            className="text-gray-400 hover:text-red-300"
                            aria-label={`Delete ${sample.name || 'SA sample'}`}
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                      {renderInlineSampleInfo(sample)}
                      {!!sampleModelIds(sample).length && (
                        <button type="button" onClick={() => setWorkspaceTab('models', { model_ids: sampleModelIds(sample).join(','), sample_ids: sample.sample_id })} className="text-[11px] text-cyan-300 hover:text-cyan-200">Show source model</button>
                      )}
                    </div>
                    ))}
                </div>
              )}
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-300">Other sampling methods</p>
              {visibleOtherSamples.length === 0 && <p className="text-[11px] text-gray-500 mt-1">No matching samples.</p>}
              {visibleOtherSamples.length > 0 && (
                <div className="space-y-1 mt-2">
                  {visibleOtherSamples.map((sample) => (
                    <div key={sample.sample_id || sample.path} className="rounded-md border border-gray-800 bg-gray-950/40 px-2 py-2 text-[11px] text-gray-300">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate">{sample.name || sample.sample_id}</span>
                        <div className="flex items-center gap-2">
                          <button type="button" onClick={() => setInfoSampleId(sample.sample_id)} className="text-gray-400 hover:text-gray-200" aria-label={`Show info for ${sample.name || sample.sample_id}`}><Info className="h-4 w-4" /></button>
                          <button type="button" onClick={() => handleDeleteSample(sample.sample_id)} className="text-gray-400 hover:text-red-300" aria-label={`Delete ${sample.name || sample.sample_id}`}><Trash2 className="h-4 w-4" /></button>
                        </div>
                      </div>
                      {renderInlineSampleInfo(sample)}
                      {!!sampleModelIds(sample).length && (
                        <button type="button" onClick={() => setWorkspaceTab('models', { model_ids: sampleModelIds(sample).join(','), sample_ids: sample.sample_id })} className="mt-1 text-[11px] text-cyan-300 hover:text-cyan-200">Show source models</button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </aside>}
      </div>

      {fitOverlayOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-4xl bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white">New Potts model</h3>
                <p className="text-xs text-gray-500">Run fitting, upload a model, or create a derived lambda model.</p>
              </div>
              <button
                type="button"
                onClick={() => setFitOverlayOpen(false)}
                className="text-gray-400 hover:text-gray-200"
                aria-label="Close"
              >
                <X className="h-5 w-5" />
              </button>
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
              <button
                type="button"
                onClick={() => setPottsFitMode('lambda')}
                className={`px-3 py-1 rounded-full border ${
                  pottsFitMode === 'lambda'
                    ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10'
                    : 'border-gray-700 text-gray-400 hover:border-gray-500'
                }`}
              >
                Lambda model
              </button>
            </div>
            {pottsFitMode === 'run' && (
              <div className="space-y-3">
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
                <div className="flex items-center gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => setPottsFitKind('standard')}
                    className={`px-3 py-1 rounded-full border ${
                      pottsFitKind === 'standard'
                        ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10'
                        : 'border-gray-700 text-gray-400 hover:border-gray-500'
                    }`}
                  >
                    Standard fit
                  </button>
                  <button
                    type="button"
                    onClick={() => setPottsFitKind('delta')}
                    className={`px-3 py-1 rounded-full border ${
                      pottsFitKind === 'delta'
                        ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10'
                        : 'border-gray-700 text-gray-400 hover:border-gray-500'
                    }`}
                  >
                    Delta fit
                  </button>
                </div>
                {(pottsFitKind === 'delta' || pottsFitStartMode !== 'existing' || pottsFitExistingMode !== 'resume') && (
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
                )}
                {pottsFitKind === 'delta' && (
                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm text-gray-300 mb-1">Base Potts model</label>
                      <select
                        value={pottsDeltaBaseModelId}
                        onChange={(event) => setPottsDeltaBaseModelId(event.target.value)}
                        disabled={!pottsModels.length}
                        className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
                      >
                        {!pottsModels.length && <option value="">No base models available</option>}
                        {pottsModels.map((model) => (
                          <option key={model.model_id} value={model.model_id}>
                            {formatPottsModelName(model)}
                          </option>
                        ))}
                      </select>
                      <p className="text-xs text-gray-500 mt-1">
                        Delta fit learns sparse patches on top of the selected base model.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <p className="text-xs text-gray-400">
                        Select one or more macro/metastable states to fit the delta patch.
                      </p>
                      {analysisStateOptions.length === 0 && (
                        <p className="text-xs text-gray-500">No states available for delta fit.</p>
                      )}
                      {analysisStateOptions.length > 0 && (
                        <div className="grid md:grid-cols-2 gap-2 text-xs text-gray-200">
                          {analysisStateOptions.map((opt) => {
                            const checked = pottsDeltaStateIds.includes(opt.value);
                            return (
                              <label key={`delta-${opt.value}`} className="flex items-center gap-2">
                                <input
                                  type="checkbox"
                                  checked={checked}
                                  onChange={() =>
                                    setPottsDeltaStateIds((prev) =>
                                      checked
                                        ? prev.filter((id) => id !== opt.value)
                                        : [...prev, opt.value]
                                    )
                                  }
                                  className="h-4 w-4 text-cyan-500 rounded border-gray-600 bg-gray-900"
                                />
                                <span>{opt.label}</span>
                              </label>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                )}
                {pottsFitKind !== 'delta' && (
                  <>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <label className="block text-sm text-gray-300">Training MD samples</label>
                          <p className="text-xs text-gray-500 mt-1">
                            Assigned labels from the selected MD samples are concatenated and used for fitting.
                          </p>
                        </div>
                        <div className="flex items-center gap-2 text-xs">
                          <button
                            type="button"
                            onClick={() => setPottsFitSampleIds(mdSamples.map((sample) => sample.sample_id))}
                            disabled={!mdSamples.length}
                            className="px-2 py-1 rounded border border-gray-700 text-gray-300 hover:border-gray-500 disabled:opacity-50"
                          >
                            All
                          </button>
                          <button
                            type="button"
                            onClick={() => setPottsFitSampleIds([])}
                            disabled={!mdSamples.length}
                            className="px-2 py-1 rounded border border-gray-700 text-gray-300 hover:border-gray-500 disabled:opacity-50"
                          >
                            None
                          </button>
                        </div>
                      </div>
                      {!mdSamples.length && (
                        <div className="rounded-md border border-dashed border-gray-700 bg-gray-950/40 p-3 text-xs text-gray-500">
                          No MD evaluation samples are available in this cluster yet.
                        </div>
                      )}
                      {mdSamples.length > 0 && (
                        <div className="grid md:grid-cols-2 gap-2 text-xs text-gray-200">
                          {mdSamples.map((sample) => {
                            const checked = pottsFitSampleIds.includes(sample.sample_id);
                            const label = sample.name || sample.sample_id;
                            const stateLabel = sample.state_id || sample.state_name || sample.method || 'md_eval';
                            return (
                              <div
                                key={`fit-sample-${sample.sample_id}`}
                                className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/50 px-3 py-2"
                              >
                                <label className="flex min-w-0 items-center gap-2">
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={() =>
                                      setPottsFitSampleIds((prev) =>
                                        checked
                                          ? prev.filter((id) => id !== sample.sample_id)
                                          : [...prev, sample.sample_id]
                                      )
                                    }
                                    className="h-4 w-4 rounded border-gray-600 bg-gray-900 text-cyan-500"
                                  />
                                  <span className="min-w-0">
                                    <span className="block truncate">{label}</span>
                                    <span className="block text-[10px] text-gray-500">{stateLabel}</span>
                                  </span>
                                </label>
                                <button
                                  type="button"
                                  onClick={() => setInfoSampleId(sample.sample_id)}
                                  className="text-gray-500 hover:text-gray-300"
                                  aria-label={`Show info for ${label}`}
                                >
                                  <Info className="h-3.5 w-3.5" />
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                    <div>
                      <label className="block text-sm text-gray-300 mb-1">Starting point</label>
                      <select
                        value={pottsFitStartMode}
                        onChange={(event) => setPottsFitStartMode(event.target.value)}
                        className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                      >
                        <option value="scratch">Fit from scratch</option>
                        <option value="existing">Start from existing model</option>
                      </select>
                    </div>
                    {pottsFitStartMode === 'existing' && (
                      <div className="space-y-2 text-sm">
                        <label className="space-y-1">
                          <span className="text-xs text-gray-400">Base model</span>
                          <select
                            value={pottsFitBaseModelId}
                            onChange={(event) => setPottsFitBaseModelId(event.target.value)}
                            disabled={!pottsModels.length}
                            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
                          >
                            {!pottsModels.length && <option value="">No models available</option>}
                            {pottsModels.map((model) => (
                              <option key={model.model_id} value={model.model_id}>
                                {formatPottsModelName(model)}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="space-y-1">
                          <span className="text-xs text-gray-400">Action</span>
                          <select
                            value={pottsFitExistingMode}
                            onChange={(event) => setPottsFitExistingMode(event.target.value)}
                            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                          >
                            <option value="resume">Improve this model</option>
                            <option value="init">New model initialized from it</option>
                          </select>
                        </label>
                        <p className="text-xs text-gray-500">
                          Contact edges will be reused from the selected model.
                        </p>
                      </div>
                    )}
                    <div>
                      <label className="block text-sm text-gray-300 mb-1">Fit method</label>
                      <select
                        value={pottsFitMethod}
                        onChange={(event) => setPottsFitMethod(event.target.value)}
                        disabled={pottsFitStartMode === 'existing'}
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
                          disabled={pottsFitStartMode === 'existing'}
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
                          disabled={pottsFitStartMode === 'existing'}
                          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                        />
                      </label>
                    </div>
                  </>
                )}
                <button
                  type="button"
                  onClick={() => setPottsFitAdvanced((prev) => !prev)}
                  className="flex items-center gap-2 text-xs text-cyan-300 hover:text-cyan-200"
                >
                  <SlidersHorizontal className="h-4 w-4" />
                  {pottsFitAdvanced ? 'Hide' : 'Show'} {pottsFitKind === 'delta' ? 'delta' : 'fit'} hyperparams
                </button>
                {pottsFitAdvanced && pottsFitKind !== 'delta' && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                      {[
                        { key: 'plm_epochs', label: 'PLM epochs', placeholder: '200' },
                        { key: 'plm_lr', label: 'PLM lr', placeholder: '1e-3' },
                        { key: 'plm_lr_min', label: 'PLM lr min', placeholder: '1e-4' },
                        { key: 'plm_l2', label: 'PLM L2', placeholder: '1e-5' },
                        { key: 'plm_batch_size', label: 'Batch size', placeholder: '512' },
                        { key: 'plm_grad_accum_steps', label: 'Grad accum steps', placeholder: '1' },
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
                      <label className="space-y-1">
                        <span className="text-xs text-gray-400">PLM device</span>
                        <input
                          type="text"
                          placeholder="auto / cpu / cuda / cuda:0"
                          value={pottsFitParams.plm_device}
                          onChange={(event) =>
                            setPottsFitParams((prev) => ({
                              ...prev,
                              plm_device: event.target.value,
                            }))
                          }
                          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                        />
                      </label>
                    </div>
                    {pottsFitMethod !== 'pmi' && pottsFitStartMode !== 'existing' && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                        <label className="space-y-1">
                          <span className="text-xs text-gray-400">PLM init</span>
                          <select
                            value={pottsFitParams.plm_init}
                            onChange={(event) =>
                              setPottsFitParams((prev) => ({
                                ...prev,
                                plm_init: event.target.value,
                              }))
                            }
                            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                          >
                            <option value="pmi">PMI</option>
                            <option value="zero">Zero</option>
                            <option value="model">Model</option>
                          </select>
                        </label>
                        <label className="space-y-1">
                          <span className="text-xs text-gray-400">Resume model path</span>
                          <input
                            type="text"
                            placeholder="path/to/model.npz"
                            value={pottsFitParams.plm_resume_model}
                            onChange={(event) => {
                              const value = event.target.value;
                              setPottsFitParams((prev) => ({
                                ...prev,
                                plm_resume_model: value,
                                plm_init: value ? 'model' : prev.plm_init,
                              }));
                            }}
                            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                          />
                        </label>
                        {pottsFitParams.plm_init === 'model' && !pottsFitParams.plm_resume_model && (
                          <label className="space-y-1">
                            <span className="text-xs text-gray-400">Init model path</span>
                            <input
                              type="text"
                              placeholder="path/to/model.npz"
                              value={pottsFitParams.plm_init_model}
                              onChange={(event) =>
                                setPottsFitParams((prev) => ({
                                  ...prev,
                                  plm_init_model: event.target.value,
                                }))
                              }
                              className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                            />
                          </label>
                        )}
                      </div>
                    )}
                    {(pottsFitMethod !== 'pmi' || pottsFitStartMode === 'existing') && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                        <label className="space-y-1">
                          <span className="text-xs text-gray-400">Val fraction</span>
                          <input
                            type="text"
                            placeholder="0"
                            value={pottsFitParams.plm_val_frac}
                            onChange={(event) =>
                              setPottsFitParams((prev) => ({
                                ...prev,
                                plm_val_frac: event.target.value,
                              }))
                            }
                            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                          />
                        </label>
                      </div>
                    )}
                  </div>
                )}
                {pottsFitAdvanced && pottsFitKind === 'delta' && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                      {[
                        { key: 'delta_epochs', label: 'Epochs', placeholder: '200' },
                        { key: 'delta_lr', label: 'Learning rate', placeholder: '1e-3' },
                        { key: 'delta_lr_min', label: 'LR min', placeholder: '1e-3' },
                        { key: 'delta_batch_size', label: 'Batch size', placeholder: '512' },
                        { key: 'delta_grad_accum_steps', label: 'Grad accum steps', placeholder: '1' },
                        { key: 'delta_seed', label: 'Random seed', placeholder: '0' },
                        { key: 'delta_l2', label: 'Delta L2', placeholder: '0.0' },
                        { key: 'delta_group_h', label: 'Group sparsity (fields)', placeholder: '0.0' },
                        { key: 'delta_group_j', label: 'Group sparsity (couplings)', placeholder: '0.0' },
                      ].map((field) => (
                        <label key={field.key} className="space-y-1">
                          <span className="text-xs text-gray-400">{field.label}</span>
                          <input
                            type="text"
                            placeholder={field.placeholder}
                            value={pottsDeltaParams[field.key]}
                            onChange={(event) =>
                              setPottsDeltaParams((prev) => ({
                                ...prev,
                                [field.key]: event.target.value,
                              }))
                            }
                            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                          />
                        </label>
                      ))}
                    </div>
                    <div className="grid md:grid-cols-2 gap-3 text-sm">
                      <label className="space-y-1">
                        <span className="text-xs text-gray-400">LR schedule</span>
                        <select
                          value={pottsDeltaParams.delta_lr_schedule}
                          onChange={(event) =>
                            setPottsDeltaParams((prev) => ({
                              ...prev,
                              delta_lr_schedule: event.target.value,
                            }))
                          }
                          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                        >
                          <option value="cosine">Cosine</option>
                          <option value="none">None</option>
                        </select>
                      </label>
                      <label className="space-y-1">
                        <span className="text-xs text-gray-400">Device</span>
                        <select
                          value={pottsDeltaParams.delta_device}
                          onChange={(event) =>
                            setPottsDeltaParams((prev) => ({
                              ...prev,
                              delta_device: event.target.value,
                            }))
                          }
                          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                        >
                          <option value="auto">Auto</option>
                          <option value="cpu">CPU</option>
                          <option value="cuda">CUDA</option>
                        </select>
                      </label>
                    </div>
                    <div className="grid md:grid-cols-2 gap-3 text-sm">
                      <label className="space-y-1">
                        <span className="text-xs text-gray-400">Unassigned policy</span>
                        <select
                          value={pottsDeltaParams.unassigned_policy}
                          onChange={(event) =>
                            setPottsDeltaParams((prev) => ({
                              ...prev,
                              unassigned_policy: event.target.value,
                            }))
                          }
                          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                        >
                          <option value="drop_frames">Drop frames</option>
                          <option value="treat_as_state">Treat as state</option>
                          <option value="error">Error</option>
                        </select>
                      </label>
                      <label className="flex items-center gap-2 text-xs text-gray-300 mt-6">
                        <input
                          type="checkbox"
                          checked={!!pottsDeltaParams.delta_no_combined}
                          onChange={(event) =>
                            setPottsDeltaParams((prev) => ({
                              ...prev,
                              delta_no_combined: event.target.checked,
                            }))
                          }
                          className="h-4 w-4 text-cyan-500 rounded border-gray-600 bg-gray-900"
                        />
                        Skip saving combined models
                      </label>
                    </div>
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
              </div>
            )}
            {pottsFitMode === 'lambda' && (
              <div className="space-y-3 border border-gray-700 rounded-md p-3">
                <p className="text-xs text-gray-400">
                  Create a new Potts model by interpolating two existing models and saving the result for later use (analysis/sampling).
                </p>
                {clusterLabel && <p className="text-[11px] text-gray-500">Selected cluster: {clusterLabel}</p>}
                <div className="grid md:grid-cols-2 gap-3 text-sm">
                  <label className="space-y-1">
                    <span className="text-xs text-gray-400">Endpoint model B (λ=0)</span>
                    <select
                      value={lambdaModelBId}
                      onChange={(event) => setLambdaModelBId(event.target.value)}
                      disabled={!pottsModels.length}
                      className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
                    >
                      {!pottsModels.length && <option value="">No models available</option>}
                      {pottsModels.map((model) => (
                        <option key={model.model_id} value={model.model_id}>
                          {formatPottsModelName(model)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="space-y-1">
                    <span className="text-xs text-gray-400">Endpoint model A (λ=1)</span>
                    <select
                      value={lambdaModelAId}
                      onChange={(event) => setLambdaModelAId(event.target.value)}
                      disabled={!pottsModels.length}
                      className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
                    >
                      {!pottsModels.length && <option value="">No models available</option>}
                      {pottsModels.map((model) => (
                        <option key={model.model_id} value={model.model_id}>
                          {formatPottsModelName(model)}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                <div className="grid md:grid-cols-[1fr_140px] gap-3 items-end">
                  <label className="space-y-1">
                    <span className="text-xs text-gray-400">λ (0..1)</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={lambdaValue}
                      onChange={(event) => setLambdaValue(Number(event.target.value))}
                      className="w-full"
                    />
                  </label>
                  <label className="space-y-1">
                    <span className="text-xs text-gray-400">λ value</span>
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step="0.01"
                      value={lambdaValue}
                      onChange={(event) => setLambdaValue(Number(event.target.value))}
                      className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                    />
                  </label>
                </div>

                <div>
                  <label className="block text-xs text-gray-400 mb-1">Model name (optional)</label>
                  <input
                    type="text"
                    value={lambdaModelName}
                    onChange={(event) => setLambdaModelName(event.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                    placeholder="Default: Lambda 0.300 ModelB -> ModelA"
                  />
                  <p className="text-[11px] text-gray-500 mt-1">Endpoints are zero-sum gauged before interpolation.</p>
                </div>

                {lambdaCreateError && <ErrorMessage message={lambdaCreateError} />}

                <button
                  type="button"
                  onClick={handleCreateLambdaModel}
                  disabled={lambdaCreateBusy || !pottsFitClusterId || !pottsModels.length}
                  className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
                >
                  {lambdaCreateBusy ? 'Saving…' : 'Save lambda model'}
                </button>
              </div>
            )}
            {pottsFitMode === 'upload' && (
              <div className="border border-gray-700 rounded-md p-3 space-y-2">
                <p className="text-xs text-gray-400">Upload a fitted model for the selected cluster.</p>
                {clusterLabel && (
                  <p className="text-[11px] text-gray-500">Selected cluster: {clusterLabel}</p>
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
                  <UploadCloud className="h-4 w-4" />
                  {pottsUploadBusy ? 'Uploading…' : 'Upload model'}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {samplingOverlayOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-4xl bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white">Potts sampling</h3>
                <p className="text-xs text-gray-500">Run sampling or upload results.</p>
              </div>
              <button
                type="button"
                onClick={() => setSamplingOverlayOpen(false)}
                className="text-gray-400 hover:text-gray-200"
                aria-label="Close"
              >
                <X className="h-5 w-5" />
              </button>
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
              <button
                type="button"
                onClick={() => setSamplingMode('assign')}
                className={`px-2 py-1 rounded-md border ${
                  samplingMode === 'assign'
                    ? 'border-cyan-400 text-cyan-200'
                    : 'border-gray-700 text-gray-400 hover:text-gray-200'
                }`}
              >
                Assign MD states
              </button>
            </div>
            {samplingMode === 'run' ? (
              <SimulationAnalysisForm clusterRuns={readyClusterRuns} onSubmit={enqueueSimulationJob} />
            ) : samplingMode === 'upload' ? (
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
            ) : (
              <div className="space-y-3">
                <p className="text-xs text-gray-400">
                  Create or refresh MD assignment samples for selected macro states.
                </p>
                <div className="max-h-64 overflow-auto rounded-md border border-gray-800 bg-gray-950/50 p-2 space-y-2">
                  {(states || [])
                    .filter((state) => Boolean(state?.descriptor_file))
                    .map((state) => (
                      <label key={state.state_id} className="flex items-center gap-2 text-xs text-gray-200">
                        <input
                          type="checkbox"
                          checked={assignStateIds.includes(state.state_id)}
                          onChange={(event) => {
                            const checked = event.target.checked;
                            setAssignStateIds((prev) => {
                              const set = new Set(prev);
                              if (checked) set.add(state.state_id);
                              else set.delete(state.state_id);
                              return Array.from(set);
                            });
                          }}
                        />
                        <span>{state.name || state.state_id}</span>
                        <span className="text-gray-500">{state.state_id}</span>
                      </label>
                    ))}
                </div>
                {assignError && <ErrorMessage message={assignError} />}
                <div className="flex items-center justify-between gap-2">
                  <button
                    type="button"
                    onClick={() =>
                      setAssignStateIds(
                        (states || [])
                          .filter((state) => Boolean(state?.descriptor_file))
                          .map((state) => state.state_id)
                          .filter(Boolean)
                      )
                    }
                    className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-300 hover:bg-gray-800"
                    disabled={assignBusy}
                  >
                    Select all
                  </button>
                  <button
                    type="button"
                    onClick={async () => {
                      setAssignError(null);
                      if (!pottsFitClusterId) {
                        setAssignError('Select a cluster first.');
                        return;
                      }
                      if (!assignStateIds.length) {
                        setAssignError('Select at least one state.');
                        return;
                      }
                      setAssignBusy(true);
                      try {
                        if (typeof enqueueMdSamplesRefreshJob !== 'function') {
                          throw new Error('State assignment job submission is unavailable.');
                        }
                        await enqueueMdSamplesRefreshJob({
                          cluster_id: pottsFitClusterId,
                          state_ids: assignStateIds,
                          overwrite: true,
                          cleanup: true,
                        });
                        setSamplingOverlayOpen(false);
                      } catch (err) {
                        setAssignError(err.message || 'Failed to assign selected states.');
                      } finally {
                        setAssignBusy(false);
                      }
                    }}
                    className="text-xs px-3 py-2 rounded-md border border-cyan-500 text-cyan-200 hover:bg-cyan-500/10 disabled:opacity-50"
                    disabled={assignBusy || !pottsFitClusterId}
                  >
                    {assignBusy ? 'Assigning…' : 'Run assignment'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {sampleBackmappingOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-2xl bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white">Build backmapping dataset</h3>
                <p className="text-xs text-gray-500">
                  Upload the trajectory corresponding to this MD sample. The dataset is saved on the sample and can be downloaded later.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSampleBackmappingOpen(false)}
                className="text-gray-400 hover:text-gray-200"
                aria-label="Close"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            {!selectedBackmappingSample && (
              <p className="text-sm text-gray-400">Sample not found.</p>
            )}
            {selectedBackmappingSample && (
              <>
                <div className="rounded-md border border-gray-800 bg-gray-950/50 p-3 space-y-1 text-xs text-gray-300">
                  <p>
                    <span className="text-gray-500">Sample:</span> {selectedBackmappingSample.name || selectedBackmappingSample.sample_id}
                  </p>
                  <p>
                    <span className="text-gray-500">State:</span>{' '}
                    {states.find((state) => state.state_id === selectedBackmappingSample.state_id)?.name ||
                      selectedBackmappingSample.state_id ||
                      'Unavailable'}
                  </p>
                  <p>
                    <span className="text-gray-500">Status:</span>{' '}
                    {selectedBackmappingSample.backmapping_dataset?.status ||
                      (selectedBackmappingSample.backmapping_dataset?.path ? 'finished' : 'not built')}
                  </p>
                  {selectedBackmappingSample.backmapping_dataset?.path && (
                    <p>
                      <span className="text-gray-500">Cached dataset:</span>{' '}
                      {selectedBackmappingSample.backmapping_dataset.n_frames || '?'} frames,{' '}
                      {selectedBackmappingSample.backmapping_dataset.n_atoms || '?'} atoms,{' '}
                      {selectedBackmappingSample.backmapping_dataset.n_residues || '?'} residues
                    </p>
                  )}
                </div>
                <div className="space-y-2">
                  <label className="block text-xs text-gray-400">Trajectory file</label>
                  <input
                    type="file"
                    accept=".xtc,.trr,.dcd,.nc,.h5,.hdf5,.pdb"
                    onChange={(event) => setSampleBackmappingFile(event.target.files?.[0] || null)}
                    className="w-full text-xs text-gray-200"
                  />
                  {sampleBackmappingUploadProgress !== null && sampleBackmappingUploadProgress !== undefined && (
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-[11px] text-gray-500">
                        <span>Upload progress</span>
                        <span>{sampleBackmappingUploadProgress}%</span>
                      </div>
                      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-cyan-500 transition-all duration-300"
                          style={{ width: `${sampleBackmappingUploadProgress || 0}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>
                {sampleBackmappingError && <ErrorMessage message={sampleBackmappingError} />}
                <div className="flex items-center justify-between gap-2">
                  <div className="text-[11px] text-gray-500">
                    {selectedBackmappingSample.backmapping_dataset?.path
                      ? 'A new run overwrites the stored dataset if it completes successfully.'
                      : 'The dataset will be stored on this sample and exposed as a download action.'}
                  </div>
                  <div className="flex items-center gap-2">
                    {selectedBackmappingSample.backmapping_dataset?.path && (
                      <button
                        type="button"
                        onClick={() => handleDownloadSampleBackmapping(pottsFitClusterId, selectedBackmappingSample)}
                        className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
                      >
                        Download current
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => setSampleBackmappingOpen(false)}
                      className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
                      disabled={sampleBackmappingBusy}
                    >
                      Close
                    </button>
                    <button
                      type="button"
                      onClick={async () => {
                        setSampleBackmappingError(null);
                        if (!pottsFitClusterId || !selectedBackmappingSample?.sample_id) {
                          setSampleBackmappingError('Select a cluster sample first.');
                          return;
                        }
                        if (!sampleBackmappingFile) {
                          setSampleBackmappingError('Upload a trajectory file.');
                          return;
                        }
                        try {
                          setSampleBackmappingBusy(true);
                          await handleSubmitSampleBackmappingDataset(
                            pottsFitClusterId,
                            selectedBackmappingSample.sample_id,
                            sampleBackmappingFile,
                            {
                              onUploadProgress: (percent) => setSampleBackmappingUploadProgress(percent),
                            }
                          );
                          setSampleBackmappingOpen(false);
                          setSampleBackmappingFile(null);
                          setSampleBackmappingUploadProgress(null);
                        } catch (err) {
                          setSampleBackmappingError(err.message || 'Failed to submit backmapping dataset job.');
                        } finally {
                          setSampleBackmappingBusy(false);
                        }
                      }}
                      className="text-xs px-3 py-2 rounded-md border border-cyan-500 text-cyan-200 hover:bg-cyan-500/10 disabled:opacity-60"
                      disabled={sampleBackmappingBusy}
                    >
                      {sampleBackmappingBusy ? 'Submitting…' : 'Build dataset'}
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
