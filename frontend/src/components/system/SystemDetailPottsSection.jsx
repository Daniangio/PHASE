import { useState } from 'react';
import { Download, Eye, Plus, SlidersHorizontal, Trash2, UploadCloud, X } from 'lucide-react';
import ErrorMessage from '../common/ErrorMessage';
import { AnalysisResultsList, InfoTooltip } from './SystemDetailWidgets';
import SimulationAnalysisForm from '../analysis/SimulationAnalysisForm';
import SimulationUploadForm from '../analysis/SimulationUploadForm';

export default function SystemDetailPottsSection(props) {
  const {
    mdSamples,
    gibbsSamples,
    saSamples,
    metastableById,
    states,
    openDescriptorExplorer,
    pottsFitClusterId,
    pottsFitMode,
    setPottsFitMode,
    setPottsFitClusterId,
    readyClusterRuns,
    pottsModelName,
    setPottsModelName,
    handleDownloadSavedCluster,
    handleBackmappingAction,
    backmappingJobStatus,
    backmappingProgressById,
    handleDeleteSavedCluster,
    handleUploadBackmappingTrajectories,
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
    navigate,
    projectId,
    systemId,
    handleDeleteSample,
  } = props;

  const [fitOverlayOpen, setFitOverlayOpen] = useState(false);
  const [samplingOverlayOpen, setSamplingOverlayOpen] = useState(false);
  const [backmappingUploadOpen, setBackmappingUploadOpen] = useState(false);
  const [backmappingUploadError, setBackmappingUploadError] = useState(null);
  const [backmappingUploadBusy, setBackmappingUploadBusy] = useState(false);
  const [backmappingFiles, setBackmappingFiles] = useState({});

  const clusterLabel =
    selectedClusterName || selectedCluster?.name || selectedCluster?.cluster_id || '';
  const clusterFileName = selectedCluster?.path?.split('/').pop();
  const backmappingJob = pottsFitClusterId ? backmappingJobStatus?.[pottsFitClusterId] : null;
  const backmappingProgress = pottsFitClusterId ? backmappingProgressById?.[pottsFitClusterId] : null;

  return (
    <div className="space-y-4">
      <div className="grid xl:grid-cols-[minmax(0,1.4fr)_320px_320px] gap-6">
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
                  onClick={() => {
                    setBackmappingUploadError(null);
                    setBackmappingUploadOpen(true);
                  }}
                  className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
                >
                  <UploadCloud className="h-4 w-4" />
                  Backmapping NPZ
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
              {backmappingProgress !== null && backmappingProgress !== undefined && (
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-[11px] text-gray-500">
                    <span>Backmapping progress</span>
                    <span>{backmappingProgress}%</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-cyan-500 transition-all duration-300"
                      style={{ width: `${backmappingProgress || 0}%` }}
                    />
                  </div>
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
            </>
          )}
        </section>

        <aside className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-3">
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
          {pottsFitClusterId && pottsModels.length === 0 && (
            <p className="text-[11px] text-gray-500">No Potts models yet.</p>
          )}
          {pottsModels.length > 0 && (
            <div className="space-y-3">
              {pottsModels.map((run) => {
                const displayName = formatPottsModelName(run);
                const value = pottsRenameValues[run.model_id] ?? displayName;
                const isBusy = pottsRenameBusy[run.model_id];
                const isDeleting = pottsDeleteBusy[run.model_id];
                return (
                  <div key={run.model_id} className="rounded-md border border-gray-800 bg-gray-950/50 p-2 space-y-2">
                    <input
                      type="text"
                      value={value}
                      onChange={(event) =>
                        setPottsRenameValues((prev) => ({
                          ...prev,
                          [run.model_id]: event.target.value,
                        }))
                      }
                      className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-xs text-white focus:ring-cyan-500"
                    />
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() =>
                          handleDownloadPottsModel(run.cluster_id, run.model_id, run.path?.split('/').pop())
                        }
                        disabled={isBusy || isDeleting}
                        className="text-[11px] px-2 py-1 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50"
                      >
                        Download
                      </button>
                      <button
                        type="button"
                        onClick={() => handleRenamePottsModel(run.cluster_id, run.model_id)}
                        disabled={isBusy || isDeleting}
                        className="text-[11px] px-2 py-1 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 disabled:opacity-50"
                      >
                        {isBusy ? 'Saving…' : 'Rename'}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDeletePottsModel(run.cluster_id, run.model_id)}
                        disabled={isDeleting || isBusy}
                        className="text-[11px] px-2 py-1 rounded-md border border-red-500/40 text-red-200 hover:bg-red-500/10 disabled:opacity-50"
                      >
                        {isDeleting ? 'Deleting…' : 'Delete'}
                      </button>
                    </div>
                  </div>
                );
              })}
              {pottsRenameError && <ErrorMessage message={pottsRenameError} />}
              {pottsDeleteError && <ErrorMessage message={pottsDeleteError} />}
            </div>
          )}
        </aside>

        <aside className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-3">
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
              {mdSamples.length === 0 && <p className="text-[11px] text-gray-500 mt-1">No MD samples yet.</p>}
              {mdSamples.length > 0 && (
                <div className="space-y-2 mt-2">
                  {mdSamples.map((sample) => {
                    const meta = sample.metastable_id ? metastableById.get(sample.metastable_id) : null;
                    const stateId = sample.state_id || meta?.macro_state_id;
                    const stateName =
                      states.find((s) => s.state_id === stateId)?.name || stateId || 'Unknown state';
                    const label = meta
                      ? `${meta.name || meta.default_name || meta.metastable_id} (${stateName})`
                      : stateName;
                    return (
                      <div
                        key={sample.sample_id || sample.path}
                        className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/50 px-2 py-1"
                      >
                        <span className="text-[11px] text-gray-300 truncate">{label}</span>
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
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-300">From Gibbs</p>
              {gibbsSamples.length === 0 && <p className="text-[11px] text-gray-500 mt-1">No Gibbs samples yet.</p>}
              {gibbsSamples.length > 0 && (
                <div className="space-y-1 mt-2">
                  {gibbsSamples.map((sample) => (
                      <div
                        key={sample.sample_id || sample.path}
                        className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/40 px-2 py-1 text-[11px] text-gray-300"
                      >
                        <span className="truncate">{sample.name || 'Gibbs sample'} • {sample.created_at || ''}</span>
                        <div className="flex items-center gap-2">
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
                    ))}
                </div>
              )}
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-300">From SA</p>
              {saSamples.length === 0 && <p className="text-[11px] text-gray-500 mt-1">No SA samples yet.</p>}
              {saSamples.length > 0 && (
                <div className="space-y-1 mt-2">
                  {saSamples.map((sample) => (
                      <div
                        key={sample.sample_id || sample.path}
                        className="flex items-center justify-between gap-2 rounded-md border border-gray-800 bg-gray-950/40 px-2 py-1 text-[11px] text-gray-300"
                      >
                        <span className="truncate">{sample.name || 'SA sample'} • {sample.created_at || ''}</span>
                        <div className="flex items-center gap-2">
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
                    ))}
                </div>
              )}
            </div>
          </div>
        </aside>
      </div>

      {fitOverlayOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-4xl bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white">Fit Potts model</h3>
                <p className="text-xs text-gray-500">Run on server or upload a fitted model.</p>
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
                      { key: 'plm_lr', label: 'PLM lr', placeholder: '1e-3' },
                      { key: 'plm_lr_min', label: 'PLM lr min', placeholder: '1e-4' },
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
      )}

      {backmappingUploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
          <div className="w-full max-w-3xl bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-white">Upload trajectories for backmapping</h3>
                <p className="text-xs text-gray-500">
                  Upload trajectories again to build a backmapping NPZ without storing them.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setBackmappingUploadOpen(false)}
                className="text-gray-400 hover:text-gray-200"
                aria-label="Close"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-2">
              {states.map((state) => (
                <div key={state.state_id} className="flex items-center gap-3">
                  <div className="w-40 text-xs text-gray-300">{state.name || state.state_id}</div>
                  <input
                    type="file"
                    accept=".xtc,.trr,.dcd,.nc,.h5,.hdf5"
                    onChange={(event) =>
                      setBackmappingFiles((prev) => ({
                        ...prev,
                        [state.state_id]: event.target.files?.[0] || null,
                      }))
                    }
                    className="flex-1 text-xs text-gray-200"
                  />
                </div>
              ))}
            </div>
            {backmappingUploadError && <ErrorMessage message={backmappingUploadError} />}
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setBackmappingUploadOpen(false)}
                className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
                disabled={backmappingUploadBusy}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={async () => {
                  setBackmappingUploadError(null);
                  if (!pottsFitClusterId) {
                    setBackmappingUploadError('Select a cluster first.');
                    return;
                  }
                  const files = Object.entries(backmappingFiles)
                    .filter(([, file]) => Boolean(file))
                    .reduce((acc, [key, file]) => {
                      acc[key] = file;
                      return acc;
                    }, {});
                  if (!Object.keys(files).length) {
                    setBackmappingUploadError('Upload at least one trajectory.');
                    return;
                  }
                  try {
                    setBackmappingUploadBusy(true);
                    await handleUploadBackmappingTrajectories(pottsFitClusterId, files);
                    setBackmappingUploadOpen(false);
                    setBackmappingFiles({});
                  } catch (err) {
                    setBackmappingUploadError(err.message || 'Failed to build backmapping NPZ.');
                  } finally {
                    setBackmappingUploadBusy(false);
                  }
                }}
                className="text-xs px-3 py-2 rounded-md border border-cyan-500 text-cyan-200 hover:bg-cyan-500/10 disabled:opacity-60"
                disabled={backmappingUploadBusy}
              >
                {backmappingUploadBusy ? 'Building…' : 'Build backmapping NPZ'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
