import { Eye, Info, Plus } from 'lucide-react';
import { InfoTooltip } from './SystemDetailWidgets';
import { getClusterDisplayName } from './systemDetailUtils';
import ErrorMessage from '../common/ErrorMessage';

export default function SystemDetailSidebar({
  states,
  metastableStates,
  metastableLocked,
  clustersUnlocked,
  clusterError,
  clusterLoading,
  clusterRuns,
  analysisMode,
  clusterJobStatus,
  setInfoOverlayState,
  setPottsFitClusterId,
  setAnalysisFocus,
  setClusterPanelOpen,
  setClusterError,
  handleDeleteSavedCluster,
  selectedClusterId,
  openDescriptorExplorer,
  openDoc,
  handleEnableMacroEditing,
  macroLocked,
  navigate,
  projectId,
  systemId,
}) {
  const buildSamplingSuffix = () => {
    const params = new URLSearchParams();
    if (selectedClusterId) params.set('cluster_id', selectedClusterId);
    const suffix = params.toString();
    return suffix ? `?${suffix}` : '';
  };

  return (
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
      {macroLocked && (
        <button
          type="button"
          onClick={handleEnableMacroEditing}
          className="w-full text-xs px-3 py-2 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10 inline-flex items-center justify-center gap-2"
        >
          Enable macro-state editing
        </button>
      )}
      <button
        type="button"
        onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/visualize${buildSamplingSuffix()}`)}
        className="w-full text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/40 inline-flex items-center justify-center gap-2"
      >
        <Eye className="h-4 w-4" />
        Sampling explorer
      </button>
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_eval${buildSamplingSuffix()}`)}
          className="text-xs px-2 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
        >
          Delta eval
        </button>
        <button
          type="button"
          onClick={() =>
            navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_commitment_3d${buildSamplingSuffix()}`)
          }
          className="text-xs px-2 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
        >
          Delta 3D
        </button>
        <button
          type="button"
          onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_js${buildSamplingSuffix()}`)}
          className="text-xs px-2 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
        >
          Delta JS
        </button>
        <button
          type="button"
          onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/delta_js_3d${buildSamplingSuffix()}`)}
          className="text-xs px-2 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
        >
          Delta JS 3D
        </button>
        <button
          type="button"
          onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/lambda_sweep${buildSamplingSuffix()}`)}
          className="text-xs px-2 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
        >
          Lambda sweep
        </button>
        <button
          type="button"
          onClick={() =>
            navigate(`/projects/${projectId}/systems/${systemId}/sampling/gibbs_relaxation${buildSamplingSuffix()}`)
          }
          className="text-xs px-2 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
        >
          Gibbs relax
        </button>
        <button
          type="button"
          onClick={() =>
            navigate(`/projects/${projectId}/systems/${systemId}/sampling/gibbs_relaxation_3d${buildSamplingSuffix()}`)
          }
          className="col-span-2 text-xs px-2 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
        >
          Gibbs relax 3D
        </button>
      </div>
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
                  <span className="flex items-center gap-2">{state.name}</span>
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
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold text-white">Cluster NPZ</h3>
                <InfoTooltip
                  ariaLabel="Storage layout info"
                  text="See how projects, systems, clusters, models, and samples are stored on disk."
                  onClick={() => openDoc('storage_layout')}
                />
              </div>
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
                    const isSelected = run.cluster_id === selectedClusterId;
                    return (
                      <div key={run.cluster_id} className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            if (run?.cluster_id) {
                              if (isSelected) {
                                setPottsFitClusterId('');
                              } else {
                                setPottsFitClusterId(run.cluster_id);
                                setAnalysisFocus('potts');
                              }
                            }
                          }}
                          className={`flex-1 text-left rounded-md border px-3 py-2 ${
                            isSelected
                              ? 'border-cyan-500 bg-cyan-500/10'
                              : 'border-gray-700 bg-gray-900/60 hover:border-cyan-500'
                          }`}
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
                        <button
                          type="button"
                          onClick={() => {
                            if (run?.cluster_id) {
                              openDescriptorExplorer({ clusterId: run.cluster_id });
                            }
                          }}
                          className="mt-1 inline-flex h-8 w-8 items-center justify-center rounded-md border border-gray-700 text-gray-200 hover:border-cyan-500 hover:text-cyan-300"
                          aria-label={`Visualize ${name}`}
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                      </div>
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
                          : '—'}
                      </p>
                      <p className="text-[11px] text-gray-500">{statusLabel}</p>
                      {isRunning && (
                        <div className="mt-2">
                          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-cyan-500 transition-all duration-300"
                              style={{ width: `${progress || 0}%` }}
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
  );
}
