import { Eye, Plus } from 'lucide-react';

import ErrorMessage from '../common/ErrorMessage';
import { InfoTooltip } from './SystemDetailWidgets';
import { getClusterDisplayName } from './systemDetailUtils';

export default function SystemClusterPanel({
  clustersUnlocked,
  clusterError,
  clusterLoading,
  clusterRuns,
  clusterJobStatus,
  selectedClusterId,
  setSelectedClusterId,
  setClusterPanelOpen,
  setClusterError,
  handleDeleteSavedCluster,
  openDescriptorExplorer,
  openDoc,
}) {
  return (
    <section className="rounded-lg border border-gray-700 bg-gray-800 p-4 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-white">Residue clusters</h2>
            <InfoTooltip
              ariaLabel="Storage layout info"
              text="Clusters provide the discrete residue states used by Potts models and sampling."
              onClick={() => openDoc('storage_layout')}
            />
          </div>
          <p className="text-xs text-gray-400">Create and inspect cluster assignments after building state descriptors.</p>
        </div>
        <button
          type="button"
          onClick={() => {
            setClusterError(null);
            setClusterPanelOpen(true);
          }}
          disabled={!clustersUnlocked || clusterLoading}
          className="inline-flex items-center gap-2 rounded-md border border-cyan-500 px-3 py-2 text-xs text-cyan-200 hover:bg-cyan-500/10 disabled:opacity-50"
        >
          <Plus className="h-4 w-4" />
          New cluster
        </button>
      </div>
      {!clustersUnlocked && <p className="text-xs text-gray-500">Build descriptors for at least one state to enable clustering.</p>}
      {clusterError && <ErrorMessage message={clusterError} />}
      {clustersUnlocked && !clusterRuns.length && <p className="text-xs text-gray-500">No cluster runs yet.</p>}
      {!!clusterRuns.length && (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {clusterRuns.slice().reverse().map((run) => {
            const snapshot = clusterJobStatus[run.cluster_id] || {};
            const status = snapshot.status || run.status || (run.path ? 'finished' : 'queued');
            const progress = snapshot?.meta?.progress ?? run.progress ?? 0;
            const ready = Boolean(run.path) && status !== 'failed';
            const selected = String(run.cluster_id) === String(selectedClusterId);
            return (
              <div key={run.cluster_id} className={`rounded-md border p-3 ${selected ? 'border-cyan-500 bg-cyan-950/20' : 'border-gray-700 bg-gray-900/50'}`}>
                <div className="flex items-start justify-between gap-2">
                  <button
                    type="button"
                    disabled={!ready}
                    onClick={() => setSelectedClusterId(run.cluster_id)}
                    className="min-w-0 flex-1 text-left disabled:cursor-default"
                  >
                    <p className="truncate text-sm font-medium text-gray-100">{getClusterDisplayName(run)}</p>
                    <p className="mt-1 text-[11px] text-gray-500">
                      {(run.cluster_algorithm || 'density peaks').toUpperCase()} · {status}
                    </p>
                  </button>
                  {ready && (
                    <button
                      type="button"
                      onClick={() => openDescriptorExplorer({ clusterId: run.cluster_id })}
                      className="rounded border border-gray-700 p-1.5 text-gray-300 hover:border-cyan-500 hover:text-cyan-200"
                      aria-label={`Visualize ${getClusterDisplayName(run)}`}
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                  )}
                </div>
                {!ready && status !== 'failed' && (
                  <div className="mt-3 h-1.5 overflow-hidden rounded bg-gray-800">
                    <div className="h-full bg-cyan-500" style={{ width: `${Math.max(0, Math.min(100, Number(progress) || 0))}%` }} />
                  </div>
                )}
                {status === 'failed' && (
                  <div className="mt-2 flex items-center justify-between gap-2 text-[11px] text-red-300">
                    <span className="truncate">{run.error || snapshot?.result?.error || 'Clustering failed.'}</span>
                    <button type="button" onClick={() => handleDeleteSavedCluster(run.cluster_id)} className="text-red-200 hover:text-red-100">Delete</button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
