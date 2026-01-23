import { useEffect, useMemo, useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

export default function SimulationUploadForm({ clusterRuns, onSubmit, isBusy = false }) {
  const [clusterId, setClusterId] = useState('');
  const [summaryFile, setSummaryFile] = useState(null);
  const [modelFile, setModelFile] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const busy = isBusy || isSubmitting;

  const clusterOptions = useMemo(() => clusterRuns || [], [clusterRuns]);
  const selectedCluster = useMemo(
    () => clusterOptions.find((run) => run.cluster_id === clusterId),
    [clusterOptions, clusterId]
  );

  useEffect(() => {
    if (!clusterOptions.length) {
      setClusterId('');
      return;
    }
    const exists = clusterOptions.some((run) => run.cluster_id === clusterId);
    if (!clusterId || !exists) {
      setClusterId(clusterOptions[clusterOptions.length - 1].cluster_id);
    }
  }, [clusterOptions, clusterId]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      if (!clusterId) {
        throw new Error('Select a saved cluster NPZ to attach the sampling results.');
      }
      if (!summaryFile) {
        throw new Error('Upload the sampling summary NPZ (run_summary.npz).');
      }
      if (!modelFile) {
        throw new Error('Upload the Potts model NPZ used for sampling.');
      }
      await onSubmit({ cluster_id: clusterId, summaryFile, modelFile });
      setSummaryFile(null);
      setModelFile(null);
    } catch (err) {
      setError(err.message || 'Failed to upload sampling results.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm text-gray-300 mb-1">Cluster NPZ</label>
        <select
          value={clusterId}
          onChange={(event) => setClusterId(event.target.value)}
          disabled={!clusterOptions.length || busy}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
        >
          {clusterOptions.length === 0 && <option value="">No saved clusters</option>}
          {clusterOptions.map((run) => {
            const name = run.name || run.path?.split('/').pop() || run.cluster_id;
            return (
              <option key={run.cluster_id} value={run.cluster_id}>
                {name}
              </option>
            );
          })}
        </select>
        {selectedCluster && (
          <p className="text-xs text-gray-500 mt-1">
            Metastable: {Array.isArray(selectedCluster.metastable_ids) ? selectedCluster.metastable_ids.join(', ') : '-'} -
            Contact {selectedCluster.contact_atom_mode || selectedCluster.contact_mode || 'CA'} @{' '}
            {selectedCluster.contact_cutoff ?? 10} A
          </p>
        )}
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Sampling summary (NPZ)</label>
        <input
          type="file"
          accept=".npz"
          onChange={(event) => setSummaryFile(event.target.files?.[0] || null)}
          disabled={busy}
          className="w-full text-sm text-gray-300 file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:bg-gray-700 file:text-gray-100 hover:file:bg-gray-600"
        />
        <p className="text-xs text-gray-500 mt-1">Upload the `run_summary.npz` produced by local sampling.</p>
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Potts model (NPZ)</label>
        <input
          type="file"
          accept=".npz"
          onChange={(event) => setModelFile(event.target.files?.[0] || null)}
          disabled={busy}
          className="w-full text-sm text-gray-300 file:mr-3 file:py-2 file:px-3 file:rounded-md file:border-0 file:bg-gray-700 file:text-gray-100 hover:file:bg-gray-600"
        />
        <p className="text-xs text-gray-500 mt-1">Upload the `potts_model.npz` used during sampling.</p>
      </div>

      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={busy || !clusterOptions.length}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {busy ? 'Uploading...' : 'Upload sampling results'}
      </button>
    </form>
  );
}
