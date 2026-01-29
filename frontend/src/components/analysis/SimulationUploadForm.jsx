import { useEffect, useMemo, useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

export default function SimulationUploadForm({ clusterRuns, onSubmit, isBusy = false }) {
  const [clusterId, setClusterId] = useState('');
  const [compareClusterIds, setCompareClusterIds] = useState([]);
  const [summaryFile, setSummaryFile] = useState(null);
  const [pottsModelId, setPottsModelId] = useState('');
  const [sampleName, setSampleName] = useState('');
  const [samplingMethod, setSamplingMethod] = useState('gibbs');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const busy = isBusy || isSubmitting;

  const clusterOptions = useMemo(() => clusterRuns || [], [clusterRuns]);
  const compareOptions = useMemo(
    () => clusterOptions.filter((run) => run.cluster_id !== clusterId),
    [clusterOptions, clusterId]
  );
  const selectedCluster = useMemo(
    () => clusterOptions.find((run) => run.cluster_id === clusterId),
    [clusterOptions, clusterId]
  );
  const pottsModels = useMemo(() => selectedCluster?.potts_models || [], [selectedCluster]);

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

  useEffect(() => {
    if (!pottsModels.length) {
      setPottsModelId('');
      return;
    }
    const exists = pottsModels.some((model) => model.model_id === pottsModelId);
    if (!pottsModelId || !exists) {
      setPottsModelId(pottsModels[0].model_id);
    }
  }, [pottsModels, pottsModelId]);

  useEffect(() => {
    if (!compareClusterIds.length) return;
    const allowed = new Set(compareOptions.map((run) => run.cluster_id));
    const filtered = compareClusterIds.filter((id) => allowed.has(id));
    if (filtered.length !== compareClusterIds.length) {
      setCompareClusterIds(filtered);
    }
  }, [compareClusterIds, compareOptions]);

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
      if (!pottsModelId) {
        throw new Error('Select the Potts model used for sampling.');
      }
      await onSubmit({
        cluster_id: clusterId,
        compare_cluster_ids: compareClusterIds,
        summaryFile,
        pottsModelId,
        sampleName,
        samplingMethod,
      });
      setCompareClusterIds([]);
      setSummaryFile(null);
      setPottsModelId('');
      setSampleName('');
      setSamplingMethod('gibbs');
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
            States:{' '}
            {Array.isArray(selectedCluster.state_ids || selectedCluster.metastable_ids)
              ? (selectedCluster.state_ids || selectedCluster.metastable_ids).join(', ')
              : '-'}
          </p>
        )}
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Compare MD clusters (optional)</label>
        <div className="space-y-2">
          {compareOptions.length === 0 ? (
            <p className="text-xs text-gray-500">No additional clusters available for comparison.</p>
          ) : (
            compareOptions.map((run) => {
              const name = run.name || run.path?.split('/').pop() || run.cluster_id;
              const checked = compareClusterIds.includes(run.cluster_id);
              return (
                <label
                  key={run.cluster_id}
                  className="flex items-center gap-2 text-sm text-gray-200 bg-gray-800/40 border border-gray-700 rounded-md px-3 py-2"
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={busy}
                    onChange={() => {
                      setCompareClusterIds((prev) =>
                        prev.includes(run.cluster_id)
                          ? prev.filter((id) => id !== run.cluster_id)
                          : [...prev, run.cluster_id]
                      );
                    }}
                    className="h-4 w-4 rounded border-gray-500 bg-gray-900 text-cyan-500 focus:ring-cyan-500"
                  />
                  <span className="truncate">{name}</span>
                </label>
              );
            })
          )}
        </div>
        <p className="text-xs text-gray-500 mt-1">
          Pick additional cluster NPZs to compare their trajectories against the uploaded samples. The selected fit cluster is excluded.
        </p>
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Sampling method</label>
        <select
          value={samplingMethod}
          onChange={(event) => setSamplingMethod(event.target.value)}
          disabled={busy}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        >
          <option value="gibbs">Gibbs</option>
          <option value="sa">SA</option>
        </select>
        <p className="text-xs text-gray-500 mt-1">Choose the sampler used to generate this summary.</p>
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Sampling name (optional)</label>
        <input
          type="text"
          value={sampleName}
          onChange={(event) => setSampleName(event.target.value)}
          disabled={busy}
          placeholder="e.g., Gibbs β=1.0, 10k samples"
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white placeholder:text-gray-500 focus:ring-cyan-500"
        />
        <p className="text-xs text-gray-500 mt-1">Used in UI lists and legends.</p>
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
        <label className="block text-sm text-gray-300 mb-1">Potts model used for sampling</label>
        <select
          value={pottsModelId}
          onChange={(event) => setPottsModelId(event.target.value)}
          disabled={busy || !pottsModels.length}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
        >
          {pottsModels.length === 0 && <option value="">No Potts models available</option>}
          {pottsModels.map((model) => (
            <option key={model.model_id} value={model.model_id}>
              {model.name || model.model_id}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500 mt-1">Choose the Potts model that generated the uploaded samples.</p>
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
