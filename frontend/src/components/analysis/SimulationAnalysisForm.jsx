import { useEffect, useMemo, useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

export default function SimulationAnalysisForm({ clusterRuns, onSubmit }) {
  const [clusterId, setClusterId] = useState('');
  const [rexBetas, setRexBetas] = useState('');
  const [rexBetaMin, setRexBetaMin] = useState('');
  const [rexBetaMax, setRexBetaMax] = useState('');
  const [rexSpacing, setRexSpacing] = useState('geom');
  const [rexSamples, setRexSamples] = useState('');
  const [rexBurnin, setRexBurnin] = useState('');
  const [rexThin, setRexThin] = useState('');
  const [saReads, setSaReads] = useState('');
  const [saSweeps, setSaSweeps] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

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

  const parseBetaList = (raw) =>
    raw
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean)
      .map((item) => {
        const val = Number(item);
        if (!Number.isFinite(val)) {
          throw new Error(`Invalid beta value: "${item}"`);
        }
        return val;
      });

  const handleSubmit = async (event) => {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      if (!clusterId) {
        throw new Error('Select a saved cluster NPZ to run Potts analysis.');
      }
      const payload = { cluster_id: clusterId };
      const betasRaw = rexBetas.trim();
      if (betasRaw) {
        payload.rex_betas = parseBetaList(betasRaw);
      } else {
        if (rexBetaMin === '' || rexBetaMax === '') {
          throw new Error('Provide rex betas or both beta min and max.');
        }
        payload.rex_beta_min = Number(rexBetaMin);
        payload.rex_beta_max = Number(rexBetaMax);
        payload.rex_spacing = rexSpacing || 'geom';
      }

      if (rexSamples !== '') payload.rex_samples = Number(rexSamples);
      if (rexBurnin !== '') payload.rex_burnin = Number(rexBurnin);
      if (rexThin !== '') payload.rex_thin = Number(rexThin);
      if (saReads !== '') payload.sa_reads = Number(saReads);
      if (saSweeps !== '') payload.sa_sweeps = Number(saSweeps);

      await onSubmit(payload);
    } catch (err) {
      setError(err.message || 'Failed to submit simulation.');
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
          disabled={!clusterOptions.length}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
        >
          {clusterOptions.length === 0 && <option value="">No saved clusters</option>}
          {clusterOptions.map((run) => {
            const name = run.path?.split('/').pop() || run.cluster_id;
            return (
              <option key={run.cluster_id} value={run.cluster_id}>
                {name}
              </option>
            );
          })}
        </select>
        {selectedCluster && (
          <p className="text-xs text-gray-500 mt-1">
            Metastable: {Array.isArray(selectedCluster.metastable_ids) ? selectedCluster.metastable_ids.join(', ') : '—'} ·
            Contact {selectedCluster.contact_atom_mode || selectedCluster.contact_mode || 'CA'} @{' '}
            {selectedCluster.contact_cutoff ?? 10} Å
          </p>
        )}
      </div>

      <div className="border border-gray-700 rounded-md p-3 space-y-3">
        <div>
          <label className="block text-sm text-gray-300 mb-1">Explicit beta ladder (optional)</label>
          <input
            type="text"
            placeholder="0.2, 0.3, 0.5, 0.8, 1.0"
            value={rexBetas}
            onChange={(event) => setRexBetas(event.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
          <p className="text-xs text-gray-500 mt-1">If provided, overrides auto ladder settings.</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="block text-sm text-gray-300 mb-1">Beta min</label>
            <input
              type="number"
              step="0.01"
              placeholder="0.2"
              value={rexBetaMin}
              onChange={(event) => setRexBetaMin(event.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-1">Beta max</label>
            <input
              type="number"
              step="0.01"
              placeholder="1.0"
              value={rexBetaMax}
              onChange={(event) => setRexBetaMax(event.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-1">Spacing</label>
            <select
              value={rexSpacing}
              onChange={(event) => setRexSpacing(event.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
            >
              <option value="geom">Geometric</option>
              <option value="lin">Linear</option>
            </select>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div>
          <label className="block text-sm text-gray-300 mb-1">REX samples (rounds)</label>
          <input
            type="number"
            min={1}
            placeholder="Default: 2000"
            value={rexSamples}
            onChange={(event) => setRexSamples(event.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">REX burn-in</label>
          <input
            type="number"
            min={1}
            placeholder="Default: 50"
            value={rexBurnin}
            onChange={(event) => setRexBurnin(event.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">REX thin</label>
          <input
            type="number"
            min={1}
            placeholder="Default: 1"
            value={rexThin}
            onChange={(event) => setRexThin(event.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label className="block text-sm text-gray-300 mb-1">SA reads</label>
          <input
            type="number"
            min={1}
            placeholder="Default: 2000"
            value={saReads}
            onChange={(event) => setSaReads(event.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">SA sweeps</label>
          <input
            type="number"
            min={1}
            placeholder="Default: 2000"
            value={saSweeps}
            onChange={(event) => setSaSweeps(event.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
        </div>
      </div>

      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={isSubmitting || !clusterOptions.length}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isSubmitting ? 'Submitting…' : 'Run Potts Sampling'}
      </button>
    </form>
  );
}
