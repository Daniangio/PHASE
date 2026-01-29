import { useEffect, useMemo, useState } from 'react';
import { Info, SlidersHorizontal } from 'lucide-react';
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
  const [saSchedules, setSaSchedules] = useState([]);
  const [pottsModelId, setPottsModelId] = useState('');
  const [samplingMethod, setSamplingMethod] = useState('gibbs');
  const [sampleName, setSampleName] = useState('');
  const [plmEpochs, setPlmEpochs] = useState('');
  const [plmLr, setPlmLr] = useState('');
  const [plmLrMin, setPlmLrMin] = useState('');
  const [plmLrSchedule, setPlmLrSchedule] = useState('cosine');
  const [plmL2, setPlmL2] = useState('');
  const [plmBatchSize, setPlmBatchSize] = useState('');
  const [plmProgressEvery, setPlmProgressEvery] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const clusterOptions = useMemo(() => clusterRuns || [], [clusterRuns]);
  const selectedCluster = useMemo(
    () => clusterOptions.find((run) => run.cluster_id === clusterId),
    [clusterOptions, clusterId]
  );
  const modelOptions = useMemo(() => {
    if (!selectedCluster) return [];
    const models = selectedCluster.potts_models || [];
    return models.map((model) => {
      const rawLabel = model.name || (model.path ? model.path.split('/').pop() : '') || 'Potts model';
      return {
        value: model.model_id,
        label: rawLabel.replace(/\.npz$/i, ''),
      };
    });
  }, [selectedCluster]);

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
    if (!modelOptions.length) {
      setPottsModelId('');
      return;
    }
    if (!pottsModelId || !modelOptions.some((opt) => opt.value === pottsModelId)) {
      setPottsModelId(modelOptions[0].value);
    }
  }, [modelOptions, pottsModelId]);

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
      if (!pottsModelId) {
        throw new Error('Select a fitted Potts model before sampling.');
      }
      const payload = { cluster_id: clusterId };
      payload.use_potts_model = true;
      payload.potts_model_id = pottsModelId;
      payload.sampling_method = samplingMethod;
      if (sampleName.trim()) payload.sample_name = sampleName.trim();

      if (samplingMethod === 'gibbs') {
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
      } else {
        if (saReads !== '') payload.sa_reads = Number(saReads);
        if (saSweeps !== '') payload.sa_sweeps = Number(saSweeps);
        const customSchedules = [];
        for (const schedule of saSchedules) {
          const hotRaw = String(schedule.betaHot ?? '').trim();
          const coldRaw = String(schedule.betaCold ?? '').trim();
          if (!hotRaw && !coldRaw) {
            continue;
          }
          if (!hotRaw || !coldRaw) {
            throw new Error('Provide both beta hot and beta cold for each SA schedule.');
          }
          const hot = Number(hotRaw);
          const cold = Number(coldRaw);
          if (!Number.isFinite(hot) || !Number.isFinite(cold)) {
            throw new Error('SA beta values must be numeric.');
          }
          if (hot <= 0 || cold <= 0) {
            throw new Error('SA beta values must be > 0.');
          }
          if (hot > cold) {
            throw new Error('SA beta hot must be <= SA beta cold.');
          }
          customSchedules.push([hot, cold]);
        }
        if (customSchedules.length) payload.sa_beta_schedules = customSchedules;
      }
      if (plmEpochs !== '') payload.plm_epochs = Number(plmEpochs);
      if (plmLr !== '') payload.plm_lr = Number(plmLr);
      if (plmLrMin !== '') payload.plm_lr_min = Number(plmLrMin);
      if (plmLrSchedule) payload.plm_lr_schedule = plmLrSchedule;
      if (plmL2 !== '') payload.plm_l2 = Number(plmL2);
      if (plmBatchSize !== '') payload.plm_batch_size = Number(plmBatchSize);
      if (plmProgressEvery !== '') payload.plm_progress_every = Number(plmProgressEvery);

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
              : '—'}{' '}
            ·
          </p>
        )}
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Potts model</label>
        <select
          value={pottsModelId}
          onChange={(event) => setPottsModelId(event.target.value)}
          disabled={!modelOptions.length}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
        >
          {!modelOptions.length && <option value="">No fitted models available</option>}
          {modelOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-gray-500 mt-1">
          Select a pre-fit model (webserver or uploaded) to run sampling.
        </p>
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Sampling method</label>
        <select
          value={samplingMethod}
          onChange={(event) => setSamplingMethod(event.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        >
          <option value="gibbs">Gibbs (single/REX)</option>
          <option value="sa">Simulated Annealing (SA)</option>
        </select>
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Sampling name (optional)</label>
        <input
          type="text"
          value={sampleName}
          onChange={(event) => setSampleName(event.target.value)}
          placeholder="e.g., Gibbs β=1.0, 10k samples"
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white placeholder:text-gray-500 focus:ring-cyan-500"
        />
      </div>

      {samplingMethod === 'gibbs' && (
        <>
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
        </>
      )}

      {samplingMethod === 'sa' && (
        <>
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
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 text-sm text-gray-300">
                SA beta schedules
                <span className="relative inline-flex group">
                  <button
                    type="button"
                    className="inline-flex items-center justify-center text-gray-500 hover:text-gray-300 focus:outline-none"
                    aria-label="SA beta schedules help"
                  >
                    <Info className="h-4 w-4" />
                  </button>
                  <span className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 w-72 -translate-x-1/2 rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-xs text-gray-200 opacity-0 shadow-lg transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
                    Auto schedule is always included. Add extra schedules as beta_hot (fast mixing) to beta_cold (low excitation). Reasonable ranges: beta_hot ~0.1–0.5, beta_cold ~2–10.
                  </span>
                </span>
              </label>
              <button
                type="button"
                onClick={() => setSaSchedules((prev) => [...prev, { betaHot: '', betaCold: '' }])}
                className="text-xs text-cyan-300 hover:text-cyan-200"
              >
                Add schedule
              </button>
            </div>

            <div className="rounded-md border border-dashed border-gray-700 bg-gray-900/30 px-3 py-2 text-xs text-gray-400">
              Auto schedule (neal default beta range)
            </div>

            {saSchedules.length > 0 && (
              <div className="space-y-2">
                {saSchedules.map((schedule, index) => (
                  <div key={`sa-schedule-${index}`} className="grid grid-cols-1 md:grid-cols-3 gap-2">
                    <input
                      type="number"
                      step="0.01"
                      placeholder="beta_hot (e.g. 0.2)"
                      value={schedule.betaHot}
                      onChange={(event) =>
                        setSaSchedules((prev) =>
                          prev.map((item, i) => (i === index ? { ...item, betaHot: event.target.value } : item))
                        )
                      }
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                    />
                    <input
                      type="number"
                      step="0.01"
                      placeholder="beta_cold (e.g. 5.0)"
                      value={schedule.betaCold}
                      onChange={(event) =>
                        setSaSchedules((prev) =>
                          prev.map((item, i) => (i === index ? { ...item, betaCold: event.target.value } : item))
                        )
                      }
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                    />
                    <button
                      type="button"
                      onClick={() => setSaSchedules((prev) => prev.filter((_, i) => i !== index))}
                      className="text-xs text-gray-400 hover:text-red-300"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      <button
        type="button"
        onClick={() => setShowAdvanced((prev) => !prev)}
        className="flex items-center gap-2 text-xs text-cyan-300 hover:text-cyan-200"
      >
        <SlidersHorizontal className="h-4 w-4" />
        {showAdvanced ? 'Hide' : 'Show'} advanced Potts settings
      </button>

      {showAdvanced && (
        <div className="border border-gray-700 rounded-md p-3 space-y-3">
          <div>
            <h4 className="text-sm font-semibold text-white">PLM hyperparameters</h4>
            <p className="text-xs text-gray-500 mt-1">
              Defaults match the script. Leave blank to keep defaults.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm text-gray-300 mb-1">Epochs</label>
              <input
                type="number"
                min={1}
                placeholder="Default: 200"
                value={plmEpochs}
                onChange={(event) => setPlmEpochs(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Batch size</label>
              <input
                type="number"
                min={1}
                placeholder="Default: 512"
                value={plmBatchSize}
                onChange={(event) => setPlmBatchSize(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Learning rate</label>
              <input
                type="number"
                step="0.0001"
                placeholder="Default: 1e-2"
                value={plmLr}
                onChange={(event) => setPlmLr(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">LR schedule</label>
              <select
                value={plmLrSchedule}
                onChange={(event) => setPlmLrSchedule(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              >
                <option value="cosine">Cosine decay</option>
                <option value="none">Fixed LR</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Min LR (cosine)</label>
              <input
                type="number"
                step="0.0001"
                placeholder="Default: 1e-3"
                value={plmLrMin}
                onChange={(event) => setPlmLrMin(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">L2 weight decay</label>
              <input
                type="number"
                step="0.0001"
                placeholder="Default: 1e-5"
                value={plmL2}
                onChange={(event) => setPlmL2(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">Progress every N epochs</label>
              <input
                type="number"
                min={1}
                placeholder="Default: 10"
                value={plmProgressEvery}
                onChange={(event) => setPlmProgressEvery(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
          </div>
        </div>
      )}

      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={isSubmitting || !clusterOptions.length || !pottsModelId}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isSubmitting ? 'Submitting…' : 'Run Potts Sampling'}
      </button>
    </form>
  );
}
