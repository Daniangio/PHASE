import { useEffect, useMemo, useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';
import { InfoTooltip } from '../system/SystemDetailWidgets';

const MODEL_PAGE_SIZE = 10;

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
  const [saChains, setSaChains] = useState('1');
  const [saScheduleMode, setSaScheduleMode] = useState('range');
  const [saScheduleType, setSaScheduleType] = useState('geometric');
  const [saCustomBetaSchedule, setSaCustomBetaSchedule] = useState('');
  const [saNumSweepsPerBeta, setSaNumSweepsPerBeta] = useState('2');
  const [saBetaHot, setSaBetaHot] = useState('0.01');
  const [saBetaCold, setSaBetaCold] = useState('2.0');
  const [saRandomizeOrder, setSaRandomizeOrder] = useState(true);
  const [saAcceptanceCriteria, setSaAcceptanceCriteria] = useState('Metropolis');
  const [saInit, setSaInit] = useState('md');
  const [saInitMdFrame, setSaInitMdFrame] = useState('');
  const [saRestart, setSaRestart] = useState('independent');
  const [saMdSampleId, setSaMdSampleId] = useState('');
  const [saMdStateIds, setSaMdStateIds] = useState('');
  const [penaltySafety, setPenaltySafety] = useState('4.0');
  const [repair, setRepair] = useState('none');
  const [pottsModelIds, setPottsModelIds] = useState([]);
  const [modelPage, setModelPage] = useState(0);
  const [samplingMethod, setSamplingMethod] = useState('gibbs');
  const [sampleName, setSampleName] = useState('');
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
    const nameById = new Map(
      models.map((model) => {
        const raw = model.name || (model.path ? model.path.split('/').pop() : '') || 'Potts model';
        return [model.model_id, raw.replace(/\.npz$/i, '')];
      })
    );
    return models.map((model) => {
      const rawLabel = model.name || (model.path ? model.path.split('/').pop() : '') || 'Potts model';
      const isDelta = !!model?.params?.delta_kind || model?.params?.fit_mode === 'delta';
      const baseModelId = model?.params?.base_model_id;
      const baseLabel = baseModelId ? nameById.get(baseModelId) : null;
      return {
        value: model.model_id,
        label: rawLabel.replace(/\.npz$/i, ''),
        isDelta,
        baseLabel,
      };
    });
  }, [selectedCluster]);
  const mdSampleOptions = useMemo(
    () => (selectedCluster?.samples || []).filter((sample) => String(sample?.type || '') === 'md_eval' && sample?.sample_id),
    [selectedCluster]
  );
  const modelPageCount = Math.max(1, Math.ceil(modelOptions.length / MODEL_PAGE_SIZE));
  const pagedModelOptions = useMemo(
    () => modelOptions.slice(modelPage * MODEL_PAGE_SIZE, (modelPage + 1) * MODEL_PAGE_SIZE),
    [modelOptions, modelPage]
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

  useEffect(() => {
    if (!modelOptions.length) {
      if (pottsModelIds.length) setPottsModelIds([]);
      return;
    }
    const allowed = new Set(modelOptions.map((opt) => opt.value));
    const filtered = pottsModelIds.filter((id) => allowed.has(id));
    if (filtered.length !== pottsModelIds.length) {
      setPottsModelIds(filtered);
      return;
    }
    if (!filtered.length) {
      setPottsModelIds([modelOptions[0].value]);
    }
  }, [modelOptions, pottsModelIds]);

  useEffect(() => {
    setModelPage(0);
  }, [clusterId]);

  useEffect(() => {
    if (modelPage >= modelPageCount) setModelPage(modelPageCount - 1);
  }, [modelPage, modelPageCount]);

  useEffect(() => {
    if (!mdSampleOptions.length) {
      setSaMdSampleId('');
      return;
    }
    if (!saMdSampleId || !mdSampleOptions.some((sample) => sample.sample_id === saMdSampleId)) {
      setSaMdSampleId(mdSampleOptions[0].sample_id || '');
    }
  }, [mdSampleOptions, saMdSampleId]);

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
      if (!pottsModelIds.length) {
        throw new Error('Select at least one fitted Potts model before sampling.');
      }
      const payload = { cluster_id: clusterId };
      payload.use_potts_model = true;
      payload.potts_model_ids = pottsModelIds;
      if (pottsModelIds.length === 1) payload.potts_model_id = pottsModelIds[0];
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
        if (!saMdSampleId) {
          throw new Error('Select an MD sample to use for SA warm-starts.');
        }
        payload.md_sample_id = saMdSampleId;
        if (saReads !== '') payload.sa_reads = Number(saReads);
        if (saChains !== '') payload.sa_chains = Number(saChains);
        if (saSweeps !== '') payload.sa_sweeps = Number(saSweeps);
        if (saNumSweepsPerBeta !== '') payload.sa_num_sweeps_per_beta = Number(saNumSweepsPerBeta);
        payload.sa_schedule_type = saScheduleMode === 'custom' ? 'custom' : saScheduleType;
        payload.sa_randomize_order = saRandomizeOrder;
        payload.sa_acceptance_criteria = saAcceptanceCriteria;
        if (saInit) payload.sa_init = saInit;
        if (saInit === 'md-frame') {
          if (saInitMdFrame === '') {
            throw new Error('Provide an MD frame index when using a fixed MD warm-start.');
          }
          const frameIndex = Number(saInitMdFrame);
          if (!Number.isInteger(frameIndex) || frameIndex < 0) {
            throw new Error('MD frame index must be a non-negative integer.');
          }
          payload.sa_init_md_frame = frameIndex;
        }
        if (saRestart) payload.sa_restart = saRestart;
        if (saMdStateIds.trim()) payload.sa_md_state_ids = saMdStateIds.trim();
        if (penaltySafety !== '') payload.penalty_safety = Number(penaltySafety);
        if (repair) payload.repair = repair;
        const sweepsPerBeta = Number(saNumSweepsPerBeta || '1');
        if (!Number.isInteger(sweepsPerBeta) || sweepsPerBeta < 1) {
          throw new Error('SA sweeps per beta must be a positive integer.');
        }
        if (saScheduleMode === 'custom') {
          const schedule = parseBetaList(saCustomBetaSchedule);
          if (!schedule.length) {
            throw new Error('Provide a custom beta schedule when SA schedule mode is custom.');
          }
          if (schedule.some((v) => v < 0)) {
            throw new Error('Custom SA beta schedule values must be >= 0.');
          }
          payload.sa_custom_beta_schedule = schedule;
        } else if (saScheduleMode === 'range') {
          const hotRaw = String(saBetaHot ?? '').trim();
          const coldRaw = String(saBetaCold ?? '').trim();
          if (!hotRaw || !coldRaw) {
            throw new Error('Provide both beta hot and beta cold for the SA range.');
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
          payload.sa_beta_hot = hot;
          payload.sa_beta_cold = cold;
        }
      }
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
        <label className="block text-sm text-gray-300 mb-1">Potts models</label>
        {!modelOptions.length && (
          <p className="text-xs text-gray-500">No fitted models available for this cluster.</p>
        )}
        {modelOptions.length > 0 && (
          <div className="space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-gray-500">
              <span>{modelOptions.length} models · {pottsModelIds.length} selected</span>
              <span>Page {modelPage + 1} of {modelPageCount}</span>
            </div>
            <div className="space-y-2 rounded-md border border-gray-700 bg-gray-950/30 p-2">
            {pagedModelOptions.map((opt) => {
              const checked = pottsModelIds.includes(opt.value);
              return (
                <label key={opt.value} className="flex items-center gap-2 text-xs text-gray-200">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() =>
                      setPottsModelIds((prev) =>
                        checked ? prev.filter((id) => id !== opt.value) : [...prev, opt.value]
                      )
                    }
                    className="h-4 w-4 text-cyan-500 rounded border-gray-600 bg-gray-900"
                  />
                  <div className="flex flex-col">
                    <span className="flex items-center gap-2">
                      <span>{opt.label}</span>
                      {opt.isDelta && (
                        <span className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold text-cyan-400 border border-cyan-500/40 bg-cyan-500/10">
                          Δ patch
                        </span>
                      )}
                    </span>
                    {opt.isDelta && opt.baseLabel && (
                      <span className="text-[10px] text-gray-500">base: {opt.baseLabel}</span>
                    )}
                  </div>
                </label>
              );
            })}
            </div>
            {modelPageCount > 1 && (
              <div className="flex items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setModelPage((page) => Math.max(0, page - 1))}
                  disabled={modelPage === 0}
                  className="rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800 disabled:opacity-40"
                >
                  Previous
                </button>
                <button
                  type="button"
                  onClick={() => setModelPage((page) => Math.min(modelPageCount - 1, page + 1))}
                  disabled={modelPage >= modelPageCount - 1}
                  className="rounded-md border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800 disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            )}
            <p className="text-xs text-gray-500">
              Select one or more models. Multiple selections will be combined (summed) for sampling.
            </p>
          </div>
        )}
      </div>

      <div>
        <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
          <span>Sampling method</span>
          <InfoTooltip
            ariaLabel="Sampling method help"
            text="Gibbs/REX targets the Potts Boltzmann distribution. SA/QUBO is a heuristic annealer on the one-hot QUBO and is better viewed as a low-energy search baseline, not as an exact Boltzmann sampler."
          />
        </label>
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
          <div>
            <label className="block text-sm text-gray-300 mb-1">SA warm-start MD sample</label>
            <select
              value={saMdSampleId}
              onChange={(event) => setSaMdSampleId(event.target.value)}
              disabled={!mdSampleOptions.length}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
            >
              {mdSampleOptions.length === 0 && <option value="">No md_eval samples on this cluster</option>}
              {mdSampleOptions.map((sample) => (
                <option key={sample.sample_id} value={sample.sample_id}>
                  {sample.name || sample.sample_id}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              SA no longer uses <code>cluster.npz</code> as its MD warm-start pool. It uses this explicit <code>md_eval</code> sample.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
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
              <label className="block text-sm text-gray-300 mb-1">SA chains</label>
              <input
                type="number"
                min={1}
                placeholder="Default: 1"
                value={saChains}
                onChange={(event) => setSaChains(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>SA sweeps</span>
                <InfoTooltip
                  ariaLabel="SA sweeps help"
                  text="Used in auto/range schedule modes. One sweep is one full update pass over all QUBO bits. neal spreads total sweeps across the whole anneal; it is not all spent at the cold end."
                />
              </label>
              <input
                type="number"
                min={1}
                placeholder="Default: 2000"
                value={saSweeps}
                onChange={(event) => setSaSweeps(event.target.value)}
                disabled={saScheduleMode === 'custom'}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
              />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>Schedule mode</span>
                <InfoTooltip
                  ariaLabel="SA schedule mode help"
                  text="auto: let neal choose the beta range. range: you set beta_hot and beta_cold and choose how to interpolate. custom: give the exact beta list to sweep."
                />
              </label>
              <select
                value={saScheduleMode}
                onChange={(event) => setSaScheduleMode(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              >
                <option value="auto">Auto range</option>
                <option value="range">Explicit range</option>
                <option value="custom">Custom schedule</option>
              </select>
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>Schedule type</span>
                <InfoTooltip
                  ariaLabel="SA schedule type help"
                  text="How beta values are interpolated for auto/range modes. Geometric is usually better when the useful beta scale spans orders of magnitude."
                />
              </label>
              <select
                value={saScheduleType}
                onChange={(event) => setSaScheduleType(event.target.value)}
                disabled={saScheduleMode === 'custom'}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
              >
                <option value="geometric">Geometric</option>
                <option value="linear">Linear</option>
              </select>
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>Sweeps / beta</span>
                <InfoTooltip
                ariaLabel="SA sweeps per beta help"
                  text="Number of update sweeps spent at each beta value in the annealing schedule. In custom mode, total sweeps are approximately len(beta_schedule) × sweeps_per_beta."
                />
              </label>
              <input
                type="number"
                min={1}
                placeholder="Default: 2"
                value={saNumSweepsPerBeta}
                onChange={(event) => setSaNumSweepsPerBeta(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
          </div>
          {saScheduleMode === 'range' && (
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm text-gray-300">
                <span>SA beta range</span>
                <InfoTooltip
                  ariaLabel="SA beta range help"
                  text="Anneal from beta_hot (hotter, more random) to beta_cold (colder, greedier). A very hot start can erase the benefit of an MD warm-start."
                />
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <input
                  type="number"
                  step="0.01"
                  placeholder="beta_hot (default 0.01)"
                  value={saBetaHot}
                  onChange={(event) => setSaBetaHot(event.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                />
                <input
                  type="number"
                  step="0.01"
                  placeholder="beta_cold (default 2.0)"
                  value={saBetaCold}
                  onChange={(event) => setSaBetaCold(event.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                />
              </div>
            </div>
          )}
          {saScheduleMode === 'custom' && (
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>Custom beta schedule</span>
                <InfoTooltip
                  ariaLabel="Custom SA beta schedule help"
                  text="Exact beta values to sweep, comma-separated. Example: 0.6,0.8,1.0,1.5,2.5,4,7,10. In this mode total sweeps are determined by the schedule length and sweeps-per-beta."
                />
              </label>
              <input
                type="text"
                placeholder="0.6,0.8,1.0,1.5,2.5,4,7,10"
                value={saCustomBetaSchedule}
                onChange={(event) => setSaCustomBetaSchedule(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>SA init</span>
                <InfoTooltip
                  ariaLabel="SA init help"
                  text="Defines the starting state of each independent SA read or the first sample of each correlated SA chain. md = random MD frame; md-frame = the chosen fixed MD frame; random-h = independent draw from exp(-beta h_i) using only fields; random-uniform = fully random labels."
                />
              </label>
              <select
                value={saInit}
                onChange={(event) => setSaInit(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              >
                <option value="md">MD warm-start (random frame)</option>
                <option value="md-frame">MD warm-start (fixed frame)</option>
                <option value="random-h">Random from h(s) (independent)</option>
                <option value="random-uniform">Random uniform</option>
              </select>
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>MD frame index</span>
                <InfoTooltip
                  ariaLabel="Fixed MD frame help"
                  text="Used only when SA init = md-frame. In chain mode, this fixed frame seeds only the first sample unless you choose independent reads."
                />
              </label>
              <input
                type="number"
                min={0}
                placeholder="Only for fixed MD init"
                value={saInitMdFrame}
                onChange={(event) => setSaInitMdFrame(event.target.value)}
                disabled={saInit !== 'md-frame'}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500 disabled:opacity-60"
              />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>SA restart</span>
                <InfoTooltip
                  ariaLabel="SA restart help"
                  text="previous: each new sample starts from the previous decoded sample, but SA reheats again from beta_hot to beta_cold. md: each new sample restarts from a fresh random MD frame. independent: no chain carry-over; every read uses SA init independently. Independent is the default because it matches the older SA behavior more closely and is less drift-prone."
                />
              </label>
              <select
                value={saRestart}
                onChange={(event) => setSaRestart(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              >
                <option value="previous">Continue chain</option>
                <option value="md">Restart from random MD each sample</option>
                <option value="independent">Independent (default)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-300 mb-1">MD state filter (optional)</label>
              <input
                type="text"
                placeholder="state_a,state_b"
                value={saMdStateIds}
                onChange={(event) => setSaMdStateIds(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>Acceptance rule</span>
                <InfoTooltip
                  ariaLabel="SA acceptance rule help"
                  text="Metropolis is the usual classical SA rule. Gibbs accepts proposals using Gibbs criteria; it can change annealing behavior and should be treated as an advanced option."
                />
              </label>
              <select
                value={saAcceptanceCriteria}
                onChange={(event) => setSaAcceptanceCriteria(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              >
                <option value="Metropolis">Metropolis</option>
                <option value="Gibbs">Gibbs</option>
              </select>
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>Update order</span>
                <InfoTooltip
                  ariaLabel="SA update order help"
                  text="Sequential order is faster. Randomized order is more symmetric and can reduce update-order bias, but changes the annealing dynamics."
                />
              </label>
              <label className="inline-flex items-center gap-2 text-sm text-gray-300 mt-2">
                <input
                  type="checkbox"
                  checked={saRandomizeOrder}
                  onChange={(event) => setSaRandomizeOrder(event.target.checked)}
                />
                Randomize variable order within each sweep
              </label>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>Penalty safety</span>
                <InfoTooltip
                  ariaLabel="Penalty safety help"
                  text="Scales the one-hot QUBO penalty. Larger values enforce validity more strongly but also raise barriers between categorical states. The exploratory default is 4.0; monitor invalid-read and movement diagnostics."
                />
              </label>
              <input
                type="number"
                min={0.1}
                step="0.1"
                value={penaltySafety}
                onChange={(event) => setPenaltySafety(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="flex items-center gap-2 text-sm text-gray-300 mb-1">
                <span>Decode repair</span>
                <InfoTooltip
                  ariaLabel="Decode repair help"
                  text="none keeps invalid one-hot samples flagged as invalid. argmax projects each residue slice to one label after sampling. This affects saved decoded labels, not the underlying QUBO anneal."
                />
              </label>
              <select
                value={repair}
                onChange={(event) => setRepair(event.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
              >
                <option value="none">none</option>
                <option value="argmax">argmax</option>
              </select>
            </div>
          </div>
          <div className="rounded-md border border-gray-800 bg-gray-950/40 px-3 py-2 text-xs text-gray-400">
            SA/QUBO here is a heuristic annealer. It is useful for low-energy search, but unlike Gibbs/REX it does not target the Potts Boltzmann distribution exactly. For Boltzmann-like sampling, use Gibbs or REX.
          </div>
        </>
      )}

      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={isSubmitting || !clusterOptions.length || pottsModelIds.length === 0}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isSubmitting ? 'Submitting…' : 'Run Potts Sampling'}
      </button>
    </form>
  );
}
