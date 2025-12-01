import { useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

const defaultParams = {
  alpha_size: 1.0,
  beta_hub: 1.0,
  beta_switch: 5.0,
  gamma_redundancy: 2.0,
  ii_threshold: 0.9,
  ii_scale: 0.6,
  soft_threshold_power: 2.0,
  filter_top_total: 120,
  filter_top_jsd: 20,
  filter_min_id: 1.5,
  num_reads: 2000,
  num_solutions: 5,
  taxonomy_switch_high: 0.8,
  taxonomy_switch_low: 0.3,
  taxonomy_hub_high_percentile: 80,
  taxonomy_hub_low_percentile: 50,
  taxonomy_delta_hub_high_percentile: 80,
  maxk: '',
  seed: '',
  static_job_uuid: '',
};

export default function QuboAnalysisForm({ onSubmit, staticOptions = [], cachePaths = { active: [], inactive: [] } }) {
  const [form, setForm] = useState(defaultParams);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      const toNumber = (value) => {
        if (value === '' || value === null || value === undefined) return undefined;
        const parsed = Number(value);
        return Number.isNaN(parsed) ? undefined : parsed;
      };

      const numericFields = [
        'alpha_size',
        'beta_hub',
        'beta_switch',
        'gamma_redundancy',
        'ii_threshold',
        'ii_scale',
        'soft_threshold_power',
        'filter_top_total',
        'filter_top_jsd',
        'filter_min_id',
        'num_reads',
        'num_solutions',
        'taxonomy_switch_high',
        'taxonomy_switch_low',
        'taxonomy_hub_high_percentile',
        'taxonomy_hub_low_percentile',
        'taxonomy_delta_hub_high_percentile',
        'maxk',
        'seed',
      ];

      const payload = {};
      numericFields.forEach((field) => {
        const val = toNumber(form[field]);
        if (val !== undefined) {
          payload[field] = val;
        }
      });

      if (form.static_job_uuid) {
        payload.static_job_uuid = form.static_job_uuid;
      }

      if (cachePaths?.active?.length) {
        payload.imbalance_matrix_paths_active = cachePaths.active;
      }
      if (cachePaths?.inactive?.length) {
        payload.imbalance_matrix_paths_inactive = cachePaths.inactive;
      }

      await onSubmit(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <section className="space-y-3">
        <h4 className="text-sm font-semibold text-gray-200">Core weights</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <NumberInput label="Alpha (size penalty)" value={form.alpha_size} step="0.1" onChange={(v) => handleChange('alpha_size', v)} />
          <NumberInput label="Beta (hub reward)" value={form.beta_hub} step="0.1" onChange={(v) => handleChange('beta_hub', v)} />
          <NumberInput label="Beta (switch reward)" value={form.beta_switch} step="0.1" onChange={(v) => handleChange('beta_switch', v)} />
          <NumberInput label="Gamma (redundancy penalty)" value={form.gamma_redundancy} step="0.1" onChange={(v) => handleChange('gamma_redundancy', v)} />
        </div>
      </section>

      <section className="space-y-3">
        <h4 className="text-sm font-semibold text-gray-200">Coverage & imbalance</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <NumberInput label="II threshold" value={form.ii_threshold} step="0.05" onChange={(v) => handleChange('ii_threshold', v)} />
          <NumberInput label="II scale (coverage cutoff)" value={form.ii_scale} step="0.05" onChange={(v) => handleChange('ii_scale', v)} />
          <NumberInput label="Soft threshold power" value={form.soft_threshold_power} step="0.05" onChange={(v) => handleChange('soft_threshold_power', v)} />
          <NumberInput
            label="k-NN neighborhood (maxk)"
            value={form.maxk}
            min={2}
            onChange={(v) => handleChange('maxk', v)}
            helper="Optional. Leave blank to auto-set."
          />
        </div>
      </section>

      <section className="space-y-3">
        <h4 className="text-sm font-semibold text-gray-200">Candidate filtering</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <NumberInput label="Top candidates" value={form.filter_top_total} step="1" onChange={(v) => handleChange('filter_top_total', v)} />
          <NumberInput label="Guaranteed JSD" value={form.filter_top_jsd} step="1" onChange={(v) => handleChange('filter_top_jsd', v)} />
          <NumberInput label="Min intrinsic dimension" value={form.filter_min_id} step="0.1" onChange={(v) => handleChange('filter_min_id', v)} />
        </div>
      </section>

      <section className="space-y-3">
        <h4 className="text-sm font-semibold text-gray-200">Sampler</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <NumberInput label="Sampler reads" value={form.num_reads} step="1" onChange={(v) => handleChange('num_reads', v)} />
          <NumberInput label="Solutions to keep" value={form.num_solutions} step="1" onChange={(v) => handleChange('num_solutions', v)} />
          <NumberInput
            label="RNG seed"
            value={form.seed}
            step="1"
            onChange={(v) => handleChange('seed', v)}
            helper="Optional. Blank = random seed."
          />
        </div>
      </section>

      <section className="space-y-3">
        <h4 className="text-sm font-semibold text-gray-200">Taxonomy thresholds</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <NumberInput label="Switch score high" value={form.taxonomy_switch_high} step="0.05" onChange={(v) => handleChange('taxonomy_switch_high', v)} />
          <NumberInput label="Switch score low" value={form.taxonomy_switch_low} step="0.05" onChange={(v) => handleChange('taxonomy_switch_low', v)} />
          <NumberInput label="Hub high percentile" value={form.taxonomy_hub_high_percentile} step="1" onChange={(v) => handleChange('taxonomy_hub_high_percentile', v)} />
          <NumberInput label="Hub low percentile" value={form.taxonomy_hub_low_percentile} step="1" onChange={(v) => handleChange('taxonomy_hub_low_percentile', v)} />
          <NumberInput
            label="Delta hub high percentile"
            value={form.taxonomy_delta_hub_high_percentile}
            step="1"
            onChange={(v) => handleChange('taxonomy_delta_hub_high_percentile', v)}
          />
        </div>
      </section>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Static Result (optional)</label>
        <select
          value={form.static_job_uuid}
          onChange={(e) => handleChange('static_job_uuid', e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        >
          <option value="">-- None (run pre-filter now) --</option>
          {staticOptions.map((result) => (
            <option key={result.job_id} value={result.job_id}>
              {new Date(result.created_at).toLocaleString()} • {result.job_id.slice(0, 8)}
            </option>
          ))}
        </select>
      </div>
      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isSubmitting ? 'Submitting…' : 'Run QUBO Analysis'}
      </button>
    </form>
  );
}

function NumberInput({ label, value, onChange, step = 'any', min, helper }) {
  return (
    <div>
      <label className="block text-sm text-gray-300 mb-1">{label}</label>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
      />
      {helper ? <p className="text-xs text-gray-500 mt-1">{helper}</p> : null}
    </div>
  );
}
