import { useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

const defaultParams = {
  alpha_size: 1.0,
  beta_hub: 2.0,
  beta_switch: 5.0,
  gamma_redundancy: 3.0,
  ii_threshold: 0.4,
  filter_top_total: 100,
  filter_top_jsd: 20,
  filter_min_id: 1.5,
  static_job_uuid: '',
};

export default function QuboAnalysisForm({ onSubmit, staticOptions = [] }) {
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
      await onSubmit({
        ...form,
        alpha_size: Number(form.alpha_size),
        beta_hub: Number(form.beta_hub),
        beta_switch: Number(form.beta_switch),
        gamma_redundancy: Number(form.gamma_redundancy),
        ii_threshold: Number(form.ii_threshold),
        filter_top_total: Number(form.filter_top_total),
        filter_top_jsd: Number(form.filter_top_jsd),
        filter_min_id: Number(form.filter_min_id),
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <NumberInput label="Alpha (size penalty)" value={form.alpha_size} onChange={(v) => handleChange('alpha_size', v)} />
        <NumberInput label="Beta (hub reward)" value={form.beta_hub} onChange={(v) => handleChange('beta_hub', v)} />
        <NumberInput label="Beta (switch reward)" value={form.beta_switch} onChange={(v) => handleChange('beta_switch', v)} />
        <NumberInput label="Gamma (redundancy)" value={form.gamma_redundancy} onChange={(v) => handleChange('gamma_redundancy', v)} />
        <NumberInput label="II threshold" value={form.ii_threshold} step="0.1" onChange={(v) => handleChange('ii_threshold', v)} />
        <NumberInput label="Top candidates" value={form.filter_top_total} onChange={(v) => handleChange('filter_top_total', v)} />
        <NumberInput label="Guaranteed JSD" value={form.filter_top_jsd} onChange={(v) => handleChange('filter_top_jsd', v)} />
        <NumberInput label="Min Intrinsic Dimension" value={form.filter_min_id} step="0.1" onChange={(v) => handleChange('filter_min_id', v)} />
      </div>
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

function NumberInput({ label, value, onChange, step = 'any' }) {
  return (
    <div>
      <label className="block text-sm text-gray-300 mb-1">{label}</label>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
      />
    </div>
  );
}
