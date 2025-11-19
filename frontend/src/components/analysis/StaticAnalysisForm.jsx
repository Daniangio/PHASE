import { useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

const metrics = [
  { value: 'auc', label: 'Logistic AUC' },
  { value: 'mi', label: 'Mutual Information' },
  { value: 'jsd', label: 'Jensen-Shannon' },
  { value: 'mmd', label: 'Maximum Mean Discrepancy' },
  { value: 'kl', label: 'Symmetrized KL' },
];

export default function StaticAnalysisForm({ onSubmit }) {
  const [stateMetric, setStateMetric] = useState('auc');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      await onSubmit({ state_metric: stateMetric });
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm text-gray-300 mb-1">State Metric</label>
        <select
          value={stateMetric}
          onChange={(e) => setStateMetric(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        >
          {metrics.map((metric) => (
            <option key={metric.value} value={metric.value}>
              {metric.label}
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
        {isSubmitting ? 'Submitting…' : 'Run Static Analysis'}
      </button>
    </form>
  );
}
