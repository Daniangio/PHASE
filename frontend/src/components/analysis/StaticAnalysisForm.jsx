import { useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

export default function StaticAnalysisForm({ onSubmit }) {
  const [stateMetric, setStateMetric] = useState('auc');
  const [cvSplits, setCvSplits] = useState('');
  const [randomState, setRandomState] = useState('');
  const [idVarianceThreshold, setIdVarianceThreshold] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      const payload = { state_metric: stateMetric };
      if (cvSplits !== '') payload.cv_splits = Number(cvSplits);
      if (randomState !== '') payload.random_state = Number(randomState);
      if (idVarianceThreshold !== '') payload.id_variance_threshold = Number(idVarianceThreshold);
      await onSubmit(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-300 mb-1">State Metric</label>
          <select
            value={stateMetric}
            onChange={(e) => setStateMetric(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          >
            <option value="auc">AUC (logistic, default)</option>
            <option value="ce">Cross-entropy (information gain)</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            Matches backend static analysis options: AUC or cross-entropy surrogate.
          </p>
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">CV folds</label>
          <input
            type="number"
            min={2}
            placeholder="Default: 5"
            value={cvSplits}
            onChange={(e) => setCvSplits(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
          <p className="text-xs text-gray-500 mt-1">Optional. Effective folds are capped by class counts.</p>
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">Random state</label>
          <input
            type="number"
            placeholder="Default: 0"
            value={randomState}
            onChange={(e) => setRandomState(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
          <p className="text-xs text-gray-500 mt-1">Optional. Seeds fold splits and classifier.</p>
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">ID variance threshold</label>
          <input
            type="number"
            step="0.01"
            min={0.1}
            max={0.999}
            placeholder="Default: 0.90"
            value={idVarianceThreshold}
            onChange={(e) => setIdVarianceThreshold(e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
          <p className="text-xs text-gray-500 mt-1">
            Fraction of variance to keep for intrinsic-dimension estimate (0-1).
          </p>
        </div>
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
