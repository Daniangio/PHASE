import { useState } from 'react';
import { Download, Info, Trash2 } from 'lucide-react';
import { resultArtifactUrl } from '../../api/jobs';
import { getArtifactDisplayName } from './systemDetailUtils';
import IconButton from '../common/IconButton';

export function InfoTooltip({ text, ariaLabel, onClick }) {
  return (
    <span className="relative inline-flex group">
      <span
        role="button"
        tabIndex={0}
        className="inline-flex items-center justify-center text-gray-500 hover:text-gray-300 focus:outline-none"
        aria-label={ariaLabel}
        aria-haspopup={onClick ? 'dialog' : undefined}
        onClick={onClick}
        onKeyDown={(e) => {
          if (!onClick) return;
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onClick(e);
          }
        }}
      >
        <Info className="h-4 w-4" />
      </span>
      <span className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 w-72 -translate-x-1/2 rounded-md border border-gray-700 bg-gray-900 px-3 py-2 text-xs text-gray-200 opacity-0 shadow-lg transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
        {text}
      </span>
    </span>
  );
}

export function StatusBadge({ status }) {
  let className = 'bg-amber-400';
  if (status === 'finished') className = 'bg-emerald-400';
  if (status === 'failed') className = 'bg-red-400';
  return <span className={`inline-block h-2 w-2 rounded-full ${className}`} />;
}

export function AnalysisResultsList({ results, emptyLabel, onOpen, onOpenSimulation, onDelete }) {
  if (!results.length) {
    return <p className="text-xs text-gray-500">{emptyLabel}</p>;
  }
  return (
    <div className="space-y-2">
      {results.map((result) => (
        <div
          key={result.job_id}
          className="flex items-center justify-between gap-3 rounded-md border border-gray-800 bg-gray-950/40 px-3 py-2"
        >
          <div className="flex items-center gap-3">
            <StatusBadge status={result.status} />
            <div>
              <p className="text-sm text-gray-200">
                {result.analysis_type === 'simulation' ? 'POTTS SAMPLING' : result.analysis_type.toUpperCase()}
              </p>
              <p className="text-xs text-gray-500">
                {result.analysis_type === 'simulation'
                  ? result.sample_name || result.potts_model_name || 'Sampling run'
                  : `Job ${result.job_id?.slice(0, 8)}`}{' '}
                · {result.status}
                {result.created_at ? ` · ${new Date(result.created_at).toLocaleString()}` : ''}
              </p>
              {result.cluster_label && (
                <p className="text-xs text-gray-500">Cluster: {result.cluster_label}</p>
              )}
              {result.analysis_type === 'simulation' && (
                <>
                  {result.potts_model_name && (
                    <p className="text-xs text-gray-500">Potts model: {result.potts_model_name}</p>
                  )}
                  {result.cluster_npz && (
                    <p className="text-xs text-gray-500">Cluster NPZ: {getArtifactDisplayName(result.cluster_npz)}</p>
                  )}
                </>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {onOpenSimulation && result.analysis_type === 'simulation' && result.status === 'finished' && (
              <button
                type="button"
                onClick={() => onOpenSimulation(result)}
                className="text-xs text-cyan-300 hover:text-cyan-200"
              >
                Sampling viz
              </button>
            )}
            {result.analysis_type === 'simulation' && result.status === 'finished' && (
              <a
                href={resultArtifactUrl(result.job_id, 'summary_npz')}
                download={`${result.job_id}-samples.npz`}
                className="text-xs text-cyan-300 hover:text-cyan-200 flex items-center gap-1"
              >
                <Download className="h-3 w-3" />
                Samples
              </a>
            )}
            <button
              type="button"
              onClick={() => onOpen(result)}
              className="text-xs text-cyan-300 hover:text-cyan-200"
            >
              Details
            </button>
            {onDelete && (
              <IconButton
                icon={Trash2}
                label="Delete result"
                onClick={() => onDelete(result)}
                className="text-gray-400 hover:text-red-400"
                iconClassName="h-4 w-4"
              />
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export function StatePairSelector({ options, value, onChange }) {
  if (options.length < 2) {
    return <p className="text-sm text-gray-400">Upload trajectories for at least two states to run analyses.</p>;
  }

  const toValue = (opt) => (opt ? `${opt.kind}:${opt.id}` : '');

  const handleChange = (key, raw) => {
    const next = options.find((o) => `${o.kind}:${o.id}` === raw) || null;
    onChange((prev) => ({ ...prev, [key]: next }));
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
      <p className="text-sm text-gray-200 mb-2">Select two states or metastable states to compare</p>
      <div className="grid md:grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-gray-400 mb-1">State A</label>
          <select
            value={toValue(value.a)}
            onChange={(e) => handleChange('a', e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          >
            <option value="">Choose</option>
            {options.map((opt) => (
              <option key={`${opt.kind}-${opt.id}`} value={`${opt.kind}:${opt.id}`}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-400 mb-1">State B</label>
          <select
            value={toValue(value.b)}
            onChange={(e) => handleChange('b', e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          >
            <option value="">Choose</option>
            {options.map((opt) => (
              <option key={`${opt.kind}-${opt.id}`} value={`${opt.kind}:${opt.id}`}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}

export function AddStateForm({ states, onAdd, isAdding }) {
  const [name, setName] = useState('');
  const [file, setFile] = useState(null);
  const [copyFrom, setCopyFrom] = useState('');
  const [residShift, setResidShift] = useState('0');
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!name.trim()) return;
    if (!file && !copyFrom) return;
    await onAdd({ name: name.trim(), file, copyFrom, residShift });
    setName('');
    setFile(null);
    setCopyFrom('');
    setResidShift('0');
  };
  return (
    <form onSubmit={handleSubmit} className="bg-gray-900 border border-gray-700 rounded-lg p-4 space-y-3">
      <h3 className="text-md font-semibold text-white">Add State</h3>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          required
        />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Upload PDB</label>
        <input
          type="file"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="w-full text-sm text-gray-300"
        />
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Or copy existing PDB</label>
        <select
          value={copyFrom}
          onChange={(e) => setCopyFrom(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        >
          <option value="">—</option>
          {states.map((state) => (
            <option key={state.state_id} value={state.state_id}>
              {state.name}
            </option>
          ))}
        </select>
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Residue shift offset</label>
        <input
          type="number"
          step="1"
          value={residShift}
          onChange={(e) => setResidShift(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        />
      </div>
      <p className="text-xs text-gray-500">
        Provide a new PDB or select an existing state to duplicate its structure. Residue shift is applied when descriptors are built.
      </p>
      <button
        type="submit"
        disabled={isAdding || (!file && !copyFrom)}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isAdding ? 'Adding…' : 'Add State'}
      </button>
    </form>
  );
}
