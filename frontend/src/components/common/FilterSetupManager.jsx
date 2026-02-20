import ErrorMessage from './ErrorMessage';

export default function FilterSetupManager({
  setups = [],
  selectedSetupId = '',
  onSelectedSetupIdChange,
  newSetupName = '',
  onNewSetupNameChange,
  onLoad,
  onSave,
  onDelete,
  error = '',
  className = '',
}) {
  return (
    <div className={`rounded-md border border-gray-800 bg-gray-950/30 p-3 space-y-2 ${className}`}>
      <div className="text-xs text-gray-300">Filter setup files</div>
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_auto] gap-2">
        <select
          value={selectedSetupId}
          onChange={(e) => onSelectedSetupIdChange?.(e.target.value)}
          className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
        >
          <option value="">Select saved setup...</option>
          {setups.map((s) => (
            <option key={s.setup_id} value={s.setup_id}>
              {s.name || s.setup_id}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={onLoad}
          disabled={!selectedSetupId}
          className="px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:border-gray-500 disabled:opacity-50"
        >
          Load
        </button>
        <button
          type="button"
          onClick={onDelete}
          disabled={!selectedSetupId}
          className="px-3 py-2 rounded-md border border-red-800 text-sm text-red-300 hover:border-red-600 disabled:opacity-50"
        >
          Delete
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-2">
        <input
          type="text"
          value={newSetupName}
          onChange={(e) => onNewSetupNameChange?.(e.target.value)}
          placeholder="New setup name"
          className="w-full bg-gray-950 border border-gray-800 rounded-md px-2 py-2 text-sm text-gray-100"
        />
        <button
          type="button"
          onClick={onSave}
          disabled={!String(newSetupName || '').trim()}
          className="px-3 py-2 rounded-md border border-cyan-700 text-sm text-cyan-200 hover:border-cyan-500 disabled:opacity-50"
        >
          Save current
        </button>
      </div>
      {error ? <ErrorMessage message={error} /> : null}
    </div>
  );
}
