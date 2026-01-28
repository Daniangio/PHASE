import { useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

const createEmptyState = (index = 0) => ({
  key: `${Date.now()}-${index}`,
  name: `State ${index + 1}`,
  file: null,
});

const buildDefaultForm = () => ({
  name: '',
  description: '',
  residueSelections: '',
  useSlugIds: false,
  states: [createEmptyState()],
});

export default function SystemForm({ onCreate }) {
  const [form, setForm] = useState(buildDefaultForm);
  const [formKey, setFormKey] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const updateState = (key, field, value) => {
    setForm((prev) => ({
      ...prev,
      states: prev.states.map((state) => (state.key === key ? { ...state, [field]: value } : state)),
    }));
  };

  const addStateRow = () => {
    setForm((prev) => ({
      ...prev,
      states: [...prev.states, createEmptyState(prev.states.length)],
    }));
  };

  const removeStateRow = (key) => {
    setForm((prev) => {
      const remaining = prev.states.filter((state) => state.key !== key);
      return { ...prev, states: remaining.length ? remaining : [createEmptyState()] };
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setUploadProgress(null);

    const validStates = form.states.filter((state) => state.file);
    if (!validStates.length) {
      setIsSubmitting(false);
      setError('Add at least one state with a PDB file.');
      return;
    }
    if (validStates.some((state) => !state.name?.trim())) {
      setIsSubmitting(false);
      setError('Every state needs a name.');
      return;
    }

    try {
      const payload = new FormData();
      if (form.name) payload.append('name', form.name);
      if (form.description) payload.append('description', form.description);
      payload.append('use_slug_ids', String(Boolean(form.useSlugIds)));
      const selectionsText = form.residueSelections.trim();
      if (selectionsText) {
        payload.append('residue_selections_text', selectionsText);
      }
      payload.append('state_names', JSON.stringify(validStates.map((state) => state.name.trim())));
      validStates.forEach((state) => {
        payload.append('pdb_files', state.file);
      });

      await onCreate(payload, {
        onUploadProgress: (percent) => setUploadProgress(percent),
      });
      setForm(buildDefaultForm());
      setFormKey((prev) => prev + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
      setUploadProgress(null);
    }
  };

  return (
    <form
      key={formKey}
      onSubmit={handleSubmit}
      className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-4"
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-300 mb-1">System Name</label>
          <input
            type="text"
            value={form.name}
            onChange={(e) => handleChange('name', e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
            placeholder="Kinase ensemble"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">Description</label>
          <input
            type="text"
            value={form.description}
            onChange={(e) => handleChange('description', e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
          />
        </div>
      </div>
      <label className="flex items-center gap-2 text-sm text-gray-300">
        <input
          type="checkbox"
          checked={form.useSlugIds}
          onChange={(e) => handleChange('useSlugIds', e.target.checked)}
          className="rounded border-gray-600 text-cyan-500 focus:ring-cyan-500"
        />
        Use system name as folder ID (slug)
      </label>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-200 font-semibold">States</p>
            <p className="text-xs text-gray-500">Upload one PDB per state. More PDBs = more states.</p>
          </div>
          <button
            type="button"
            onClick={addStateRow}
            className="text-sm px-3 py-1 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10"
          >
            + Add state
          </button>
        </div>

        <div className="space-y-3">
          {form.states.map((state, idx) => (
            <div key={state.key} className="rounded-md border border-gray-700 p-3 bg-gray-900 space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-300">State {idx + 1}</p>
                {form.states.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeStateRow(state.key)}
                    className="text-xs text-red-300 hover:text-red-200"
                  >
                    Remove
                  </button>
                )}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Name</label>
                  <input
                    type="text"
                    value={state.name}
                    onChange={(e) => updateState(state.key, 'name', e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
                    required
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">PDB File</label>
                  <input
                    type="file"
                    onChange={(e) => updateState(state.key, 'file', e.target.files?.[0] || null)}
                    className="block w-full text-sm text-gray-300 bg-gray-800 border border-gray-700 rounded-md cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-cyan-50 file:text-cyan-700 hover:file:bg-cyan-100"
                    required
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Residue Selections (optional)</label>
        <textarea
          rows={4}
          value={form.residueSelections}
          onChange={(e) => handleChange('residueSelections', e.target.value)}
          placeholder={'resid 50 51\nchain A and resid 10 to 15 [singles]\nsegid CORE and resid 20 to 25 [pairs]'}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        />
        <p className="text-xs text-gray-500 mt-1">
          Enter one selection per line. Use optional [singles] or [pairs] wildcards to expand entries automatically.
        </p>
      </div>

      {uploadProgress !== null && (
        <div>
          <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
            <span>Uploading files</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-cyan-500 transition-all duration-200"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isSubmitting ? 'Creating…' : 'Create System'}
      </button>
    </form>
  );
}
