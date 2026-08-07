import { useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

const buildDefaultForm = () => ({
  name: '',
  description: '',
  residueSelections: '',
  useSlugIds: false,
});

export default function SystemForm({ onCreate }) {
  const [form, setForm] = useState(buildDefaultForm);
  const [formKey, setFormKey] = useState(0);
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
      const payload = new FormData();
      if (form.name) payload.append('name', form.name);
      if (form.description) payload.append('description', form.description);
      payload.append('use_slug_ids', String(Boolean(form.useSlugIds)));
      const selectionsText = form.residueSelections.trim();
      if (selectionsText) {
        payload.append('residue_selections_text', selectionsText);
      }

      await onCreate(payload);
      setForm(buildDefaultForm());
      setFormKey((prev) => prev + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form
      key={formKey}
      onSubmit={handleSubmit}
      className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-4"
    >
      <div className="rounded-md border border-cyan-900/70 bg-cyan-950/20 p-3 text-sm text-cyan-400">
        Create the system first, then open it and add states from the States panel. This matches the console workflow and avoids forcing an initial PDB upload.
      </div>

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

      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isSubmitting ? 'Creating...' : 'Create System'}
      </button>
    </form>
  );
}
