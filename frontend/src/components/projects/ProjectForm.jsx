import { useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

export default function ProjectForm({ onCreate }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [useSlugIds, setUseSlugIds] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      await onCreate({ name, description, use_slug_ids: useSlugIds });
      setName('');
      setDescription('');
    } catch (err) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-4">
      <div>
        <label className="block text-sm text-gray-300 mb-1">Project Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-2 focus:ring-cyan-500"
        />
      </div>
      <div>
        <label className="block text-sm text-gray-300 mb-1">Description</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-2 focus:ring-cyan-500"
        />
      </div>
      <label className="flex items-center gap-2 text-sm text-gray-300">
        <input
          type="checkbox"
          checked={useSlugIds}
          onChange={(e) => setUseSlugIds(e.target.checked)}
          className="rounded border-gray-600 text-cyan-500 focus:ring-cyan-500"
        />
        Use project name as folder ID (slug)
      </label>
      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isSubmitting ? 'Creating...' : 'Create Project'}
      </button>
    </form>
  );
}
