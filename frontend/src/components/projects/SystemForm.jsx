import { useState } from 'react';
import ErrorMessage from '../common/ErrorMessage';

const defaultForm = {
  name: '',
  description: '',
  activePdb: null,
  inactivePdb: null,
  activeTraj: null,
  inactiveTraj: null,
  activeStride: 1,
  inactiveStride: 1,
  residueSelections: '',
};

export default function SystemForm({ onCreate }) {
  const [form, setForm] = useState(defaultForm);
  const [formKey, setFormKey] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleFileChange = (field, files) => {
    handleChange(field, files?.[0] || null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.activePdb || !form.activeTraj || !form.inactiveTraj) {
      setError('Active PDB, active trajectory, and inactive trajectory are required.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    try {
      const payload = new FormData();
      if (form.name) payload.append('name', form.name);
      if (form.description) payload.append('description', form.description);
      payload.append('active_pdb', form.activePdb);
      if (form.inactivePdb) payload.append('inactive_pdb', form.inactivePdb);
      payload.append('active_traj', form.activeTraj);
      payload.append('inactive_traj', form.inactiveTraj);
      payload.append('active_stride', form.activeStride);
      payload.append('inactive_stride', form.inactiveStride);
      if (form.residueSelections) {
        payload.append('residue_selections_json', form.residueSelections);
      }

      await onCreate(payload);
      setForm(defaultForm);
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
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-300 mb-1">System Name</label>
          <input
            type="text"
            value={form.name}
            onChange={(e) => handleChange('name', e.target.value)}
            className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
            placeholder="Inactive vs Active complex"
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FileInput label="Active PDB" required onChange={(files) => handleFileChange('activePdb', files)} />
        <FileInput label="Inactive PDB (optional)" onChange={(files) => handleFileChange('inactivePdb', files)} />
        <FileInput label="Active Trajectory" required onChange={(files) => handleFileChange('activeTraj', files)} />
        <FileInput label="Inactive Trajectory" required onChange={(files) => handleFileChange('inactiveTraj', files)} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StrideInput
          label="Active Stride"
          value={form.activeStride}
          onChange={(value) => handleChange('activeStride', value)}
        />
        <StrideInput
          label="Inactive Stride"
          value={form.inactiveStride}
          onChange={(value) => handleChange('inactiveStride', value)}
        />
      </div>

      <div>
        <label className="block text-sm text-gray-300 mb-1">Residue Selections JSON (optional)</label>
        <textarea
          rows={4}
          value={form.residueSelections}
          onChange={(e) => handleChange('residueSelections', e.target.value)}
          placeholder='{"res_50": "resid 50"}'
          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
        />
      </div>

      <ErrorMessage message={error} />
      <button
        type="submit"
        disabled={isSubmitting}
        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
      >
        {isSubmitting ? 'Processing...' : 'Build Descriptor System'}
      </button>
    </form>
  );
}

function FileInput({ label, onChange, required }) {
  return (
    <div>
      <label className="block text-sm text-gray-300 mb-1">
        {label} {required && <span className="text-red-400">*</span>}
      </label>
      <input
        type="file"
        onChange={(e) => onChange(e.target.files)}
        required={required}
        className="block w-full text-sm text-gray-300 bg-gray-900 border border-gray-700 rounded-md cursor-pointer file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-cyan-50 file:text-cyan-700 hover:file:bg-cyan-100"
      />
    </div>
  );
}

function StrideInput({ label, value, onChange }) {
  return (
    <div>
      <label className="block text-sm text-gray-300 mb-1">{label}</label>
      <input
        type="number"
        min={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white focus:ring-cyan-500"
      />
    </div>
  );
}
