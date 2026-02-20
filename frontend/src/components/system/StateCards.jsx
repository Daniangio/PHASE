import { useState } from 'react';
import { Download } from 'lucide-react';

export function StateCard({
  state,
  onDownload,
  onUpload,
  onDeleteTrajectory,
  onDeleteState,
  uploading,
  progress,
  processing,
}) {
  const [file, setFile] = useState(null);
  const [sliceSpec, setSliceSpec] = useState(state?.slice_spec || '');
  const [residueSelection, setResidueSelection] = useState(state?.residue_selection || '');
  const [residShift, setResidShift] = useState(String(state?.resid_shift ?? 0));

  const descriptorLabel = state?.descriptor_file ? 'Ready' : 'Not built';
  const trajectoryLabel = state?.source_traj || '—';

  return (
    <article className="bg-gray-900 rounded-lg p-4 border border-gray-700 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-md font-semibold text-white">{state.name}</h3>
          <p className="text-xs text-gray-500">{state.state_id?.slice(0, 8)}</p>
        </div>
        <button
          onClick={onDownload}
          disabled={!state?.pdb_file}
          className="flex items-center space-x-1 text-sm text-cyan-400 hover:text-cyan-300 disabled:opacity-50"
        >
          <Download className="h-4 w-4" />
          <span>PDB</span>
        </button>
      </div>

      <dl className="space-y-1 text-sm text-gray-300">
        <div className="flex justify-between">
          <dt>Trajectory</dt>
          <dd className="truncate text-gray-200">{trajectoryLabel}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Frames</dt>
          <dd>{state?.n_frames ?? 0}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Slice</dt>
          <dd>{state?.slice_spec || `::${state?.stride ?? 1}`}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Resid shift</dt>
          <dd>{state?.resid_shift ?? 0}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Descriptors</dt>
          <dd className="truncate text-gray-200">{descriptorLabel}</dd>
        </div>
      </dl>

      {(uploading || progress !== undefined || processing) && (
        <div className="space-y-1 text-xs text-gray-300">
          <div className="flex items-center justify-between">
            <span>{processing ? 'Processing descriptors' : 'Uploading trajectory'}</span>
            <span>{progress !== undefined ? `${progress}%` : ''}</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full ${processing ? 'bg-amber-400 animate-pulse' : 'bg-cyan-500'}`}
              style={{ width: progress !== undefined ? `${progress}%` : '33%' }}
            />
          </div>
        </div>
      )}

      <div className="space-y-2">
        <div>
          <label className="block text-sm text-gray-300 mb-1">Upload trajectory</label>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="w-full text-sm text-gray-300"
          />
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">Frame slice (start:stop:step)</label>
          <input
            type="text"
            value={sliceSpec}
            onChange={(e) => setSliceSpec(e.target.value)}
            placeholder="e.g. 0:10000:10 or ::5"
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
          />
          <p className="text-[11px] text-gray-500 mt-1">Leave blank for full trajectory. Single number = step.</p>
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">Residue filter (MDAnalysis)</label>
          <input
            type="text"
            value={residueSelection}
            onChange={(e) => setResidueSelection(e.target.value)}
            placeholder="protein (default), e.g. resnum 10-300"
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
          />
          <p className="text-[11px] text-gray-500 mt-1">Applied as: protein and (&lt;filter&gt;).</p>
        </div>
        <div>
          <label className="block text-sm text-gray-300 mb-1">Residue shift offset</label>
          <input
            type="number"
            step="1"
            value={residShift}
            onChange={(e) => setResidShift(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
          />
          <p className="text-[11px] text-gray-500 mt-1">
            Applied to saved residue keys (e.g. -2 maps <span className="font-mono">res_4</span> to <span className="font-mono">res_2</span>).
          </p>
        </div>
        <button
          onClick={() => onUpload(state.state_id, file, sliceSpec, residueSelection, residShift)}
          disabled={uploading || !file}
          className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md transition-colors disabled:opacity-50"
        >
          {uploading ? 'Uploading...' : 'Upload & Build'}
        </button>
      </div>

      <div className="flex items-center justify-between text-xs">
        <button
          onClick={onDeleteTrajectory}
          className="text-red-300 hover:text-red-200 disabled:opacity-50"
          disabled={!state?.trajectory_file && !state?.descriptor_file}
        >
          Remove trajectory
        </button>
        <button onClick={onDeleteState} className="text-gray-400 hover:text-red-300">
          Delete state
        </button>
      </div>
    </article>
  );
}

export function MetastableCard({ meta, onRename }) {
  const [name, setName] = useState(meta.name || meta.default_name || `Meta ${meta.metastable_index}`);
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    if (!meta.metastable_id) return;
    setIsSaving(true);
    try {
      await onRename(meta.metastable_id, name);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-400">{meta.macro_state}</p>
        <span className="text-xs text-gray-500">Frames: {meta.n_frames ?? '—'}</span>
      </div>
      <div>
        <label className="block text-xs text-gray-400 mb-1">Name</label>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white text-sm"
        />
      </div>
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span>ID: {meta.metastable_id || meta.metastable_index}</span>
        <button
          onClick={handleSave}
          disabled={isSaving || !meta.metastable_id}
          className="text-cyan-400 hover:text-cyan-300 disabled:opacity-50"
        >
          {isSaving ? 'Saving…' : 'Save'}
        </button>
      </div>
      {meta.representative_pdb && (
        <p className="text-xs text-cyan-400 break-all">
          Representative PDB: {meta.representative_pdb}
        </p>
      )}
    </div>
  );
}
