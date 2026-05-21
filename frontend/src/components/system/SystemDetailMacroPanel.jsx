import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import ErrorMessage from '../common/ErrorMessage';
import { StateCard } from './StateCards';
import { AddStateForm } from './SystemDetailWidgets';

export default function SystemDetailMacroPanel({
  states,
  systemStatus,
  descriptorsReady,
  downloadError,
  actionError,
  actionMessage,
  handleDownloadStructure,
  handleUploadTrajectory,
  handleDeleteTrajectory,
  handleDeleteState,
  uploadingState,
  uploadProgress,
  processingState,
  handleAddState,
  addingState,
}) {
  const [expanded, setExpanded] = useState(false);
  const previewNames = (states || [])
    .map((state) => state?.name || state?.state_id)
    .filter(Boolean)
    .slice(0, 6);

  return (
    <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">States</h2>
          <p className="text-xs text-gray-400">
            {states.length} states
            {!!previewNames.length && ` · ${previewNames.join(', ')}${states.length > previewNames.length ? '…' : ''}`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <p className="text-xs text-gray-400">Status: {systemStatus}</p>
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
          >
            {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
            {expanded ? 'Hide details' : 'Show details'}
          </button>
        </div>
      </div>
      {downloadError && <ErrorMessage message={downloadError} />}
      {actionError && <ErrorMessage message={actionError} />}
      {actionMessage && <p className="text-sm text-emerald-400">{actionMessage}</p>}
      {!expanded && (
        <div className="space-y-2">
        <div className="flex flex-wrap gap-2">
          {states.length === 0 && <p className="text-sm text-gray-400">No states yet.</p>}
          {states.map((state) => (
            <span
              key={state.state_id}
              className="text-xs px-2 py-1 rounded-md border border-gray-700 text-gray-300 bg-gray-900/60"
            >
              {state.name || state.state_id}
              <span className={`ml-1 ${state.trajectory_file ? 'text-emerald-300' : 'text-amber-300'}`}>
                {state.trajectory_file ? 'traj' : 'no traj'}
              </span>
            </span>
          ))}
        </div>
        <p className="text-[11px] text-gray-500">
          Open state details to upload or replace trajectories and rebuild descriptors for existing states.
        </p>
        </div>
      )}
      {expanded && (
        <>
      {!descriptorsReady && states.length > 0 && (
        <p className="text-xs text-amber-300">
          Trajectories are optional. Upload them later to build descriptors; PDB-only states can still be used for single-pose Potts energy evaluation.
        </p>
      )}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="md:col-span-2 grid sm:grid-cols-2 gap-3">
          {states.length === 0 && <p className="text-sm text-gray-400">No states yet.</p>}
          {states.map((state) => (
            <StateCard
              key={state.state_id}
              state={state}
              onDownload={() => handleDownloadStructure(state.state_id, state.name)}
              onUpload={handleUploadTrajectory}
              onDeleteTrajectory={() => handleDeleteTrajectory(state.state_id)}
              onDeleteState={() => handleDeleteState(state.state_id, state.name)}
              uploading={uploadingState === state.state_id}
              progress={uploadProgress[state.state_id]}
              processing={processingState === state.state_id}
            />
          ))}
        </div>
        <AddStateForm states={states} onAdd={handleAddState} isAdding={addingState} />
      </div>
        </>
      )}
    </section>
  );
}
