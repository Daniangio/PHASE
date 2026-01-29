import ErrorMessage from '../common/ErrorMessage';
import { StateCard } from './StateCards';
import { AddStateForm } from './SystemDetailWidgets';

export default function SystemDetailMacroPanel({
  states,
  systemStatus,
  descriptorsReady,
  handleConfirmMacro,
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
  return (
    <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">States</h2>
        <div className="flex items-center space-x-3">
          <p className="text-xs text-gray-400">Status: {systemStatus}</p>
          <button
            onClick={handleConfirmMacro}
            disabled={states.length === 0 || !descriptorsReady}
            className="text-xs px-3 py-1 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50"
          >
            Confirm states
          </button>
        </div>
      </div>
      {downloadError && <ErrorMessage message={downloadError} />}
      {actionError && <ErrorMessage message={actionError} />}
      {actionMessage && <p className="text-sm text-emerald-400">{actionMessage}</p>}
      {!descriptorsReady && states.length > 0 && (
        <p className="text-xs text-amber-300">
          Upload trajectories and build descriptors for every state before confirming.
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
    </section>
  );
}
