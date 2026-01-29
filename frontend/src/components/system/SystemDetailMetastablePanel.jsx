import Loader from '../common/Loader';
import ErrorMessage from '../common/ErrorMessage';
import { InfoTooltip } from './SystemDetailWidgets';
import { MetastableCard } from './StateCards';

export default function SystemDetailMetastablePanel({
  metastableLocked,
  metaLoading,
  metaParamsOpen,
  setMetaParamsOpen,
  metaParams,
  setMetaParams,
  metaError,
  metaActionError,
  metastableStates,
  handleRunMetastable,
  handleConfirmMetastable,
  handleRenameMetastable,
  openDoc,
  navigate,
  projectId,
  systemId,
}) {
  return (
    <section className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold text-white">Metastable States (TICA)</h2>
          <InfoTooltip
            ariaLabel="Metastable states info"
            text="Open detailed documentation for the metastable pipeline and related methods."
            onClick={() => openDoc('metastable_states')}
          />
        </div>
        <div className="flex items-center space-x-2">
          {!metastableLocked && (
            <button
              onClick={() => setMetaParamsOpen((prev) => !prev)}
              className="text-xs px-3 py-1 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
            >
              Hyperparams
            </button>
          )}
          <button
            onClick={handleRunMetastable}
            disabled={metaLoading || metastableLocked}
            className="text-xs px-3 py-1 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10 disabled:opacity-50"
          >
            {metastableLocked ? 'Locked' : metaLoading ? 'Running…' : 'Recompute'}
          </button>
          <button
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/metastable/visualize`)}
            className="text-xs px-3 py-1 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10"
          >
            Visualize
          </button>
          {!metastableLocked && (
            <button
              onClick={handleConfirmMetastable}
              disabled={metaLoading || metastableStates.length === 0}
              className="text-xs px-3 py-1 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50"
            >
              Confirm metastable
            </button>
          )}
        </div>
      </div>
      {metaParamsOpen && !metastableLocked && (
        <div className="bg-gray-900 border border-gray-700 rounded-md p-3 space-y-3 text-sm">
          <div className="grid md:grid-cols-3 gap-3">
            {[
              {
                key: 'n_microstates',
                label: 'Microstates (k-means)',
                min: 2,
                info: 'Number of k-means clusters in TICA space before coarse-graining.',
              },
              {
                key: 'k_meta_min',
                label: 'Metastable min k',
                min: 1,
                info: 'Minimum metastable state count to test via spectral gap.',
              },
              {
                key: 'k_meta_max',
                label: 'Metastable max k',
                min: 1,
                info: 'Maximum metastable state count to test via spectral gap.',
              },
              {
                key: 'tica_lag_frames',
                label: 'TICA lag (frames)',
                min: 1,
                info: 'Lag time in frames for TICA projection.',
              },
              {
                key: 'tica_dim',
                label: 'TICA dims',
                min: 1,
                info: 'Number of TICA components retained for clustering.',
              },
              {
                key: 'random_state',
                label: 'Random seed',
                min: 0,
                info: 'Seed for k-means and MSM initialization.',
              },
            ].map((field) => (
              <label key={field.key} className="space-y-1">
                <span className="flex items-center gap-2 text-xs text-gray-400">
                  {field.label}
                  <InfoTooltip ariaLabel={`${field.label} info`} text={field.info} />
                </span>
                <input
                  type="number"
                  min={field.min}
                  value={metaParams[field.key]}
                  onChange={(e) =>
                    setMetaParams((prev) => ({
                      ...prev,
                      [field.key]: Number(e.target.value),
                    }))
                  }
                  className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                />
              </label>
            ))}
          </div>
        </div>
      )}
      {metaError && <ErrorMessage message={`Failed to load metastable states: ${metaError}`} />}
      {metaActionError && <ErrorMessage message={metaActionError} />}
      {metaLoading && <Loader message="Computing metastable states..." />}
      {!metaLoading && metastableStates.length === 0 && (
        <p className="text-sm text-gray-400">
          Metastable analysis is manual. Click Recompute after uploading trajectories and building descriptors.
        </p>
      )}
      {!metaLoading && metastableStates.length > 0 && (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
          {metastableStates.map((m) => (
            <MetastableCard
              key={m.metastable_id || `${m.macro_state}-${m.metastable_index}`}
              meta={m}
              onRename={handleRenameMetastable}
            />
          ))}
        </div>
      )}
    </section>
  );
}
