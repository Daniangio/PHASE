import { useEffect, useState } from 'react';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { cleanupResults, healthCheck } from '../api/jobs';

export default function HealthPage() {
  const [report, setReport] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [cleanupInfo, setCleanupInfo] = useState(null);
  const [cleanupError, setCleanupError] = useState(null);
  const [cleanupBusy, setCleanupBusy] = useState(false);

  useEffect(() => {
    const load = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await healthCheck();
        setReport(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    load();
  }, []);

  if (isLoading) return <Loader message="Running health check..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!report) return null;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold text-white">System Health</h1>
        <button
          type="button"
          onClick={async () => {
            setCleanupBusy(true);
            setCleanupError(null);
            try {
              const data = await cleanupResults(true);
              setCleanupInfo(data);
            } catch (err) {
              setCleanupError(err.message || 'Failed to run cleanup.');
            } finally {
              setCleanupBusy(false);
            }
          }}
          className="px-3 py-1.5 rounded-md bg-cyan-600 text-sm text-white hover:bg-cyan-500 disabled:opacity-60"
          disabled={cleanupBusy}
        >
          {cleanupBusy ? 'Cleaning...' : 'Run Cleanup'}
        </button>
      </div>
      {cleanupError && <ErrorMessage message={cleanupError} />}
      {cleanupInfo && (
        <div className="text-xs text-gray-400">
          Removed {cleanupInfo.empty_result_dirs_removed} empty result folders and{' '}
          {cleanupInfo.tmp_artifacts_removed} tmp artifacts.
        </div>
      )}
      <section className="grid md:grid-cols-3 gap-4">
        {Object.entries(report).map(([key, value]) => {
          const normalized = typeof value === 'string' ? { status: value } : value;
          return (
            <article key={key} className="bg-gray-800 border border-gray-700 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-white capitalize mb-2">{key.replace('_', ' ')}</h2>
              <p
                className={`text-sm font-semibold ${
                  normalized.status === 'ok' ? 'text-emerald-400' : 'text-red-400'
                }`}
              >
                {normalized.status}
              </p>
              {normalized.info && <p className="text-xs text-gray-400 mt-2">{normalized.info}</p>}
              {normalized.error && <p className="text-xs text-red-400 mt-2">{normalized.error}</p>}
            </article>
          );
        })}
      </section>
    </div>
  );
}
