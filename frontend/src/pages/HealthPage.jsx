import { useEffect, useState } from 'react';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { healthCheck } from '../api/jobs';

export default function HealthPage() {
  const [report, setReport] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

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
      <h1 className="text-2xl font-bold text-white">System Health</h1>
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
