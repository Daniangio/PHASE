import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchResult } from '../api/jobs';

const metricNames = {
  auc: 'Logistic AUC',
  mi: 'Mutual Information',
  jsd: 'Jensen-Shannon Divergence',
  mmd: 'Maximum Mean Discrepancy',
  kl: 'Symmetrized KL',
};

export default function ResultDetailPage() {
  const { jobId } = useParams();
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const load = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchResult(jobId);
        setResult(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    load();
  }, [jobId]);

  if (isLoading) return <Loader message="Loading result..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!result) return null;

  const systemRef = result.system_reference || {};

  const canVisualize =
    result.residue_selections_mapping &&
    ((result.analysis_type === 'static' && typeof result.results === 'object') ||
      (result.analysis_type === 'qubo' &&
        (result.results?.qubo_active?.solutions?.length > 0 ||
          result.results?.qubo_inactive?.solutions?.length > 0)));

  return (
    <div className="space-y-6">
      <button onClick={() => navigate('/results')} className="text-cyan-400 hover:text-cyan-300 text-sm">
        ← Back to results
      </button>

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h1 className="text-2xl font-bold text-white mb-2">Job {result.job_id}</h1>
        <dl className="grid md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>
            <dt className="text-gray-400">Type</dt>
            <dd className="capitalize">{result.analysis_type}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Status</dt>
            <dd className="capitalize">{result.status}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Project</dt>
            <dd>{systemRef.project_name || systemRef.project_id || '—'}</dd>
          </div>
          <div>
            <dt className="text-gray-400">System</dt>
            <dd>{systemRef.system_name || systemRef.system_id || '—'}</dd>
          </div>
          <div>
            <dt className="text-gray-400">State A</dt>
            <dd>{systemRef.states?.state_a?.name || systemRef.states?.state_a?.id || '—'}</dd>
          </div>
          <div>
            <dt className="text-gray-400">State B</dt>
            <dd>{systemRef.states?.state_b?.name || systemRef.states?.state_b?.id || '—'}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Created</dt>
            <dd>{new Date(result.created_at).toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Completed</dt>
            <dd>{result.completed_at ? new Date(result.completed_at).toLocaleString() : '—'}</dd>
          </div>
        </dl>
        {result.params?.state_metric && (
          <p className="text-xs text-gray-500 mt-2">
            Metric: {metricNames[result.params.state_metric] || result.params.state_metric}
          </p>
        )}
      </section>

      {result.error && <ErrorMessage message={result.error} />}

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-lg font-semibold text-white">Result Payload</h2>
          {canVisualize && (
            <button
              onClick={() => navigate(`/visualize/${result.job_id}`)}
              className="px-3 py-1 bg-cyan-600 rounded-md text-white text-sm"
            >
              Visualize
            </button>
          )}
        </div>
        <pre className="bg-gray-900 rounded-md p-4 overflow-auto text-xs text-gray-200">
          {JSON.stringify(result.results, null, 2)}
        </pre>
      </section>
    </div>
  );
}
