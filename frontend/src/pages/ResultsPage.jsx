import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import EmptyState from '../components/common/EmptyState';
import IconButton from '../components/common/IconButton';
import { fetchResults, resultArtifactUrl } from '../api/jobs';
import { Download, AlertTriangle, CheckCircle, Loader2, Trash2 } from 'lucide-react';
import { confirmAndDeleteResult } from '../utils/results';

export default function ResultsPage() {
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const loadResults = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchResults();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadResults();
  }, []);

  const handleDelete = async (jobId) => {
    await confirmAndDeleteResult(jobId, {
      onSuccess: loadResults,
      onError: (err) => setError(err.message || 'Failed to delete result.'),
    });
  };

  if (isLoading) return <Loader message="Loading results..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!results.length) {
    return (
      <EmptyState
        title="No analysis results yet"
        description="Queue a static or Potts simulation job from the system page to see results here."
        action={
          <button
            onClick={() => navigate('/projects')}
            className="px-4 py-2 bg-cyan-600 rounded-md text-white"
          >
            Go to Projects
          </button>
        }
      />
    );
  }

  return (
    <div className="space-y-4">
      {results.map((result) => (
        <article
          key={result.job_id}
          className="bg-gray-800 border border-gray-700 rounded-lg p-4 flex items-center justify-between"
        >
          <div>
            <div className="flex items-center space-x-3">
              <StatusIcon status={result.status} />
              <div>
                <p className="text-white font-semibold">{result.analysis_type.toUpperCase()}</p>
                <p className="text-sm text-gray-400">Job {result.job_id}</p>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-400 flex flex-wrap gap-2">
              {(result.project_name || result.project_id) && (
                <span>Project: {result.project_name || result.project_id.slice(0, 8)}</span>
              )}
              {(result.system_name || result.system_id) && (
                <span>System: {result.system_name || result.system_id.slice(0, 8)}</span>
              )}
              {result.analysis_type === 'potts_fit' && result.cluster_npz && (
                <span>Cluster NPZ: {result.cluster_npz.split('/').pop()}</span>
              )}
              {result.created_at && <span>Submitted: {new Date(result.created_at).toLocaleString()}</span>}
            </div>
          </div>
          <div className="flex items-center space-x-3">
            {result.rq_job_id && (
              <button
                onClick={() => navigate(`/jobs/${result.rq_job_id}`)}
                className="text-sm text-cyan-400 hover:text-cyan-300"
              >
                Status
              </button>
            )}
            <button
              onClick={() => navigate(`/results/${result.job_id}`)}
              className="text-sm text-cyan-400 hover:text-cyan-300"
            >
              Details
            </button>
            {result.analysis_type === 'simulation' && result.status === 'finished' && (
              <button
                onClick={() => navigate(`/simulation/${result.job_id}`)}
                className="text-sm text-cyan-400 hover:text-cyan-300"
              >
                Simulation
              </button>
            )}
            {result.analysis_type === 'simulation' && result.status === 'finished' && (
              <a
                href={resultArtifactUrl(result.job_id, 'summary_npz')}
                download={`${result.job_id}-samples.npz`}
                className="text-sm text-cyan-400 hover:text-cyan-300"
              >
                Samples
              </a>
            )}
            <a
              href={`/api/v1/results/${result.job_id}`}
              download={`${result.job_id}.json`}
              className="text-gray-400 hover:text-cyan-400"
            >
              <Download className="h-5 w-5" />
            </a>
            <IconButton
              icon={Trash2}
              label="Delete result"
              onClick={() => handleDelete(result.job_id)}
              className="text-gray-400 hover:text-red-400"
            />
          </div>
        </article>
      ))}
    </div>
  );
}

function StatusIcon({ status }) {
  if (status === 'finished') return <CheckCircle className="h-6 w-6 text-emerald-400" />;
  if (status === 'failed') return <AlertTriangle className="h-6 w-6 text-red-400" />;
  return <Loader2 className="h-6 w-6 text-amber-400 animate-spin" />;
}
