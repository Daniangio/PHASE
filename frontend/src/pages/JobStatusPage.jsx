import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchJobStatus } from '../api/jobs';

export default function JobStatusPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let timer;
    const poll = async () => {
      try {
        const data = await fetchJobStatus(jobId);
        setStatus(data);
        if (['finished', 'failed'].includes(data.status)) {
          clearInterval(timer);
        }
      } catch (err) {
        setError(err.message);
        clearInterval(timer);
      }
    };
    poll();
    timer = setInterval(poll, 4000);
    return () => clearInterval(timer);
  }, [jobId]);

  if (error) return <ErrorMessage message={error} />;
  if (!status) return <Loader message="Checking job status..." />;

  return (
    <div className="space-y-4">
      <button onClick={() => navigate('/projects')} className="text-cyan-400 hover:text-cyan-300 text-sm">
        ← Back to projects
      </button>
      <h1 className="text-2xl font-bold text-white">Job {jobId}</h1>
      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <p className="text-lg text-white">
          Status:{' '}
          <span className="font-semibold capitalize">
            {status.status}
          </span>
        </p>
        <p className="text-sm text-gray-400">Progress: {status.meta?.progress ?? 0}%</p>
        {status.meta?.status && <p className="text-sm text-gray-400 mt-1">{status.meta.status}</p>}
        {status.status === 'finished' && status.result?.job_id && (
          <button
            onClick={() => navigate(`/results/${status.result.job_id}`)}
            className="mt-4 px-4 py-2 bg-cyan-600 rounded-md text-white text-sm"
          >
            View Result
          </button>
        )}
      </section>
    </div>
  );
}
