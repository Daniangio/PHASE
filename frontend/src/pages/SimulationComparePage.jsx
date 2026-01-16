import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ArrowLeftRight, ExternalLink } from 'lucide-react';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchResult, fetchResults, resultArtifactUrl } from '../api/jobs';

function formatTime(ts) {
  if (!ts) return '—';
  try {
    return new Date(ts).toLocaleString();
  } catch (err) {
    return ts;
  }
}

function SimulationPanel({ label, job, jobId }) {
  if (!jobId) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 text-sm text-gray-400">
        Select a simulation job to compare.
      </div>
    );
  }
  if (!job) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 text-sm text-gray-400">
        Loading {label}…
      </div>
    );
  }
  const artifacts = job.results || {};
  const hasMarginals = Boolean(artifacts.marginals_plot);
  const hasBetaScan = Boolean(artifacts.beta_scan_plot);

  return (
    <div className="space-y-4">
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-lg font-semibold text-white">{label}</h2>
          <div className="text-xs text-gray-400">{job.job_id}</div>
        </div>
        <dl className="grid grid-cols-2 gap-3 text-sm text-gray-300 mt-3">
          <div>
            <dt className="text-gray-400">Created</dt>
            <dd>{formatTime(job.created_at)}</dd>
          </div>
          <div>
            <dt className="text-gray-400">beta_eff</dt>
            <dd>{artifacts.beta_eff ?? '—'}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Cluster ID</dt>
            <dd className="truncate">{job.system_reference?.cluster_id || '—'}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Status</dt>
            <dd className="capitalize">{job.status}</dd>
          </div>
        </dl>
      </div>

      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between gap-2 mb-2">
          <h3 className="text-sm font-semibold text-white">Marginal comparison</h3>
          {hasMarginals && (
            <a
              href={resultArtifactUrl(jobId, 'marginals_plot')}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-cyan-300 hover:text-cyan-200 flex items-center gap-1"
            >
              <ExternalLink className="h-3 w-3" />
              Open
            </a>
          )}
        </div>
        {hasMarginals ? (
          <iframe
            title={`Marginal comparison ${label}`}
            src={resultArtifactUrl(jobId, 'marginals_plot')}
            className="w-full h-[520px] rounded-md border border-gray-700 bg-gray-900"
          />
        ) : (
          <p className="text-sm text-gray-400">No marginal comparison plot available.</p>
        )}
      </div>

      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between gap-2 mb-2">
          <h3 className="text-sm font-semibold text-white">beta_eff scan</h3>
          {hasBetaScan && (
            <a
              href={resultArtifactUrl(jobId, 'beta_scan_plot')}
              target="_blank"
              rel="noreferrer"
              className="text-xs text-cyan-300 hover:text-cyan-200 flex items-center gap-1"
            >
              <ExternalLink className="h-3 w-3" />
              Open
            </a>
          )}
        </div>
        {hasBetaScan ? (
          <iframe
            title={`beta_eff scan ${label}`}
            src={resultArtifactUrl(jobId, 'beta_scan_plot')}
            className="w-full h-[380px] rounded-md border border-gray-700 bg-gray-900"
          />
        ) : (
          <p className="text-sm text-gray-400">No beta_eff scan plot available.</p>
        )}
      </div>
    </div>
  );
}

export default function SimulationComparePage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [resultsList, setResultsList] = useState([]);
  const [leftJobId, setLeftJobId] = useState('');
  const [rightJobId, setRightJobId] = useState('');
  const [leftJob, setLeftJob] = useState(null);
  const [rightJob, setRightJob] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const simulationResults = useMemo(
    () =>
      (resultsList || []).filter(
        (item) => item.analysis_type === 'simulation' && item.status === 'finished'
      ),
    [resultsList]
  );

  useEffect(() => {
    const loadList = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchResults();
        setResultsList(data || []);
      } catch (err) {
        setError(err.message || 'Failed to load results.');
      } finally {
        setIsLoading(false);
      }
    };
    loadList();
  }, []);

  useEffect(() => {
    if (!simulationResults.length) return;
    const left = searchParams.get('left') || simulationResults[0]?.job_id;
    const right = searchParams.get('right') || simulationResults[1]?.job_id || left;
    setLeftJobId(left || '');
    setRightJobId(right || '');
  }, [searchParams, simulationResults]);

  useEffect(() => {
    const loadJob = async (jobId, setter) => {
      if (!jobId) {
        setter(null);
        return;
      }
      try {
        const data = await fetchResult(jobId);
        setter(data);
      } catch (err) {
        setter(null);
      }
    };
    loadJob(leftJobId, setLeftJob);
    loadJob(rightJobId, setRightJob);
  }, [leftJobId, rightJobId]);

  const handleSwap = () => {
    const nextLeft = rightJobId;
    const nextRight = leftJobId;
    setLeftJobId(nextLeft);
    setRightJobId(nextRight);
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev);
      if (nextLeft) params.set('left', nextLeft);
      if (nextRight) params.set('right', nextRight);
      return params;
    });
  };

  const updateSelection = (side, value) => {
    if (side === 'left') {
      setLeftJobId(value);
    } else {
      setRightJobId(value);
    }
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev);
      if (side === 'left') {
        value ? params.set('left', value) : params.delete('left');
      } else {
        value ? params.set('right', value) : params.delete('right');
      }
      return params;
    });
  };

  if (isLoading) return <Loader message="Loading simulation results..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!simulationResults.length) {
    return (
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 text-sm text-gray-300">
        No finished simulation results available to compare yet.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <button onClick={() => navigate('/results')} className="text-cyan-400 hover:text-cyan-300 text-sm">
          ← Back to results
        </button>
        <button
          onClick={handleSwap}
          className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-md text-sm text-gray-100 flex items-center gap-2"
        >
          <ArrowLeftRight className="h-4 w-4" />
          Swap
        </button>
      </div>

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h1 className="text-xl font-semibold text-white">Compare Simulation Runs</h1>
        <p className="text-sm text-gray-400 mt-1">
          Pick two finished simulation runs to compare their marginal and beta_eff plots.
        </p>
        <div className="grid md:grid-cols-2 gap-4 mt-4">
          <div>
            <label className="text-xs text-gray-400">Run A</label>
            <select
              value={leftJobId}
              onChange={(event) => updateSelection('left', event.target.value)}
              className="mt-1 w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-100"
            >
              {simulationResults.map((item) => (
                <option key={item.job_id} value={item.job_id}>
                  {item.job_id} · {formatTime(item.created_at)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400">Run B</label>
            <select
              value={rightJobId}
              onChange={(event) => updateSelection('right', event.target.value)}
              className="mt-1 w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-100"
            >
              {simulationResults.map((item) => (
                <option key={item.job_id} value={item.job_id}>
                  {item.job_id} · {formatTime(item.created_at)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </section>

      <div className="grid lg:grid-cols-2 gap-6">
        <SimulationPanel label="Run A" job={leftJob} jobId={leftJobId} />
        <SimulationPanel label="Run B" job={rightJob} jobId={rightJobId} />
      </div>
    </div>
  );
}
