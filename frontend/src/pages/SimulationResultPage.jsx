import { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Download, ExternalLink, BarChart3, ArrowLeftRight } from 'lucide-react';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchResult, downloadResultArtifact, resultArtifactUrl } from '../api/jobs';

const artifactButtons = [
  { key: 'summary_npz', label: 'Summary NPZ' },
  { key: 'metadata_json', label: 'Run metadata' },
  { key: 'cluster_npz', label: 'Cluster NPZ' },
];

function getFilename(pathValue, fallback) {
  if (typeof pathValue !== 'string') return fallback;
  const parts = pathValue.split('/');
  return parts[parts.length - 1] || fallback;
}

export default function SimulationResultPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloadError, setDownloadError] = useState(null);

  useEffect(() => {
    const load = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchResult(jobId);
        if (data.analysis_type !== 'simulation') {
          throw new Error('This result is not a simulation job.');
        }
        setResult(data);
      } catch (err) {
        setError(err.message || 'Failed to load simulation result.');
      } finally {
        setIsLoading(false);
      }
    };
    load();
  }, [jobId]);

  const artifacts = useMemo(() => result?.results || {}, [result]);
  const hasMarginals = Boolean(artifacts?.marginals_plot);
  const hasBetaScan = Boolean(artifacts?.beta_scan_plot);
  const betaEff = artifacts?.beta_eff;
  const clusterId = result?.system_reference?.cluster_id || '—';

  const handleDownload = async (key) => {
    setDownloadError(null);
    try {
      const blob = await downloadResultArtifact(jobId, key);
      const filename = getFilename(artifacts?.[key], `${jobId}-${key}`);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadError(err.message || 'Download failed.');
    }
  };

  if (isLoading) return <Loader message="Loading simulation result..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!result) return null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <button
          onClick={() => navigate(`/results/${jobId}`)}
          className="text-cyan-400 hover:text-cyan-300 text-sm"
        >
          ← Back to result
        </button>
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate(`/simulation/compare?left=${jobId}`)}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-md text-sm text-gray-100 flex items-center gap-2"
          >
            <ArrowLeftRight className="h-4 w-4" />
            Compare
          </button>
        </div>
      </div>

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between gap-3">
          <h1 className="text-2xl font-bold text-white">Simulation Summary</h1>
          <div className="text-xs text-gray-400">Job {result.job_id}</div>
        </div>
        <dl className="grid md:grid-cols-2 gap-4 text-sm text-gray-300 mt-4">
          <div>
            <dt className="text-gray-400">Status</dt>
            <dd className="capitalize">{result.status}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Cluster ID</dt>
            <dd className="truncate">{clusterId}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Created</dt>
            <dd>{new Date(result.created_at).toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Completed</dt>
            <dd>{result.completed_at ? new Date(result.completed_at).toLocaleString() : '—'}</dd>
          </div>
          <div>
            <dt className="text-gray-400">beta_eff</dt>
            <dd>{betaEff !== undefined && betaEff !== null ? betaEff : '—'}</dd>
          </div>
          <div>
            <dt className="text-gray-400">Results directory</dt>
            <dd className="truncate">{artifacts?.results_dir || '—'}</dd>
          </div>
        </dl>
      </section>

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-lg font-semibold text-white">Downloads</h2>
          {downloadError && <p className="text-xs text-red-400">{downloadError}</p>}
        </div>
        <div className="flex flex-wrap gap-3 mt-3">
          {artifactButtons.map((item) => (
            <button
              key={item.key}
              onClick={() => handleDownload(item.key)}
              disabled={!artifacts?.[item.key]}
              className={`px-3 py-2 rounded-md text-sm flex items-center gap-2 ${
                artifacts?.[item.key]
                  ? 'bg-gray-700 hover:bg-gray-600 text-gray-100'
                  : 'bg-gray-900 text-gray-500 cursor-not-allowed'
              }`}
            >
              <Download className="h-4 w-4" />
              {item.label}
            </button>
          ))}
        </div>
      </section>

      <section className="space-y-4">
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between gap-2 mb-3">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-cyan-400" />
              Marginal comparison
            </h2>
            {hasMarginals && (
              <a
                href={resultArtifactUrl(jobId, 'marginals_plot')}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-cyan-300 hover:text-cyan-200 flex items-center gap-1"
              >
                <ExternalLink className="h-3 w-3" />
                Open in new tab
              </a>
            )}
          </div>
          {hasMarginals ? (
            <iframe
              title="Marginal comparison"
              src={resultArtifactUrl(jobId, 'marginals_plot')}
              className="w-full h-[680px] rounded-md border border-gray-700 bg-gray-900"
            />
          ) : (
            <p className="text-sm text-gray-400">No marginal comparison plot available.</p>
          )}
        </div>

        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between gap-2 mb-3">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-cyan-400" />
              beta_eff scan
            </h2>
            {hasBetaScan && (
              <a
                href={resultArtifactUrl(jobId, 'beta_scan_plot')}
                target="_blank"
                rel="noreferrer"
                className="text-xs text-cyan-300 hover:text-cyan-200 flex items-center gap-1"
              >
                <ExternalLink className="h-3 w-3" />
                Open in new tab
              </a>
            )}
          </div>
          {hasBetaScan ? (
            <iframe
              title="beta_eff scan"
              src={resultArtifactUrl(jobId, 'beta_scan_plot')}
              className="w-full h-[520px] rounded-md border border-gray-700 bg-gray-900"
            />
          ) : (
            <p className="text-sm text-gray-400">No beta_eff scan plot available.</p>
          )}
        </div>
      </section>
    </div>
  );
}
