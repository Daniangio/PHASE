import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import {
  fetchSystem,
  downloadStructure,
} from '../api/projects';
import {
  submitStaticJob,
  submitDynamicJob,
  submitQuboJob,
  fetchResults,
} from '../api/jobs';
import StaticAnalysisForm from '../components/analysis/StaticAnalysisForm';
import DynamicAnalysisForm from '../components/analysis/DynamicAnalysisForm';
import QuboAnalysisForm from '../components/analysis/QuboAnalysisForm';
import { Download } from 'lucide-react';

export default function SystemDetailPage() {
  const { projectId, systemId } = useParams();
  const [system, setSystem] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloadError, setDownloadError] = useState(null);
  const [staticResults, setStaticResults] = useState([]);
  const [staticError, setStaticError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const loadSystem = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    loadSystem();
  }, [projectId, systemId]);

  useEffect(() => {
    const loadStatics = async () => {
      if (!projectId || !systemId) return;
      try {
        const allResults = await fetchResults();
        const matching = allResults.filter(
          (res) => res.analysis_type === 'static' && res.system_id === systemId && res.status === 'finished'
        );
        setStaticResults(matching);
        setStaticError(null);
      } catch (err) {
        setStaticError(err.message);
      }
    };
    loadStatics();
  }, [projectId, systemId]);

  const enqueueJob = async (runner, params) => {
    const response = await runner({
      project_id: projectId,
      system_id: systemId,
      ...params,
    });
    navigate(`/jobs/${response.job_id}`, { state: { analysis_uuid: response.analysis_uuid } });
  };

  const handleDownloadStructure = async (stateKey) => {
    setDownloadError(null);
    try {
      const blob = await downloadStructure(projectId, systemId, stateKey);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${system?.name || systemId}-${stateKey}.pdb`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadError(err.message);
    }
  };

  if (isLoading) return <Loader message="Loading system..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!system) return null;

  return (
    <div className="space-y-8">
      <div>
        <button
          onClick={() => navigate('/projects')}
          className="text-cyan-400 hover:text-cyan-300 text-sm mb-2"
        >
          ← Back to Projects
        </button>
        <h1 className="text-2xl font-bold text-white">{system.name}</h1>
        <p className="text-gray-400 text-sm">{system.description || 'No description provided.'}</p>
      </div>

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h2 className="text-lg font-semibold text-white mb-3">States</h2>
        {downloadError && <ErrorMessage message={downloadError} />}
        <div className="grid md:grid-cols-2 gap-4">
          {Object.entries(system.states || {}).map(([key, state]) => (
            <article key={key} className="bg-gray-900 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-md font-semibold capitalize">{key}</h3>
                <button
                  onClick={() => handleDownloadStructure(key)}
                  className="flex items-center space-x-1 text-sm text-cyan-400 hover:text-cyan-300"
                >
                  <Download className="h-4 w-4" />
                  <span>Download PDB</span>
                </button>
              </div>
              <dl className="space-y-1 text-sm text-gray-300">
                <div className="flex justify-between">
                  <dt>Frames</dt>
                  <dd>{state.n_frames}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Stride</dt>
                  <dd>{state.stride}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Source</dt>
                  <dd className="truncate">{state.source_traj || 'Uploaded'}</dd>
                </div>
              </dl>
            </article>
          ))}
        </div>
      </section>

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        <h2 className="text-lg font-semibold text-white mb-4">Run Analysis</h2>
        {staticError && <ErrorMessage message={`Failed to load static jobs: ${staticError}`} />}
        <div className="grid md:grid-cols-3 gap-6">
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-md font-semibold text-white mb-2">Static Reporters</h3>
            <StaticAnalysisForm onSubmit={(params) => enqueueJob(submitStaticJob, params)} />
          </div>
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-md font-semibold text-white mb-2">Dynamic TE</h3>
            <DynamicAnalysisForm onSubmit={(params) => enqueueJob(submitDynamicJob, params)} />
          </div>
          <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
            <h3 className="text-md font-semibold text-white mb-2">QUBO</h3>
            <QuboAnalysisForm
              staticOptions={staticResults}
              onSubmit={(params) => enqueueJob(submitQuboJob, params)}
            />
          </div>
        </div>
      </section>
    </div>
  );
}
