import { useNavigate } from 'react-router-dom';

export default function NearestNeighborWorkspaceTabs({ projectId, systemId, clusterId, active }) {
  const navigate = useNavigate();
  const base = `/projects/${encodeURIComponent(projectId)}/systems/${encodeURIComponent(systemId)}/sampling`;
  const suffix = clusterId ? `?cluster_id=${encodeURIComponent(clusterId)}` : '';
  const tabs = [
    { key: 'mapping', label: 'Mapping', path: `${base}/potts_nn_mapping${suffix}` },
    { key: 'compare', label: 'Compare', path: `${base}/potts_nn_mapping_compare${suffix}` },
  ];

  return (
    <nav className="flex gap-1 rounded-lg border border-gray-800 bg-gray-900/60 p-1" aria-label="Nearest-neighbour pages">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          type="button"
          onClick={() => navigate(tab.path)}
          className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            active === tab.key ? 'bg-cyan-600 text-white' : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
