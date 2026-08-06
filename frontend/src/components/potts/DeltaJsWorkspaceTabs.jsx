import { useNavigate } from 'react-router-dom';

export default function DeltaJsWorkspaceTabs({ projectId, systemId, clusterId, active }) {
  const navigate = useNavigate();
  const base = `/projects/${encodeURIComponent(projectId)}/systems/${encodeURIComponent(systemId)}/sampling`;
  const suffix = clusterId ? `?cluster_id=${encodeURIComponent(clusterId)}` : '';
  const tabs = [
    { key: 'analysis', label: 'A/B analysis', path: `${base}/delta_js${suffix}` },
    { key: 'comparison', label: 'Ensemble comparison', path: `${base}/delta_js_table${suffix}` },
    { key: 'structure', label: 'Structure view', path: `${base}/delta_js_3d${suffix}` },
  ];

  return (
    <nav className="flex gap-1 overflow-x-auto rounded-lg border border-gray-800 bg-gray-900/60 p-1" aria-label="Delta JS pages">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          type="button"
          onClick={() => navigate(tab.path)}
          className={`shrink-0 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            active === tab.key ? 'bg-cyan-600 text-white' : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
