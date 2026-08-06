import { useEffect, useMemo, useState } from 'react';
import { Activity, Boxes, Database, GitCompareArrows, Microscope, Network, Waypoints } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';

import { fetchSystem } from '../../api/projects';

const MAIN_DESTINATIONS = [
  { key: 'system', label: 'System', icon: Database, suffix: '' },
  { key: 'descriptors', label: 'Visualize descriptors', icon: Microscope, suffix: '/descriptors/visualize' },
  { key: 'potts', label: 'Potts models', icon: Boxes, suffix: '/potts' },
  { key: 'sampling', label: 'Sampling explorer', icon: Activity, suffix: '/sampling/visualize' },
  { key: 'model_pair', label: 'Model-pair analysis', icon: GitCompareArrows, suffix: '/sampling/delta_eval' },
  { key: 'delta_js', label: 'Delta JS', icon: Network, suffix: '/sampling/delta_js' },
  { key: 'nearest', label: 'Nearest neighbours', icon: Waypoints, suffix: '/sampling/potts_nn_mapping' },
];

const ANALYSIS_DESTINATIONS = [
  { label: 'More analyses...', suffix: '' },
  { label: 'NN mismatch graph', suffix: '/sampling/potts_nn_mapping_graph' },
  { label: 'Hamiltonian spectra', suffix: '/sampling/hamiltonian_spectral' },
  { label: 'Spectral pistons', suffix: '/sampling/spectral_intersection' },
  { label: 'Transient states', suffix: '/sampling/transient_states' },
  { label: 'Residue selections', suffix: '/residue_selections' },
];

function parseSystemPath(pathname) {
  const match = String(pathname || '').match(/^\/projects\/([^/]+)\/systems\/([^/]+)/);
  if (!match) return null;
  return { projectId: decodeURIComponent(match[1]), systemId: decodeURIComponent(match[2]) };
}

function activeDestination(pathname, basePath) {
  if (pathname === basePath || pathname === `${basePath}/`) return 'system';
  if (pathname.startsWith(`${basePath}/descriptors/`)) return 'descriptors';
  if (pathname.startsWith(`${basePath}/potts`)) return 'potts';
  if (pathname.startsWith(`${basePath}/sampling/visualize`)) return 'sampling';
  if (pathname.startsWith(`${basePath}/sampling/delta_eval`) || pathname.startsWith(`${basePath}/sampling/delta_commitment_3d`)) return 'model_pair';
  if (pathname.startsWith(`${basePath}/sampling/delta_js`)) return 'delta_js';
  if (pathname.startsWith(`${basePath}/sampling/potts_nn_mapping`)) return 'nearest';
  return '';
}

export default function SystemWorkspaceNav() {
  const location = useLocation();
  const navigate = useNavigate();
  const route = useMemo(() => parseSystemPath(location.pathname), [location.pathname]);
  const [system, setSystem] = useState(null);

  useEffect(() => {
    let active = true;
    if (!route) {
      setSystem(null);
      return () => { active = false; };
    }
    fetchSystem(route.projectId, route.systemId)
      .then((payload) => { if (active) setSystem(payload); })
      .catch(() => { if (active) setSystem(null); });
    return () => { active = false; };
  }, [route]);

  if (!route) return null;

  const basePath = `/projects/${encodeURIComponent(route.projectId)}/systems/${encodeURIComponent(route.systemId)}`;
  const activeKey = activeDestination(location.pathname, basePath);
  const clusterId = new URLSearchParams(location.search).get('cluster_id') || '';
  const counts = {
    states: Object.keys(system?.states || {}).length,
    metastable: Array.isArray(system?.metastable_states) ? system.metastable_states.length : 0,
    clusters: Array.isArray(system?.metastable_clusters) ? system.metastable_clusters.filter((item) => item?.path).length : 0,
  };
  const withCluster = (path) => {
    if (!clusterId || path === basePath || path.includes('/descriptors/')) return path;
    const params = new URLSearchParams();
    params.set('cluster_id', clusterId);
    return `${path}?${params.toString()}`;
  };

  return (
    <section className="mb-6 overflow-hidden rounded-xl border border-gray-700 bg-gray-800/90 shadow-lg">
      <div className="flex flex-col gap-3 border-b border-gray-700 px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <p className="text-[10px] uppercase tracking-[0.22em] text-gray-500">System workspace</p>
          <div className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h2 className="truncate text-base font-semibold text-white">{system?.name || route.systemId}</h2>
            <p className="text-xs text-gray-400">
              {counts.states} states · {counts.metastable} metastable · {counts.clusters} clusters
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Network className="h-4 w-4 text-cyan-300" />
          <select
            value=""
            onChange={(event) => {
              const suffix = event.target.value;
              if (suffix) navigate(withCluster(`${basePath}${suffix}`));
            }}
            className="rounded-md border border-gray-600 bg-gray-900 px-3 py-2 text-xs text-gray-200"
            aria-label="Open another system analysis page"
          >
            {ANALYSIS_DESTINATIONS.map((item) => (
              <option key={item.label} value={item.suffix}>{item.label}</option>
            ))}
          </select>
        </div>
      </div>
      <nav className="flex gap-1 overflow-x-auto px-3 py-2" aria-label="System workspace">
        {MAIN_DESTINATIONS.map(({ key, label, icon: Icon, suffix }) => (
          <button
            key={key}
            type="button"
            onClick={() => navigate(withCluster(`${basePath}${suffix}`))}
            className={`inline-flex shrink-0 items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
              activeKey === key
                ? 'bg-cyan-600 text-white'
                : 'text-gray-300 hover:bg-gray-700 hover:text-white'
            }`}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </nav>
    </section>
  );
}
