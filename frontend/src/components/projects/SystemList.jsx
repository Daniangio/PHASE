import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronDown, ChevronRight, Circle, CheckCircle, AlertTriangle, Copy, Trash2 } from 'lucide-react';
import IconButton from '../common/IconButton';

const statusIcon = {
  ready: CheckCircle,
  processing: Circle,
  failed: AlertTriangle,
};

export default function SystemList({ projectId, systems, onDelete, onClone }) {
  const [openStatesBySystem, setOpenStatesBySystem] = useState({});

  if (!projectId) {
    return <p className="text-gray-400 text-sm">Select a project to view its systems.</p>;
  }
  if (!systems?.length) {
    return <p className="text-gray-400 text-sm">No systems built yet.</p>;
  }
  return (
    <ul className="space-y-2">
      {systems.map((system) => {
        const Icon = statusIcon[system.status] || Circle;
        const states = Object.values(system.states || {});
        const stateNames = states.map((state) => state?.name || state?.state_id).filter(Boolean);
        const isOpen = Boolean(openStatesBySystem[system.system_id]);
        const statePreview = stateNames.slice(0, 3).join(', ');
        return (
          <li key={system.system_id} className="bg-gray-800 border border-gray-700 rounded-lg">
            <div className="flex items-center justify-between px-4 py-3">
              <Link
                to={`/projects/${projectId}/systems/${system.system_id}`}
                className="flex items-center space-x-3 flex-1 hover:text-white"
              >
                <div>
                  <p className="font-semibold text-white">{system.name}</p>
                  <div className="flex items-center space-x-2 text-sm text-gray-400">
                    <Icon className="h-4 w-4" />
                    <span className="capitalize">{system.status}</span>
                    <span>• {Object.keys(system.states || {}).length} states</span>
                    {!!stateNames.length && <span className="truncate max-w-[24rem]">• {statePreview}{stateNames.length > 3 ? '…' : ''}</span>}
                  </div>
                </div>
              </Link>
              <div className="flex items-center space-x-3">
                <button
                  type="button"
                  onClick={() =>
                    setOpenStatesBySystem((prev) => ({
                      ...prev,
                      [system.system_id]: !prev[system.system_id],
                    }))
                  }
                  className="text-gray-500 hover:text-gray-300"
                  aria-label={isOpen ? 'Collapse states' : 'Expand states'}
                >
                  {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                </button>
                <ChevronRight className="h-5 w-5 text-gray-500" />
                {onClone && (
                  <IconButton
                    icon={Copy}
                    label={`Clone ${system.name || 'system'}`}
                    title="Clone states, clusters, Potts models, and MD samples"
                    onClick={() => onClone(system)}
                    className="text-gray-500 hover:text-cyan-300"
                  />
                )}
                {onDelete && (
                  <IconButton
                    icon={Trash2}
                    label="Delete system"
                    onClick={() => onDelete(system.system_id)}
                    className="text-gray-500 hover:text-red-400"
                  />
                )}
              </div>
            </div>
            {isOpen && (
              <div className="border-t border-gray-700 px-4 py-3 space-y-2">
                {stateNames.length === 0 ? (
                  <p className="text-xs text-gray-400">No states.</p>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    {stateNames.map((name, idx) => (
                      <span key={`${system.system_id}:${idx}`} className="text-xs px-2 py-1 rounded-md border border-gray-700 text-gray-300 bg-gray-900/60">
                        {name}
                      </span>
                    ))}
                  </div>
                )}
                <div className="pt-1">
                  <Link
                    to={`/projects/${projectId}/systems/${system.system_id}`}
                    className="text-xs text-cyan-300 hover:text-cyan-400"
                  >
                    Open system to add / modify / delete states
                  </Link>
                </div>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}
