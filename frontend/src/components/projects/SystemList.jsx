import { Link } from 'react-router-dom';
import { ChevronRight, Circle, CheckCircle, AlertTriangle, Trash2 } from 'lucide-react';

const statusIcon = {
  ready: CheckCircle,
  processing: Circle,
  failed: AlertTriangle,
};

export default function SystemList({ projectId, systems, onDelete }) {
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
                  </div>
                </div>
              </Link>
              <div className="flex items-center space-x-3">
                <ChevronRight className="h-5 w-5 text-gray-500" />
                {onDelete && (
                  <button
                    onClick={() => onDelete(system.system_id)}
                    className="text-gray-500 hover:text-red-400"
                    title="Delete system"
                  >
                    <Trash2 className="h-5 w-5" />
                  </button>
                )}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
