import { CheckCircle, Clock } from 'lucide-react';

export default function ProjectList({ projects, selectedId, onSelect }) {
  if (!projects.length) {
    return <p className="text-gray-400 text-sm">No projects yet.</p>;
  }

  return (
    <ul className="space-y-2">
      {projects.map((project) => (
        <li key={project.project_id}>
          <button
            onClick={() => onSelect(project.project_id)}
            className={`w-full text-left px-4 py-3 rounded-lg border transition-colors ${
              selectedId === project.project_id
                ? 'bg-cyan-600/20 border-cyan-500 text-white'
                : 'bg-gray-800 border-gray-700 text-gray-200 hover:border-cyan-500'
            }`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="font-semibold">{project.name}</p>
                <p className="text-xs text-gray-400">{project.description || 'No description'}</p>
              </div>
              <div className="text-right text-xs text-gray-400">
                <p>Created</p>
                <p>{new Date(project.created_at).toLocaleDateString()}</p>
              </div>
            </div>
          </button>
        </li>
      ))}
    </ul>
  );
}
