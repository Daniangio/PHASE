import { useEffect, useState } from 'react';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import ProjectForm from '../components/projects/ProjectForm';
import ProjectList from '../components/projects/ProjectList';
import SystemForm from '../components/projects/SystemForm';
import SystemList from '../components/projects/SystemList';
import EmptyState from '../components/common/EmptyState';
import {
  fetchProjects,
  createProject,
  listSystems,
  createSystem,
  deleteProject,
  deleteSystem,
} from '../api/projects';

export default function ProjectsPage() {
  const [projects, setProjects] = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState(null);
  const [systems, setSystems] = useState([]);
  const [isLoadingProjects, setLoadingProjects] = useState(true);
  const [isLoadingSystems, setLoadingSystems] = useState(false);
  const [error, setError] = useState(null);
  const [actionMessage, setActionMessage] = useState(null);

  useEffect(() => {
    const loadProjects = async () => {
      setLoadingProjects(true);
      setError(null);
      try {
        const data = await fetchProjects();
        setProjects(data);
        if (data.length > 0) {
          setSelectedProjectId(data[0].project_id);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingProjects(false);
      }
    };
    loadProjects();
  }, []);

  useEffect(() => {
    if (!selectedProjectId) {
      setSystems([]);
      return;
    }
    const loadSystems = async () => {
      setLoadingSystems(true);
      setError(null);
      try {
        const data = await listSystems(selectedProjectId);
        setSystems(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingSystems(false);
      }
    };
    loadSystems();
  }, [selectedProjectId]);

  const handleProjectCreated = async (payload) => {
    const newProject = await createProject(payload);
    setProjects((prev) => [newProject, ...prev]);
    setSelectedProjectId(newProject.project_id);
  };

  const handleDeleteProject = async () => {
    if (!selectedProjectId) return;
    const project = projects.find((p) => p.project_id === selectedProjectId);
    const label = project?.name || selectedProjectId;
    if (!window.confirm(`Delete project "${label}" and all of its systems? This cannot be undone.`)) {
      return;
    }
    try {
      await deleteProject(selectedProjectId);
      const remaining = projects.filter((p) => p.project_id !== selectedProjectId);
      setProjects(remaining);
      const nextSelection = remaining.length ? remaining[0].project_id : null;
      setSelectedProjectId(nextSelection);
      setSystems([]);
      setActionMessage(`Project "${label}" deleted.`);
      setTimeout(() => setActionMessage(null), 4000);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleDeleteSystem = async (systemId) => {
    if (!selectedProjectId) return;
    const system = systems.find((s) => s.system_id === systemId);
    const label = system?.name || systemId;
    if (!window.confirm(`Delete system "${label}"? This will remove stored descriptors and structures.`)) {
      return;
    }
    try {
      await deleteSystem(selectedProjectId, systemId);
      const updated = systems.filter((s) => s.system_id !== systemId);
      setSystems(updated);
      setActionMessage(`System "${label}" deleted.`);
      setTimeout(() => setActionMessage(null), 4000);
    } catch (err) {
      setError(err.message);
    }
  };

  const selectedProject = projects.find((p) => p.project_id === selectedProjectId);

  return (
    <div className="grid lg:grid-cols-3 gap-8">
      <section className="lg:col-span-1 space-y-4">
        <div>
          <h2 className="text-lg font-semibold text-white mb-2">Projects</h2>
          {isLoadingProjects ? (
            <Loader message="Loading projects..." />
          ) : projects.length === 0 ? (
            <EmptyState
              title="No projects yet"
              description="Create your first project to start building descriptor systems."
            />
          ) : (
            <ProjectList projects={projects} selectedId={selectedProjectId} onSelect={setSelectedProjectId} />
          )}
        </div>
        <div>
          <h3 className="text-md font-semibold text-white mb-2">Create Project</h3>
          <ProjectForm onCreate={handleProjectCreated} />
        </div>
      </section>

      <section className="lg:col-span-2 space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h2 className="text-lg font-semibold text-white">
              {selectedProject ? selectedProject.name : 'Select a project'}
            </h2>
            <p className="text-sm text-gray-400">
              {selectedProject?.description || 'Descriptor systems are grouped per project.'}
            </p>
          </div>
          {selectedProject && (
            <button
              onClick={handleDeleteProject}
              className="text-sm px-3 py-1 rounded-md border border-red-500 text-red-300 hover:bg-red-500/10"
            >
              Delete Project
            </button>
          )}
        </div>

        {error && <ErrorMessage message={error} />}
        {actionMessage && <p className="text-sm text-emerald-400">{actionMessage}</p>}

        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <h3 className="text-md font-semibold text-white mb-2">Systems</h3>
            {isLoadingSystems ? (
              <Loader message="Loading systems..." />
            ) : (
              <SystemList projectId={selectedProjectId} systems={systems} onDelete={handleDeleteSystem} />
            )}
          </div>
          <div>
            <h3 className="text-md font-semibold text-white mb-2">New System</h3>
            {selectedProjectId ? (
              <SystemForm
                onCreate={async (formData) => {
                  await createSystem(selectedProjectId, formData);
                  const data = await listSystems(selectedProjectId);
                  setSystems(data);
                }}
              />
            ) : (
              <EmptyState title="Select a project" description="Choose a project to attach the new system to." />
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
