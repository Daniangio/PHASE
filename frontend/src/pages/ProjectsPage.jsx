import { useEffect, useRef, useState } from 'react';
import { Download, Plus, Upload, X } from 'lucide-react';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import ProjectForm from '../components/projects/ProjectForm';
import ProjectList from '../components/projects/ProjectList';
import SystemForm from '../components/projects/SystemForm';
import SystemList from '../components/projects/SystemList';
import EmptyState from '../components/common/EmptyState';
import {
  fetchProjects,
  downloadProjectsDump,
  restoreProjectsArchive,
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
  const [showProjectForm, setShowProjectForm] = useState(false);
  const [showSystemForm, setShowSystemForm] = useState(false);
  const [isDumping, setIsDumping] = useState(false);
  const [dumpError, setDumpError] = useState(null);
  const [isRestoring, setIsRestoring] = useState(false);
  const [restoreError, setRestoreError] = useState(null);
  const restoreInputRef = useRef(null);

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
    setShowProjectForm(false);
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

  const handleDownloadDump = async () => {
    setIsDumping(true);
    setDumpError(null);
    try {
      const blob = await downloadProjectsDump();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      link.href = url;
      link.download = `projects_dump_${stamp}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      setActionMessage('Projects dump downloaded.');
      setTimeout(() => setActionMessage(null), 4000);
    } catch (err) {
      setDumpError(err.message || 'Failed to download projects dump.');
    } finally {
      setIsDumping(false);
    }
  };

  const handleRestoreClick = () => {
    if (restoreInputRef.current) {
      restoreInputRef.current.click();
    }
  };

  const handleRestoreFile = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setIsRestoring(true);
    setRestoreError(null);
    try {
      await restoreProjectsArchive(file);
      const data = await fetchProjects();
      setProjects(data);
      setSelectedProjectId(data.length ? data[0].project_id : null);
      setActionMessage('Projects restored from archive.');
      setTimeout(() => setActionMessage(null), 4000);
    } catch (err) {
      setRestoreError(err.message || 'Failed to restore projects.');
    } finally {
      setIsRestoring(false);
      event.target.value = '';
    }
  };

  const selectedProject = projects.find((p) => p.project_id === selectedProjectId);

  return (
    <div className="grid lg:grid-cols-[300px_1fr] gap-6">
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Projects</h2>
          <button
            type="button"
            onClick={() => setShowProjectForm(true)}
            className="inline-flex items-center justify-center rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10 p-2"
            aria-label="Create project"
          >
            <Plus className="h-4 w-4" />
          </button>
        </div>
        {isLoadingProjects ? (
          <Loader message="Loading projects..." />
        ) : projects.length === 0 ? (
          <div className="space-y-3">
            <EmptyState
              title="No projects yet"
              description="Create your first project to start building descriptor systems."
            />
            <button
              type="button"
              onClick={() => setShowProjectForm(true)}
              className="w-full text-sm px-3 py-2 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10"
            >
              Create project
            </button>
          </div>
        ) : (
          <ProjectList projects={projects} selectedId={selectedProjectId} onSelect={setSelectedProjectId} />
        )}
      </section>

      <section className="bg-gray-800 border border-gray-700 rounded-lg p-5 space-y-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-gray-500">Project detail</p>
            <h2 className="text-lg font-semibold text-white mt-2">
              {selectedProject ? selectedProject.name : 'Select a project'}
            </h2>
            <p className="text-sm text-gray-400">
              {selectedProject?.description || 'Descriptor systems are grouped per project.'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <input
              ref={restoreInputRef}
              type="file"
              accept=".zip"
              className="hidden"
              onChange={handleRestoreFile}
            />
            <button
              type="button"
              onClick={handleRestoreClick}
              disabled={isRestoring}
              className="inline-flex items-center gap-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 px-3 py-1 text-sm disabled:opacity-60"
            >
              <Upload className="h-4 w-4" />
              {isRestoring ? 'Uploading...' : 'Restore projects'}
            </button>
            <button
              type="button"
              onClick={handleDownloadDump}
              disabled={isDumping}
              className="inline-flex items-center gap-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60 px-3 py-1 text-sm disabled:opacity-60"
            >
              <Download className="h-4 w-4" />
              {isDumping ? 'Preparing...' : 'Dump projects'}
            </button>
            {selectedProject && (
              <>
                <button
                  type="button"
                  onClick={() => setShowSystemForm(true)}
                  className="inline-flex items-center justify-center rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10 p-2"
                  aria-label="Create system"
                >
                  <Plus className="h-4 w-4" />
                </button>
                <button
                  onClick={handleDeleteProject}
                  className="text-sm px-3 py-1 rounded-md border border-red-500 text-red-300 hover:bg-red-500/10"
                >
                  Delete Project
                </button>
              </>
            )}
          </div>
        </div>

        {error && <ErrorMessage message={error} />}
        {dumpError && <ErrorMessage message={dumpError} />}
        {restoreError && <ErrorMessage message={restoreError} />}
        {actionMessage && <p className="text-sm text-emerald-400">{actionMessage}</p>}

        <div>
          <div className="flex items-center justify-between">
            <h3 className="text-md font-semibold text-white">Systems</h3>
            {selectedProject && (
              <button
                type="button"
                onClick={() => setShowSystemForm(true)}
                className="text-xs text-cyan-300 hover:text-cyan-200"
              >
                New system
              </button>
            )}
          </div>
          <div className="mt-3">
            {isLoadingSystems ? (
              <Loader message="Loading systems..." />
            ) : (
              <SystemList projectId={selectedProjectId} systems={systems} onDelete={handleDeleteSystem} />
            )}
          </div>
        </div>
      </section>

      {showProjectForm && (
        <Overlay title="Create project" onClose={() => setShowProjectForm(false)}>
          <ProjectForm onCreate={handleProjectCreated} />
        </Overlay>
      )}

      {showSystemForm && (
        <Overlay title="Create system" onClose={() => setShowSystemForm(false)}>
          {selectedProjectId ? (
            <SystemForm
              onCreate={async (formData, options) => {
                await createSystem(selectedProjectId, formData, options);
                const data = await listSystems(selectedProjectId);
                setSystems(data);
                setShowSystemForm(false);
              }}
            />
          ) : (
            <EmptyState title="Select a project" description="Choose a project to attach the new system to." />
          )}
        </Overlay>
      )}
    </div>
  );
}

function Overlay({ title, onClose, children }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4 py-8">
      <div className="w-full max-w-3xl bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-lg font-semibold text-white">{title}</h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}
