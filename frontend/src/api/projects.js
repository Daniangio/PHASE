import { requestJSON, requestBlob } from './client';

export function fetchProjects() {
  return requestJSON('/projects');
}

export function fetchProject(projectId) {
  return requestJSON(`/projects/${projectId}`);
}

export function createProject(payload) {
  return requestJSON('/projects', {
    method: 'POST',
    body: payload,
  });
}

export function createSystem(projectId, payload) {
  return requestJSON(`/projects/${projectId}/systems`, {
    method: 'POST',
    body: payload,
  });
}

export function listSystems(projectId) {
  return requestJSON(`/projects/${projectId}/systems`);
}

export function fetchSystem(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}`);
}

export function downloadStructure(projectId, systemId, state) {
  return requestBlob(`/projects/${projectId}/systems/${systemId}/structures/${state}`);
}

export function deleteProject(projectId) {
  return requestJSON(`/projects/${projectId}`, {
    method: 'DELETE',
  });
}

export function deleteSystem(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}`, {
    method: 'DELETE',
  });
}
