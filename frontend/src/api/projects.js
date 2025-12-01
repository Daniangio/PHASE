import { requestJSON, requestBlob, API_BASE } from './client';

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

export function createSystem(projectId, payload, options = {}) {
  const { onUploadProgress, onProcessing } = options;

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/projects/${projectId}/systems`);
    xhr.responseType = 'json';

    const signalProcessing = (isProcessing) => {
      if (typeof onProcessing === 'function') {
        onProcessing(isProcessing);
      }
    };

    if (xhr.upload) {
      xhr.upload.addEventListener('loadstart', () => {
        if (typeof onUploadProgress === 'function') {
          onUploadProgress(0);
        }
      });
      xhr.upload.addEventListener('progress', (event) => {
        if (typeof onUploadProgress !== 'function') return;
        if (!event.lengthComputable) return;
        const percent = Math.round((event.loaded / event.total) * 100);
        onUploadProgress(Math.min(100, percent));
      });
      xhr.upload.addEventListener('loadend', () => {
        if (typeof onUploadProgress === 'function') {
          onUploadProgress(100);
        }
        signalProcessing(true);
      });
    } else {
      signalProcessing(true);
    }

    const parseResponseJSON = () => {
      if (xhr.response !== null && xhr.response !== undefined) {
        return xhr.response;
      }
      try {
        return xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch (err) {
        return null;
      }
    };

    const handleError = () => {
      signalProcessing(false);
      const response = parseResponseJSON() || {};
      const baseMessage =
        response.detail ||
        response.error ||
        (typeof response === 'string' ? response : '') ||
        xhr.statusText ||
        'Failed to create system.';
      const message =
        xhr.status === 413
          ? 'Upload exceeds the 5 GB limit. Please stride or split the files.'
          : baseMessage;
      reject(new Error(message));
    };

    xhr.onreadystatechange = () => {
      if (xhr.readyState !== XMLHttpRequest.DONE) return;
      signalProcessing(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parseResponseJSON());
      } else {
        handleError();
      }
    };

    xhr.onerror = () => {
      signalProcessing(false);
      reject(new Error('Network error while creating system.'));
    };

    xhr.send(payload);
  });
}

export function listSystems(projectId) {
  return requestJSON(`/projects/${projectId}/systems`);
}

export function fetchSystem(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}`);
}

export function downloadStructure(projectId, systemId, stateId) {
  return requestBlob(`/projects/${projectId}/systems/${systemId}/structures/${stateId}`);
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

export function uploadStateTrajectory(projectId, systemId, stateId, payload, options = {}) {
  const { onUploadProgress, onProcessing } = options;

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/projects/${projectId}/systems/${systemId}/states/${stateId}/trajectory`);
    xhr.responseType = 'json';

    const signalProcessing = (isProcessing) => {
      if (typeof onProcessing === 'function') {
        onProcessing(isProcessing);
      }
    };

    if (xhr.upload) {
      xhr.upload.addEventListener('loadstart', () => {
        if (typeof onUploadProgress === 'function') onUploadProgress(0);
      });
      xhr.upload.addEventListener('progress', (event) => {
        if (typeof onUploadProgress !== 'function') return;
        if (!event.lengthComputable) return;
        const percent = Math.round((event.loaded / event.total) * 100);
        onUploadProgress(Math.min(100, percent));
      });
      xhr.upload.addEventListener('loadend', () => {
        if (typeof onUploadProgress === 'function') onUploadProgress(100);
        signalProcessing(true);
      });
    } else {
      signalProcessing(true);
    }

    const parseResponseJSON = () => {
      if (xhr.response !== null && xhr.response !== undefined) {
        return xhr.response;
      }
      try {
        return xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch (err) {
        return null;
      }
    };

    const handleError = () => {
      signalProcessing(false);
      const response = parseResponseJSON() || {};
      const baseMessage =
        response.detail ||
        response.error ||
        (typeof response === 'string' ? response : '') ||
        xhr.statusText ||
        'Failed to upload trajectory.';
      const message =
        xhr.status === 413
          ? 'Trajectory upload exceeds the 5 GB limit. Please stride or split the file.'
          : baseMessage;
      reject(new Error(message));
    };

    xhr.onreadystatechange = () => {
      if (xhr.readyState !== XMLHttpRequest.DONE) return;
      signalProcessing(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parseResponseJSON());
      } else {
        handleError();
      }
    };

    xhr.onerror = () => {
      signalProcessing(false);
      reject(new Error('Network error while uploading trajectory.'));
    };

    xhr.send(payload);
  });
}

export function deleteStateTrajectory(projectId, systemId, stateId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/states/${stateId}/trajectory`, {
    method: 'DELETE',
  });
}

export function addSystemState(projectId, systemId, payload) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/states`, {
    method: 'POST',
    body: payload,
  });
}

export function deleteState(projectId, systemId, stateId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/states/${stateId}`, {
    method: 'DELETE',
  });
}

export function fetchStateDescriptors(projectId, systemId, stateId, params = {}) {
  const qs = new URLSearchParams();
  if (params.residue_keys) qs.set('residue_keys', params.residue_keys);
  if (params.max_points) qs.set('max_points', params.max_points);
  const suffix = qs.toString() ? `?${qs.toString()}` : '';
  return requestJSON(`/projects/${projectId}/systems/${systemId}/states/${stateId}/descriptors${suffix}`);
}

export function fetchMetastableStates(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable`);
}

export function recomputeMetastableStates(projectId, systemId, params = {}) {
  const qs = new URLSearchParams();
  if (params.n_microstates) qs.set('n_microstates', params.n_microstates);
  if (params.k_meta_min) qs.set('k_meta_min', params.k_meta_min);
  if (params.k_meta_max) qs.set('k_meta_max', params.k_meta_max);
  if (params.tica_lag_frames) qs.set('tica_lag_frames', params.tica_lag_frames);
  if (params.tica_dim) qs.set('tica_dim', params.tica_dim);
  if (params.random_state !== undefined) qs.set('random_state', params.random_state);
  const suffix = qs.toString() ? `?${qs.toString()}` : '';
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/recompute${suffix}`, {
    method: 'POST',
  });
}

export function renameMetastableState(projectId, systemId, metastableId, name) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/${encodeURIComponent(metastableId)}`, {
    method: 'PATCH',
    body: { name },
  });
}

export function metastablePdbUrl(projectId, systemId, metastableId) {
  return `${API_BASE}/projects/${projectId}/systems/${systemId}/metastable/${encodeURIComponent(
    metastableId
  )}/pdb`;
}
