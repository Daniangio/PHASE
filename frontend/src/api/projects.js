import { requestJSON, requestBlob, requestBlobWithBody, API_BASE } from './client';

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


export function renameState(projectId, systemId, stateId, name) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/states/${stateId}`, {
    method: 'PATCH',
    body: { name },
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
  if (params.metastable_ids && params.metastable_ids.length) {
    qs.set('metastable_ids', params.metastable_ids.join(','));
  }
  if (params.cluster_id) qs.set('cluster_id', params.cluster_id);
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

export function clearMetastableStates(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/clear`, {
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

export function downloadMetastableClusters(projectId, systemId, metastableIds, params = {}) {
  const payload = {
    metastable_ids: metastableIds,
  };
  if (params.cluster_name) payload.cluster_name = params.cluster_name;
  if (params.max_clusters_per_residue) payload.max_clusters_per_residue = params.max_clusters_per_residue;
  if (params.max_cluster_frames) payload.max_cluster_frames = params.max_cluster_frames;
  if (params.random_state !== undefined) payload.random_state = params.random_state;
  if (params.contact_atom_mode) payload.contact_atom_mode = params.contact_atom_mode;
  if (params.contact_cutoff) payload.contact_cutoff = params.contact_cutoff;
  if (params.cluster_algorithm) payload.cluster_algorithm = params.cluster_algorithm;
  if (params.algorithm_params) payload.algorithm_params = params.algorithm_params;
  if (params.dbscan_eps) payload.dbscan_eps = params.dbscan_eps;
  if (params.dbscan_min_samples) payload.dbscan_min_samples = params.dbscan_min_samples;
  return requestBlobWithBody(
    `/projects/${projectId}/systems/${systemId}/metastable/cluster_vectors`,
    {
      method: 'POST',
      body: payload,
    }
  );
}

export function submitMetastableClusterJob(projectId, systemId, metastableIds, params = {}) {
  const payload = {
    metastable_ids: metastableIds,
  };
  if (params.cluster_name) payload.cluster_name = params.cluster_name;
  if (params.max_clusters_per_residue) payload.max_clusters_per_residue = params.max_clusters_per_residue;
  if (params.max_cluster_frames) payload.max_cluster_frames = params.max_cluster_frames;
  if (params.random_state !== undefined) payload.random_state = params.random_state;
  if (params.contact_atom_mode) payload.contact_atom_mode = params.contact_atom_mode;
  if (params.contact_cutoff) payload.contact_cutoff = params.contact_cutoff;
  if (params.cluster_algorithm) payload.cluster_algorithm = params.cluster_algorithm;
  if (params.algorithm_params) payload.algorithm_params = params.algorithm_params;
  if (params.dbscan_eps) payload.dbscan_eps = params.dbscan_eps;
  if (params.dbscan_min_samples) payload.dbscan_min_samples = params.dbscan_min_samples;
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/cluster_jobs`, {
    method: 'POST',
    body: payload,
  });
}

export function renameMetastableCluster(projectId, systemId, clusterId, name) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}`, {
    method: 'PATCH',
    body: { name },
  });
}

export function confirmMacroStates(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/states/confirm`, { method: 'POST' });
}

export function confirmMetastableStates(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/confirm`, { method: 'POST' });
}

export function downloadSavedCluster(projectId, systemId, clusterId) {
  return requestBlob(`/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}`);
}

export function deleteSavedCluster(projectId, systemId, clusterId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}`, {
    method: 'DELETE',
  });
}
