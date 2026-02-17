import { requestJSON, requestBlob, requestBlobWithBody, API_BASE } from './client';

export function fetchProjects() {
  return requestJSON('/projects');
}

export function downloadProjectsDump() {
  return requestBlob('/projects/dump');
}

export function restoreProjectsArchive(file) {
  const formData = new FormData();
  formData.append('archive', file);
  return requestJSON('/projects/restore', {
    method: 'POST',
    body: formData,
  });
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

export function fetchSamplingSummary(projectId, systemId, clusterId, sampleId) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/samples/${sampleId}/summary`
  );
}

export function fetchPottsClusterInfo(projectId, systemId, clusterId, options = {}) {
  const { modelId } = options;
  const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/potts/cluster_info${query}`
  );
}

export function fetchClusterAnalyses(projectId, systemId, clusterId, options = {}) {
  const { analysisType } = options;
  const query = analysisType ? `?analysis_type=${encodeURIComponent(analysisType)}` : '';
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/analyses${query}`);
}

export function fetchClusterPatches(projectId, systemId, clusterId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/patches`);
}

export function createClusterPatch(projectId, systemId, clusterId, payload) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/patches`, {
    method: 'POST',
    body: payload,
  });
}

export function confirmClusterPatch(projectId, systemId, clusterId, patchId) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/patches/${encodeURIComponent(
      patchId
    )}/confirm`,
    { method: 'POST' }
  );
}

export function discardClusterPatch(projectId, systemId, clusterId, patchId) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/patches/${encodeURIComponent(
      patchId
    )}`,
    { method: 'DELETE' }
  );
}

export function fetchClusterAnalysisData(projectId, systemId, clusterId, analysisType, analysisId) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/analyses/${analysisType}/${analysisId}/data`
  );
}

export function fetchSampleStats(projectId, systemId, clusterId, sampleId) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/samples/${sampleId}/stats`
  );
}

export function fetchSampleResidueProfile(projectId, systemId, clusterId, sampleId, payload) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/samples/${sampleId}/residue_profile`,
    {
      method: 'POST',
      body: payload,
    }
  );
}

export function deleteSamplingSample(projectId, systemId, clusterId, sampleId) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/samples/${sampleId}`,
    { method: 'DELETE' }
  );
}

export function assignClusterStates(projectId, systemId, clusterId, stateIds = []) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/assign_states`,
    {
      method: 'POST',
      body: { state_ids: Array.isArray(stateIds) ? stateIds : [] },
    }
  );
}

export function downloadStructure(projectId, systemId, stateId) {
  return requestBlob(`/projects/${projectId}/systems/${systemId}/structures/${stateId}`);
}

export function downloadStateDescriptors(projectId, systemId, stateId) {
  return requestBlob(`/projects/${projectId}/systems/${systemId}/states/${stateId}/descriptors/npz`);
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

export function rescanSystemStates(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/states/rescan`, {
    method: 'POST',
  });
}

export function unlockMacroStateEditing(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/states/unlock-editing`, {
    method: 'POST',
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
  if (params.cluster_label_mode) qs.set('cluster_label_mode', params.cluster_label_mode);
  if (params.cluster_variant_id) qs.set('cluster_variant_id', params.cluster_variant_id);
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

export function setMacroOnlyAnalysis(projectId, systemId) {
  return requestJSON(`/projects/${projectId}/systems/${systemId}/analysis/macro`, {
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

export function downloadMetastableClusters(projectId, systemId, stateIds, params = {}) {
  const payload = {
    state_ids: stateIds,
  };
  if (params.cluster_name) payload.cluster_name = params.cluster_name;
  if (params.max_cluster_frames) payload.max_cluster_frames = params.max_cluster_frames;
  if (params.random_state !== undefined) payload.random_state = params.random_state;
  if (params.density_maxk) payload.density_maxk = params.density_maxk;
  if (params.density_z !== undefined) payload.density_z = params.density_z;
  return requestBlobWithBody(
    `/projects/${projectId}/systems/${systemId}/metastable/cluster_vectors`,
    {
      method: 'POST',
      body: payload,
    }
  );
}

export function submitMetastableClusterJob(projectId, systemId, stateIds, params = {}) {
  const payload = {
    state_ids: stateIds,
  };
  if (params.cluster_name) payload.cluster_name = params.cluster_name;
  if (params.max_cluster_frames) payload.max_cluster_frames = params.max_cluster_frames;
  if (params.random_state !== undefined) payload.random_state = params.random_state;
  if (params.density_maxk) payload.density_maxk = params.density_maxk;
  if (params.density_z !== undefined) payload.density_z = params.density_z;
  return requestJSON(`/projects/${projectId}/systems/${systemId}/metastable/cluster_jobs`, {
    method: 'POST',
    body: payload,
  });
}

export function downloadBackmappingCluster(projectId, systemId, clusterId, options = {}) {
  const { onProgress } = options;
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(
      'GET',
      `${API_BASE}/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/backmapping_npz`
    );
    xhr.responseType = 'blob';

    xhr.addEventListener('progress', (event) => {
      if (!event.lengthComputable || typeof onProgress !== 'function') return;
      const percent = Math.round((event.loaded / event.total) * 100);
      onProgress(Math.min(100, percent));
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response);
        return;
      }
      reject(new Error(xhr.statusText || 'Failed to download backmapping NPZ.'));
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Failed to download backmapping NPZ.'));
    });

    xhr.send();
  });
}

export function submitBackmappingClusterJob(projectId, systemId, clusterId) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/backmapping_npz/job`,
    {
      method: 'POST',
    }
  );
}

export function uploadBackmappingTrajectories(projectId, systemId, clusterId, payload, options = {}) {
  const { onProgress } = options;
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(
      'POST',
      `${API_BASE}/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/backmapping_npz/upload`
    );
    xhr.responseType = 'blob';

    if (xhr.upload && typeof onProgress === 'function') {
      xhr.upload.addEventListener('loadstart', () => onProgress(0));
      xhr.upload.addEventListener('progress', (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.round((event.loaded / event.total) * 100);
        onProgress(Math.min(100, percent));
      });
      xhr.upload.addEventListener('loadend', () => onProgress(100));
    }

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response);
        return;
      }
      reject(new Error(xhr.statusText || 'Failed to build backmapping NPZ.'));
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Failed to build backmapping NPZ.'));
    });

    xhr.send(payload);
  });
}

export function uploadMetastableClusterNp(projectId, systemId, payload, options = {}) {
  const { onUploadProgress } = options;
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/projects/${projectId}/systems/${systemId}/metastable/clusters/upload`);
    xhr.responseType = 'json';

    if (xhr.upload && typeof onUploadProgress === 'function') {
      xhr.upload.addEventListener('loadstart', () => onUploadProgress(0));
      xhr.upload.addEventListener('progress', (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.round((event.loaded / event.total) * 100);
        onUploadProgress(Math.min(100, percent));
      });
      xhr.upload.addEventListener('loadend', () => onUploadProgress(100));
    }

    const parseResponseJSON = () => {
      if (xhr.response !== null && xhr.response !== undefined) return xhr.response;
      try {
        return xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch (err) {
        return null;
      }
    };

    const handleError = () => {
      const response = parseResponseJSON() || {};
      const message = response.detail || response.error || xhr.statusText || 'Upload failed';
      reject(new Error(message));
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parseResponseJSON());
      } else {
        handleError();
      }
    };
    xhr.onerror = handleError;
    xhr.onabort = handleError;
    xhr.send(payload);
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

export function uploadPottsModel(projectId, systemId, clusterId, file, options = {}) {
  const { onUploadProgress } = options;
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(
      'POST',
      `${API_BASE}/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/potts_models`
    );
    xhr.responseType = 'json';

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
      });
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

    xhr.onreadystatechange = () => {
      if (xhr.readyState !== XMLHttpRequest.DONE) return;
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parseResponseJSON());
      } else {
        const response = parseResponseJSON() || {};
        const message =
          response.detail ||
          response.error ||
          (typeof response === 'string' ? response : '') ||
          xhr.statusText ||
          'Failed to upload Potts model.';
        reject(new Error(message));
      }
    };

    xhr.onerror = () => {
      reject(new Error('Network error while uploading Potts model.'));
    };

    const payload = new FormData();
    payload.append('model', file);
    xhr.send(payload);
  });
}

export function downloadPottsModel(projectId, systemId, clusterId, modelId) {
  return requestBlob(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/potts_models/${modelId}`
  );
}

export function renamePottsModel(projectId, systemId, clusterId, modelId, name) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/potts_models/${modelId}`,
    {
      method: 'PATCH',
      body: { name },
    }
  );
}

export function deletePottsModel(projectId, systemId, clusterId, modelId) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/potts_models/${modelId}`,
    {
      method: 'DELETE',
    }
  );
}

export function createLambdaPottsModel(projectId, systemId, clusterId, payload) {
  return requestJSON(
    `/projects/${projectId}/systems/${systemId}/metastable/clusters/${clusterId}/potts_models/lambda`,
    {
      method: 'POST',
      body: payload,
    }
  );
}
