import { API_BASE, requestBlob, requestJSON } from './client';

export function submitStaticJob(payload) {
  return requestJSON('/submit/static', {
    method: 'POST',
    body: payload,
  });
}

export function submitSimulationJob(payload) {
  return requestJSON('/submit/simulation', {
    method: 'POST',
    body: payload,
  });
}

export function submitPottsFitJob(payload) {
  return requestJSON('/submit/potts_fit', {
    method: 'POST',
    body: payload,
  });
}

export function submitPottsAnalysisJob(payload) {
  return requestJSON('/submit/potts_analysis', {
    method: 'POST',
    body: payload,
  });
}

export function submitMdSamplesRefreshJob(payload) {
  return requestJSON('/submit/md_samples_refresh', {
    method: 'POST',
    body: payload,
  });
}

export function submitDeltaEvalJob(payload) {
  return requestJSON('/submit/delta_eval', {
    method: 'POST',
    body: payload,
  });
}

export function submitDeltaTransitionJob(payload) {
  return requestJSON('/submit/delta_transition', {
    method: 'POST',
    body: payload,
  });
}

export function submitDeltaCommitmentJob(payload) {
  return requestJSON('/submit/delta_commitment', {
    method: 'POST',
    body: payload,
  });
}

export function submitDeltaJsJob(payload) {
  return requestJSON('/submit/delta_js', {
    method: 'POST',
    body: payload,
  });
}

export function submitLambdaSweepJob(payload) {
  return requestJSON('/submit/lambda_sweep', {
    method: 'POST',
    body: payload,
  });
}

export function submitGibbsRelaxationJob(payload) {
  return requestJSON('/submit/gibbs_relaxation', {
    method: 'POST',
    body: payload,
  });
}

export function fetchResults() {
  return requestJSON('/results');
}

export function fetchResult(jobId) {
  return requestJSON(`/results/${jobId}`);
}

export function deleteResult(jobId) {
  return requestJSON(`/results/${jobId}`, {
    method: 'DELETE',
  });
}

export function fetchJobStatus(jobId) {
  return requestJSON(`/job/status/${jobId}`);
}

export function healthCheck() {
  return requestJSON('/health/check');
}

export function cleanupResults(includeTmp = true) {
  const query = includeTmp ? '?include_tmp=true' : '?include_tmp=false';
  return requestJSON(`/results/cleanup${query}`, { method: 'POST' });
}

export function downloadResultArtifact(jobId, artifact) {
  return requestBlob(`/results/${jobId}/artifacts/${artifact}`);
}

export function resultArtifactUrl(jobId, artifact) {
  return `${API_BASE}/results/${jobId}/artifacts/${artifact}`;
}

export function uploadSimulationResults(
  projectId,
  systemId,
  clusterId,
  compareClusterIds,
  summaryFile,
  pottsModelId,
  sampleName,
  samplingMethod,
  options = {}
) {
  const { onUploadProgress } = options;
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/results/simulation/upload`);
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
          'Failed to upload sampling results.';
        reject(new Error(message));
      }
    };

    xhr.onerror = () => {
      reject(new Error('Network error while uploading sampling results.'));
    };

    const payload = new FormData();
    payload.append('project_id', projectId);
    payload.append('system_id', systemId);
    payload.append('cluster_id', clusterId);
    if (Array.isArray(compareClusterIds)) {
      compareClusterIds.forEach((cid) => {
        if (cid) payload.append('compare_cluster_ids', cid);
      });
    }
    payload.append('summary_npz', summaryFile);
    if (pottsModelId) payload.append('potts_model_id', pottsModelId);
    if (sampleName) payload.append('sample_name', sampleName);
    if (samplingMethod) payload.append('sampling_method', samplingMethod);
    xhr.send(payload);
  });
}
