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

export function downloadResultArtifact(jobId, artifact) {
  return requestBlob(`/results/${jobId}/artifacts/${artifact}`);
}

export function resultArtifactUrl(jobId, artifact) {
  return `${API_BASE}/results/${jobId}/artifacts/${artifact}`;
}
