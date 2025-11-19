import { requestJSON } from './client';

export function submitStaticJob(payload) {
  return requestJSON('/submit/static', {
    method: 'POST',
    body: payload,
  });
}

export function submitDynamicJob(payload) {
  return requestJSON('/submit/dynamic', {
    method: 'POST',
    body: payload,
  });
}

export function submitQuboJob(payload) {
  return requestJSON('/submit/qubo', {
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
