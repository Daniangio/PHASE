const API_BASE = '/api/v1';

const defaultHeaders = {
  Accept: 'application/json',
};

async function parseError(response) {
  try {
    const data = await response.json();
    return data.detail || data.error || JSON.stringify(data);
  } catch (err) {
    return response.statusText || 'Unknown error';
  }
}

export async function requestJSON(path, { method = 'GET', body, headers = {} } = {}) {
  const isFormData = body instanceof FormData;
  const headersToUse = isFormData
    ? { ...defaultHeaders, ...headers }
    : { ...defaultHeaders, 'Content-Type': 'application/json', ...headers };

  const options = {
    method,
    headers: headersToUse,
    body: undefined,
  };

  if (body !== undefined) {
    options.body = isFormData ? body : JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

export async function requestBlob(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const message = await parseError(response);
    throw new Error(message);
  }
  return response.blob();
}
