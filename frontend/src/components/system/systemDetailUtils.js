export function getClusterDisplayName(run) {
  if (!run) return 'Cluster';
  return run.name || run.path?.split('/').pop() || run.cluster_id || 'Cluster';
}

export function getArtifactDisplayName(pathValue) {
  if (typeof pathValue !== 'string' || !pathValue) return '—';
  const parts = pathValue.split('/');
  return parts[parts.length - 1] || pathValue;
}

export function formatClusterAlgorithm(run) {
  if (!run) return '';
  const algo = (run.cluster_algorithm || 'density_peaks').toLowerCase();
  if (algo !== 'density_peaks') return algo;
  const params = run.algorithm_params || {};
  const maxk = params.density_maxk ?? params.maxk ?? '—';
  const zVal = params.density_z ?? params.Z ?? 2.0;
  return `density_peaks (maxk=${maxk}, Z=${zVal})`;
}
