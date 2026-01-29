export const formatPottsModelName = (run) => {
  const raw = run?.name || (run?.path ? run.path.split('/').pop() : '') || 'Potts model';
  return raw.replace(/\.npz$/i, '');
};
