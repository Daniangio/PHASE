function asList(value) {
  if (Array.isArray(value)) return value.map((x) => String(x || '').trim()).filter(Boolean);
  if (typeof value === 'string') return value.split(',').map((x) => x.trim()).filter(Boolean);
  return [];
}

function shortText(value, max = 42) {
  const s = String(value || '').trim();
  if (!s) return '';
  return s.length > max ? `${s.slice(0, Math.max(1, max - 1))}…` : s;
}

function labelSample(id, sampleNameById) {
  const sid = String(id || '').trim();
  if (!sid) return '';
  return sampleNameById?.get?.(sid) || sid.slice(0, 8);
}

function formatSampleList(ids, sampleNameById, maxItems = 2) {
  const values = asList(ids);
  if (!values.length) return '';
  const shown = values.slice(0, maxItems).map((sid) => shortText(labelSample(sid, sampleNameById), 28));
  const suffix = values.length > maxItems ? ` +${values.length - maxItems}` : '';
  return `${shown.join(', ')}${suffix}`;
}

export function makeSampleNameById(samples) {
  const map = new Map();
  (Array.isArray(samples) ? samples : []).forEach((sample) => {
    const id = String(sample?.sample_id || '').trim();
    if (!id) return;
    const state = sample?.state_id ? ` (${sample.state_id})` : '';
    const type = sample?.type ? `${sample.type}: ` : '';
    map.set(id, String(sample?.name || sample?.label || `${type}${id.slice(0, 8)}${state}`));
  });
  return map;
}

export function formatDeltaJsAnalysisName(entry, sampleNameById) {
  const refA = formatSampleList(entry?.reference_sample_ids_a, sampleNameById);
  const refB = formatSampleList(entry?.reference_sample_ids_b, sampleNameById);
  if (refA || refB) {
    return `A endpoint: ${refA || 'n/a'} | B endpoint: ${refB || 'n/a'}`;
  }
  if (entry?.model_a_id || entry?.model_b_id) {
    return `${shortText(entry?.model_a_name || entry?.model_a_id || 'Model A', 34)} vs ${shortText(entry?.model_b_name || entry?.model_b_id || 'Model B', 34)}`;
  }
  const edge = String(entry?.edge_source || entry?.edge_mode || 'cluster');
  return `Delta JS · edge=${edge}`;
}

export function formatDeltaJsAnalysisDetails(entry) {
  const ts = String(entry?.updated_at || entry?.created_at || '').slice(0, 19) || 'n/a';
  const n = Number(entry?.summary?.n_samples || 0);
  const md = String(entry?.md_label_mode || 'assigned');
  const invalid = Boolean(entry?.drop_invalid) ? 'drop invalid' : 'keep invalid';
  const edge = String(entry?.edge_source || entry?.edge_mode || 'cluster');
  const model = entry?.model_a_id || entry?.model_b_id
    ? ` · ${(entry?.model_a_name || entry?.model_a_id || 'A')} vs ${(entry?.model_b_name || entry?.model_b_id || 'B')}`
    : '';
  return `${ts} · n=${n} · edge=${edge}${model} · ${md} · ${invalid}`;
}
