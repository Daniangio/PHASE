import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchSystem, fetchStateDescriptors } from '../api/projects';

const colors = [
  '#22d3ee',
  '#a855f7',
  '#f97316',
  '#10b981',
  '#f43f5e',
  '#8b5cf6',
  '#06b6d4',
  '#fde047',
  '#60a5fa',
  '#f59e0b',
];

export default function DescriptorVizPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [error, setError] = useState(null);

  const [selectedStates, setSelectedStates] = useState([]);
  const [residueFilter, setResidueFilter] = useState('');
  const [selectedResidue, setSelectedResidue] = useState('');
  const [residueOptions, setResidueOptions] = useState([]);
  const [residueLabelCache, setResidueLabelCache] = useState({});

  const sortResidues = useCallback((keys) => {
    const unique = Array.from(new Set(keys || [])).filter((k) => k.startsWith('res_'));
    return unique.sort((a, b) => {
      const pa = parseInt((a.split('_')[1] || '').replace(/\D+/g, ''), 10);
      const pb = parseInt((b.split('_')[1] || '').replace(/\D+/g, ''), 10);
      if (Number.isFinite(pa) && Number.isFinite(pb) && pa !== pb) {
        return pa - pb;
      }
      return a.localeCompare(b);
    });
  }, []);
  const [maxPoints, setMaxPoints] = useState(2000);
  const [appliedStateQuery, setAppliedStateQuery] = useState('');
  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [clusterLegend, setClusterLegend] = useState([]);
  const [clusterLabelMode, setClusterLabelMode] = useState('halo');
  const [haloSummary, setHaloSummary] = useState(null);
  const [selectedHaloCondition, setSelectedHaloCondition] = useState('');

  const [anglesByState, setAnglesByState] = useState({});
  const [metaByState, setMetaByState] = useState({});
  const [loadingAngles, setLoadingAngles] = useState(false);
  const [anglesError, setAnglesError] = useState(null);

  useEffect(() => {
    const loadSystem = async () => {
      setLoadingSystem(true);
      setError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
        const descriptorStates = Object.values(data.states || {}).filter((s) => s.descriptor_file);
        if (descriptorStates.length) {
          setSelectedStates((prev) => (prev.length ? prev : descriptorStates.map((s) => s.state_id)));
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingSystem(false);
      }
    };
    loadSystem();
  }, [projectId, systemId]);

  const descriptorStates = useMemo(
    () => Object.values(system?.states || {}).filter((s) => s.descriptor_file),
    [system]
  );
  const metastableStates = useMemo(() => system?.metastable_states || [], [system]);
  const clusterOptions = useMemo(
    () => (system?.metastable_clusters || []).filter((run) => run.path && run.status !== 'failed'),
    [system]
  );

  // Hydrate from query params (cluster selection) whenever search changes.
  useEffect(() => {
    const params = new URLSearchParams(location.search || '');
    const clusterId = params.get('cluster_id');
    if (clusterId) {
      setSelectedClusterId(clusterId);
    }
  }, [location.search]);

  // Hydrate initial macro-state selection from query params.
  useEffect(() => {
    if (!system) return;
    const params = new URLSearchParams(location.search || '');
    const stateParam = params.getAll('state_id').filter(Boolean);
    const stateIdsParam = params.get('state_ids');
    const queryKey = [stateIdsParam || '', ...stateParam].join('|');
    if (!stateParam.length && !stateIdsParam) return;
    if (queryKey === appliedStateQuery) return;

    const collected = [...stateParam];
    if (stateIdsParam) {
      if (stateIdsParam.trim().toLowerCase() === 'all') {
        setSelectedStates(descriptorStates.map((s) => s.state_id));
        setAppliedStateQuery(queryKey);
        return;
      }
      stateIdsParam
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .forEach((id) => collected.push(id));
    }
    if (collected.length) {
      const valid = descriptorStates
        .filter((s) => collected.includes(s.state_id))
        .map((s) => s.state_id);
      if (valid.length) {
        setSelectedStates(valid);
      }
    }
    setAppliedStateQuery(queryKey);
  }, [appliedStateQuery, descriptorStates, location.search, system]);


  const residueKeys = useMemo(() => sortResidues(residueOptions), [residueOptions, sortResidues]);

  const residueLabel = useCallback(
    (key) => {
      if (residueLabelCache[key]) return residueLabelCache[key];
      const metasInOrder = [
        ...selectedStates.map((stateId) => metaByState[stateId]).filter(Boolean),
        ...Object.values(metaByState),
      ];

      for (const meta of metasInOrder) {
        const labels = meta?.residue_labels || {};
        if (labels[key]) return labels[key];
        const mapping = meta?.residue_mapping || {};
        if (mapping[key]) {
          const raw = mapping[key] || '';
          const match = raw.match(/\b([A-Z]{3})\b/);
          const resname = match ? match[1].toUpperCase() : null;
          if (resname) return `${key}_${resname}`;
        }
      }

      return key;
    },
    [metaByState, residueLabelCache, selectedStates]
  );

  const filteredResidues = useMemo(() => {
    if (!residueFilter.trim()) return residueKeys;
    const needle = residueFilter.toLowerCase();
    return residueKeys.filter((key) => {
      const label = residueLabel(key).toLowerCase();
      return label.includes(needle);
    });
  }, [residueFilter, residueKeys, residueLabel]);

  const stateName = useCallback(
    (stateId) => descriptorStates.find((s) => s.state_id === stateId)?.name || stateId,
    [descriptorStates]
  );

  const metastableLookupByState = useMemo(() => {
    const mapping = {};
    selectedStates.forEach((stateId) => {
      const legend = (metaByState[stateId]?.metastable_legend || []).length
        ? metaByState[stateId]?.metastable_legend || []
        : metastableStates
            .filter((m) => m.macro_state_id === stateId)
            .map((m) => ({
              id: m.metastable_id,
              index: m.metastable_index,
              label: m.name || m.default_name || m.metastable_id,
            }));
      const perState = {};
      legend.forEach((entry) => {
        if (entry.index === null || entry.index === undefined) return;
        perState[entry.index] = entry.label || entry.id || `Metastable ${entry.index + 1}`;
      });
      mapping[stateId] = perState;
    });
    return mapping;
  }, [metaByState, metastableStates, selectedStates]);

  const stateColors = useMemo(() => {
    const mapping = {};
    selectedStates.forEach((stateId, idx) => {
      mapping[stateId] = colors[idx % colors.length];
    });
    return mapping;
  }, [selectedStates]);

  const clusterColorMap = useMemo(() => {
    const mapping = {};
    clusterLegend.forEach((c, idx) => {
      mapping[c.id] = colors[idx % colors.length];
    });
    return mapping;
  }, [clusterLegend]);

  const clusterLabelLookup = useMemo(() => {
    const mapping = {};
    clusterLegend.forEach((c) => {
      mapping[c.id] = c.label;
    });
    return mapping;
  }, [clusterLegend]);

  const residueSymbols = useMemo(() => {
    const symbols = [
      'circle',
      'square',
      'diamond',
      'cross',
      'triangle-up',
      'triangle-down',
      'triangle-left',
      'triangle-right',
      'x',
      'star',
      'hexagram',
    ];
    const mapping = {};
    residueKeys.forEach((key, idx) => {
      mapping[key] = symbols[idx % symbols.length];
    });
    return mapping;
  }, [residueKeys]);

  const buildGroupedTraces = useCallback(
    (stateId, residueKey, data, axes) => {
      if (!data) return [];
      const xVals = data[axes.xKey] || [];
      const yVals = data[axes.yKey] || [];
      const zVals = axes.zKey ? data[axes.zKey] || [] : [];
      if (!xVals.length || !yVals.length || (axes.zKey && !zVals.length)) return [];

      const macroLabel = stateName(stateId);
      const metaLabels = metaByState[stateId]?.metastable_labels || [];
      const metaLookup = metastableLookupByState[stateId] || {};
      const useMeta =
        Array.isArray(metaLabels) &&
        metaLabels.length === xVals.length &&
        metaLabels.some((v) => Number.isFinite(v) && v >= 0);

      const clusterLabels = data.cluster_labels;
      const clusterColors =
        clusterLabels && clusterLegend.length
          ? clusterLabels.map((c) => clusterColorMap[c] || '#9ca3af')
          : null;
      const clusterHover =
        clusterLabels && clusterLegend.length
          ? clusterLabels.map((c) => (c >= 0 ? clusterLabelLookup[c] || `Cluster ${c}` : 'No cluster'))
          : null;

      const pick = (arr, indices) => indices.map((idx) => arr[idx]);
      const groups = {};
      if (useMeta) {
        metaLabels.forEach((label, idx) => {
          const key = Number.isFinite(label) ? label : -1;
          if (!groups[key]) groups[key] = [];
          groups[key].push(idx);
        });
      } else {
        groups[macroLabel] = Array.from({ length: xVals.length }, (_, i) => i);
      }

      const groupKeys = useMeta
        ? Object.keys(groups)
            .map((k) => Number(k))
            .sort((a, b) => (a === -1 ? 1 : b === -1 ? -1 : a - b))
        : Object.keys(groups);

      return groupKeys.map((groupKey, idx) => {
        const indices = groups[groupKey];
        const metaLabel =
          useMeta && groupKey !== -1
            ? metaLookup[groupKey] || `Metastable ${Number(groupKey) + 1}`
            : useMeta
            ? 'Outliers'
            : macroLabel;
        const traceName = useMeta ? metaLabel : macroLabel;
        const legendgrouptitle = idx === 0 ? { text: macroLabel } : undefined;
        const metaHover = useMeta ? `<br>Metastable: ${metaLabel}` : '';

        return {
          type: axes.zKey ? 'scatter3d' : 'scattergl',
          mode: 'markers',
          x: pick(xVals, indices),
          y: pick(yVals, indices),
          ...(axes.zKey ? { z: pick(zVals, indices) } : {}),
          name: traceName,
          legendgroup: macroLabel,
          legendgrouptitle,
          marker: {
            size: axes.zKey ? 3 : 4,
            opacity: axes.zKey ? 0.75 : 0.7,
            color: clusterColors ? pick(clusterColors, indices) : stateColors[stateId],
            symbol: residueSymbols[residueKey] || 'circle',
          },
          customdata: clusterHover ? pick(clusterHover, indices) : null,
          hovertemplate:
            `Residue: ${residueLabel(residueKey)}<br>State: ${macroLabel}` +
            metaHover +
            (axes.zKey
              ? '<br>Phi: %{x:.2f}°<br>Psi: %{y:.2f}°<br>Chi1: %{z:.2f}°'
              : `<br>${axes.xKey.toUpperCase()}: %{x:.2f}°<br>${axes.yKey.toUpperCase()}: %{y:.2f}°`) +
            (clusterHover ? '<br>Cluster: %{customdata}' : '') +
            '<extra></extra>',
        };
      });
    },
    [
      clusterColorMap,
      clusterLabelLookup,
      clusterLegend,
      metaByState,
      metastableLookupByState,
      residueLabel,
      residueSymbols,
      stateColors,
      stateName,
    ]
  );


  const selectResidue = (key) => {
    setSelectedResidue(key);
  };

  // Preload residue labels/resnames so the list keeps informative names even before a residue is loaded
  useEffect(() => {
    const bootstrapLabels = async () => {
      if (!selectedStates.length) return;
      try {
        const stateId = selectedStates[0];
        const data = await fetchStateDescriptors(projectId, systemId, stateId, { max_points: 1 });
        const labels = data.residue_labels || {};
        const mapping = data.residue_mapping || {};
        const combined = { ...labels };
        Object.entries(mapping).forEach(([k, raw]) => {
          if (combined[k]) return;
          const match = (raw || '').match(/\b([A-Z]{3})\b/);
          const resname = match ? match[1].toUpperCase() : null;
          if (resname) combined[k] = `${k}_${resname}`;
        });
        if (Object.keys(combined).length) {
          setResidueLabelCache((prev) => ({ ...prev, ...combined }));
        }
        if (Array.isArray(data.residue_keys)) {
          setResidueOptions((prev) => {
            const merged = new Set([...(prev || []), ...data.residue_keys]);
            return sortResidues(Array.from(merged));
          });
          if (!selectedResidue && data.residue_keys.length) {
            setSelectedResidue(sortResidues(data.residue_keys)[0]);
          }
        }
      } catch (err) {
        // keep silent; fallback labels will be used
      }
    };
    bootstrapLabels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, systemId, selectedStates, sortResidues]);

  const loadAngles = useCallback(async () => {
    if (!selectedStates.length) {
      setAnglesByState({});
      setMetaByState({});
      setClusterLegend([]);
      setHaloSummary(null);
      setSelectedHaloCondition('');
      setSelectedResidue('');
      return;
    }
    setLoadingAngles(true);
    setAnglesError(null);
    try {
      const bootstrapOnly = !selectedResidue;
      const qs = { max_points: bootstrapOnly ? Math.min(maxPoints, 500) : maxPoints };
      if (selectedClusterId) {
        qs.cluster_id = selectedClusterId;
        qs.cluster_label_mode = clusterLabelMode;
      }
      if (selectedResidue) {
        qs.residue_keys = selectedResidue;
      }

      const responses = await Promise.all(
        selectedStates.map(async (stateId) => {
          const data = await fetchStateDescriptors(projectId, systemId, stateId, qs);
          return { stateId, data };
        })
      );

      const newAngles = {};
      const newMeta = {};
      const unionResidues = new Set();
      let nextHaloSummary = null;

      responses.forEach(({ stateId, data }) => {
        newAngles[stateId] = data.angles || {};
        newMeta[stateId] = {
          residue_keys: data.residue_keys || [],
          residue_mapping: data.residue_mapping || {},
          residue_labels: data.residue_labels || {},
          n_frames: data.n_frames,
          sample_stride: data.sample_stride,
          cluster_legend: data.cluster_legend || [],
          metastable_labels: data.metastable_labels || [],
          metastable_legend: data.metastable_legend || [],
        };
        if (!nextHaloSummary && Array.isArray(data.halo_rate_matrix)) {
          nextHaloSummary = {
            matrix: data.halo_rate_matrix,
            conditionIds: data.halo_rate_condition_ids || [],
            conditionLabels: data.halo_rate_condition_labels || [],
            conditionTypes: data.halo_rate_condition_types || [],
            residueKeys: data.halo_rate_residue_keys || data.residue_keys || [],
          };
        }
        (data.residue_keys || []).forEach((key) => unionResidues.add(key));
        // Cache labels from this response to keep names informative in the list
        const labels = data.residue_labels || {};
        const mapping = data.residue_mapping || {};
        const combined = {};
        Object.entries(labels).forEach(([k, v]) => {
          if (v) combined[k] = v;
        });
        Object.entries(mapping).forEach(([k, raw]) => {
          if (combined[k]) return;
          const match = (raw || '').match(/\b([A-Z]{3})\b/);
          const resname = match ? match[1].toUpperCase() : null;
          if (resname) combined[k] = `${k}_${resname}`;
        });
        if (Object.keys(combined).length) {
          setResidueLabelCache((prev) => ({ ...prev, ...combined }));
        }
      });

      // Cluster legend: use from first response if present
      const firstLegend = responses.find((r) => (r.data.cluster_legend || []).length);
      setClusterLegend(firstLegend ? firstLegend.data.cluster_legend || [] : []);
      setMetaByState(newMeta);
      setHaloSummary(nextHaloSummary);
      if (nextHaloSummary?.conditionIds?.length) {
        setSelectedHaloCondition((prev) =>
          nextHaloSummary.conditionIds.includes(prev) ? prev : nextHaloSummary.conditionIds[0]
        );
      } else {
        setSelectedHaloCondition('');
      }

      const sortedResidues = sortResidues(Array.from(unionResidues));
      setResidueOptions((prev) => {
        const merged = [...(prev || []), ...sortedResidues];
        return sortResidues(merged);
      });
      if (!selectedResidue && sortedResidues.length) {
        setSelectedResidue(sortedResidues[0]);
      } else if (selectedResidue && sortedResidues.length && !unionResidues.has(selectedResidue)) {
        setSelectedResidue(sortedResidues[0]);
      } else if (selectedResidue && !sortedResidues.length) {
        setSelectedResidue('');
      }

      if (!bootstrapOnly && selectedResidue) {
        setAnglesByState(newAngles);
      } else {
        // During bootstrap, only populate residue options; a follow-up call will load the selected residue
        setAnglesByState({});
      }
    } catch (err) {
      setAnglesError(err.message);
    } finally {
      setLoadingAngles(false);
    }
  }, [
    maxPoints,
    clusterLabelMode,
    projectId,
    selectedClusterId,
    selectedResidue,
    selectedStates,
    systemId,
  ]);

  useEffect(() => {
    if (selectedStates.length) {
      loadAngles();
    } else {
      setAnglesByState({});
      setMetaByState({});
      setSelectedResidue('');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStates, selectedClusterId, selectedResidue, clusterLabelMode]);

  const traces3d = useMemo(() => {
    const traces = [];
    const residuesToPlot = selectedResidue ? [selectedResidue] : [];
    selectedStates.forEach((stateId) => {
      const perState = anglesByState[stateId] || {};
      residuesToPlot.forEach((key) => {
        const data = perState[key];
        if (!data) return;
        traces.push(
          ...buildGroupedTraces(stateId, key, data, { xKey: 'phi', yKey: 'psi', zKey: 'chi1' })
        );
      });
    });
    // Add legend for clusters if present
    if (clusterLegend.length) {
      clusterLegend.forEach((c) => {
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: [null],
          y: [null],
          z: [null],
          name: c.label,
          showlegend: true,
          marker: { color: clusterColorMap[c.id] || '#9ca3af' },
          hoverinfo: 'none',
        });
      });
    }
    return traces;
  }, [anglesByState, buildGroupedTraces, clusterColorMap, clusterLegend, selectedResidue, selectedStates]);

  const make2DTraces = useCallback(
    (axisX, axisY) =>
      selectedStates
        .map((stateId) => {
          const perState = anglesByState[stateId] || {};
          const residuesToPlot = selectedResidue ? [selectedResidue] : [];
          return residuesToPlot.map((key) => {
            const data = perState[key];
            if (!data) return null;
            return buildGroupedTraces(stateId, key, data, { xKey: axisX, yKey: axisY });
          });
        })
        .flat(2)
        .filter(Boolean),
    [anglesByState, buildGroupedTraces, selectedResidue, selectedStates]
  );

  const hasAngles = useMemo(
    () =>
      !!selectedResidue &&
      Object.values(anglesByState).some((residues) => Boolean((residues || {})[selectedResidue])),
    [anglesByState, selectedResidue]
  );

  const hasHaloSummary = useMemo(() => {
    const matrix = haloSummary?.matrix;
    return Array.isArray(matrix) && matrix.length > 0;
  }, [haloSummary]);

  const haloConditionOptions = useMemo(() => {
    if (!hasHaloSummary) return [];
    return (haloSummary.conditionIds || []).map((id, idx) => ({
      id,
      label: haloSummary.conditionLabels?.[idx] || id,
      type: haloSummary.conditionTypes?.[idx] || 'condition',
    }));
  }, [haloSummary, hasHaloSummary]);

  const haloResidueLabels = useMemo(() => {
    if (!hasHaloSummary) return [];
    const keys = haloSummary.residueKeys || [];
    return keys.map((key) => residueLabel(key));
  }, [haloSummary, hasHaloSummary, residueLabel]);

  const haloRanking = useMemo(() => {
    if (!hasHaloSummary) return [];
    const idx = (haloSummary.conditionIds || []).indexOf(selectedHaloCondition);
    if (idx < 0) return [];
    const row = haloSummary.matrix?.[idx] || [];
    return (haloSummary.residueKeys || []).map((key, i) => ({
      key,
      label: residueLabel(key),
      value: row?.[i],
    })).filter((entry) => Number.isFinite(entry.value))
      .sort((a, b) => b.value - a.value);
  }, [haloSummary, hasHaloSummary, selectedHaloCondition, residueLabel]);

  const haloHeatmapData = useMemo(() => {
    if (!hasHaloSummary) return [];
    const conditionLabels = haloConditionOptions.map((opt) => {
      const prefix = opt.type === 'metastable' ? 'Metastable' : 'Macro';
      return `${prefix}: ${opt.label}`;
    });
    return [
      {
        type: 'heatmap',
        z: haloSummary.matrix,
        x: haloResidueLabels,
        y: conditionLabels,
        colorscale: 'YlOrRd',
        zmin: 0,
        zmax: 1,
        hovertemplate: 'Condition: %{y}<br>Residue: %{x}<br>Halo rate: %{z:.3f}<extra></extra>',
      },
    ];
  }, [haloConditionOptions, haloResidueLabels, haloSummary, hasHaloSummary]);

  const haloShowResidueTicks = useMemo(() => {
    if (!hasHaloSummary) return false;
    return haloResidueLabels.length <= 60;
  }, [haloResidueLabels, hasHaloSummary]);

  const stateSummaries = useMemo(
    () =>
      selectedStates.map((stateId) => ({
        stateId,
        name: stateName(stateId),
        frames: metaByState[stateId]?.n_frames,
        stride: metaByState[stateId]?.sample_stride,
      })),
    [metaByState, selectedStates, stateName]
  );

  if (loadingSystem) return <Loader message="Loading system..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!system) return null;

  return (
    <div className="space-y-4">
      <button
        onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)}
        className="text-cyan-400 hover:text-cyan-300 text-sm"
      >
        ← Back to system
      </button>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Descriptor Explorer</h1>
          <p className="text-sm text-gray-400">
            Visualize per-residue phi/psi/chi1 angles. Data are down-sampled for plotting.
          </p>
        </div>
        <div className="text-xs text-gray-400 text-right space-y-0.5">
          <div>
            States: {selectedStates.length ? stateSummaries.map((s) => s.name).join(', ') : '—'}
          </div>
          <div>
            Frames:{' '}
            {stateSummaries.length
              ? stateSummaries.map((s) => `${s.name}: ${s.frames ?? '—'}`).join(' • ')
              : '—'}
          </div>
          <div>
            Sample stride:{' '}
            {stateSummaries.length
              ? stateSummaries.map((s) => `${s.name}: ${s.stride ?? '—'}`).join(' • ')
              : '—'}
          </div>
        </div>
      </div>

      {descriptorStates.length === 0 ? (
        <ErrorMessage message="No descriptor-ready states. Upload trajectories and build descriptors first." />
      ) : (
        <div className="lg:grid lg:grid-cols-[minmax(220px,20%)_1fr] gap-4">
          <aside className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">Max points per residue</label>
              <input
                type="number"
                min={10}
                max={50000}
                value={maxPoints}
                onChange={(e) => setMaxPoints(Number(e.target.value) || 2000)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              />
            </div>
            <button
              onClick={loadAngles}
              disabled={loadingAngles || !selectedStates.length}
              className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md disabled:opacity-50"
            >
              {loadingAngles ? 'Loading…' : 'Refresh data'}
            </button>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Cluster NPZ (optional coloring)</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                <option value="">None</option>
                {clusterOptions.map((c) => (
                  <option key={c.cluster_id} value={c.cluster_id}>
                    {c.name || c.path?.split('/').pop() || c.cluster_id}
                  </option>
                ))}
              </select>
              {selectedClusterId && (
                <div className="mt-2 space-y-1">
                  <p className="text-[11px] text-gray-500">Cluster label mode</p>
                  <div className="flex items-center gap-2 text-xs text-gray-300">
                    <label className="flex items-center gap-2">
                      <input
                        type="radio"
                        name="cluster-label-mode"
                        value="halo"
                        checked={clusterLabelMode === 'halo'}
                        onChange={() => setClusterLabelMode('halo')}
                        className="accent-cyan-500"
                      />
                      Halo (-1)
                    </label>
                    <label className="flex items-center gap-2">
                      <input
                        type="radio"
                        name="cluster-label-mode"
                        value="assigned"
                        checked={clusterLabelMode === 'assigned'}
                        onChange={() => setClusterLabelMode('assigned')}
                        className="accent-cyan-500"
                      />
                      Assigned
                    </label>
                  </div>
                </div>
              )}
              {clusterLegend.length > 0 && (
                <p className="text-[11px] text-gray-500 mt-2">
                  Clusters loaded: {clusterLegend.map((c) => c.label).join(' • ')}
                </p>
              )}
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Filter residues</label>
              <input
                type="text"
                value={residueFilter}
                onChange={(e) => setResidueFilter(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                placeholder="Search residue keys"
              />
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-2">Residues</p>
              <div className="space-y-2 max-h-80 overflow-y-auto border border-gray-700 rounded-md p-2 bg-gray-900">
                {filteredResidues.length === 0 && (
                  <p className="text-sm text-gray-500">No residues match this filter.</p>
                )}
                {filteredResidues.map((key) => (
                  <label key={key} className="flex items-center space-x-2 text-sm text-gray-200">
                    <input
                      type="radio"
                      name="residue-select"
                      checked={selectedResidue === key}
                      onChange={() => selectResidue(key)}
                      className="accent-cyan-500"
                    />
                    <span>{residueLabel(key)}</span>
                  </label>
                ))}
              </div>
            </div>
          </aside>

          <section className="space-y-4">
            <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-gray-500">Legend</p>
                <p className="text-[11px] text-gray-400 mt-1">
                  Click legend items to show/hide individual metastables; macro-states are grouped in the legend.
                </p>
              </div>
            </div>

            {anglesError && <ErrorMessage message={anglesError} />}
            {loadingAngles && <Loader message="Loading angles..." />}

            {!loadingAngles && !hasAngles && (
              <p className="text-sm text-gray-400">
                {selectedStates.length
                  ? 'Pick a residue to load and color its angles.'
                  : 'Select at least one state to load descriptor data.'}
              </p>
            )}

            {hasAngles && (
              <div className="space-y-4">
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                  <Plot
                    data={traces3d}
                    layout={{
                      height: 500,
                      paper_bgcolor: '#111827',
                      plot_bgcolor: '#111827',
                      font: { color: '#e5e7eb' },
                      scene: {
                        xaxis: { title: 'Phi (°)', range: [-180, 180] },
                        yaxis: { title: 'Psi (°)', range: [-180, 180] },
                        zaxis: { title: 'Chi1 (°)', range: [-180, 180] },
                        aspectmode: 'cube',
                      },
                      margin: { l: 0, r: 0, t: 10, b: 0 },
                      legend: { bgcolor: 'rgba(0,0,0,0)', groupclick: 'toggleitem', itemdoubleclick: 'toggleothers' },
                    }}
                    useResizeHandler
                    style={{ width: '100%', height: '100%' }}
                    config={{ displaylogo: false, responsive: true }}
                  />
                </div>

                <div className="grid md:grid-cols-3 gap-3">
                  {[
                    { x: 'phi', y: 'psi', title: 'Phi vs Psi' },
                    { x: 'phi', y: 'chi1', title: 'Phi vs Chi1' },
                    { x: 'psi', y: 'chi1', title: 'Psi vs Chi1' },
                  ].map((axes) => (
                    <div key={axes.title} className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                      <Plot
                        data={make2DTraces(axes.x, axes.y)}
                        layout={{
                          height: 350,
                          paper_bgcolor: '#111827',
                          plot_bgcolor: '#111827',
                          font: { color: '#e5e7eb' },
                          margin: { l: 40, r: 10, t: 30, b: 40 },
                          xaxis: { title: `${axes.x.toUpperCase()} (°)`, range: [-180, 180] },
                          yaxis: { title: `${axes.y.toUpperCase()} (°)`, range: [-180, 180] },
                          legend: { bgcolor: 'rgba(0,0,0,0)', groupclick: 'toggleitem', itemdoubleclick: 'toggleothers' },
                        }}
                        useResizeHandler
                        style={{ width: '100%', height: '100%' }}
                        config={{ displaylogo: false, responsive: true }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {hasHaloSummary && (
              <div className="space-y-4">
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-semibold text-white">Halo rate heatmap</h3>
                      <p className="text-[11px] text-gray-400">
                        Fraction of frames labeled as halo (-1) per residue and condition.
                      </p>
                    </div>
                    <div className="text-[11px] text-gray-500">
                      Conditions: {haloConditionOptions.length} • Residues: {haloResidueLabels.length}
                    </div>
                  </div>
                  <Plot
                    data={haloHeatmapData}
                    layout={{
                      height: Math.max(320, 26 * haloConditionOptions.length),
                      paper_bgcolor: '#111827',
                      plot_bgcolor: '#111827',
                      font: { color: '#e5e7eb' },
                      margin: { l: 110, r: 10, t: 20, b: 80 },
                      xaxis: {
                        title: 'Residues',
                        showticklabels: haloShowResidueTicks,
                        tickangle: -40,
                      },
                      yaxis: { title: 'Condition', automargin: true },
                    }}
                    useResizeHandler
                    style={{ width: '100%', height: '100%' }}
                    config={{ displaylogo: false, responsive: true }}
                  />
                </div>

                <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-semibold text-white">Halo ranking</h3>
                      <p className="text-[11px] text-gray-400">
                        Ranked residues by halo rate for the selected condition.
                      </p>
                    </div>
                    <div className="min-w-[220px]">
                      <label className="block text-[11px] text-gray-500 mb-1">Condition</label>
                      <select
                        value={selectedHaloCondition}
                        onChange={(e) => setSelectedHaloCondition(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-white text-xs"
                      >
                        {haloConditionOptions.map((opt) => (
                          <option key={opt.id} value={opt.id}>
                            {opt.type === 'metastable' ? 'Metastable' : 'Macro'}: {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      {haloRanking.length === 0 && (
                        <p className="text-xs text-gray-500">No halo ranking available.</p>
                      )}
                      {haloRanking.slice(0, 25).map((entry, idx) => (
                        <div
                          key={`${entry.key}-${idx}`}
                          className="flex items-center justify-between text-xs text-gray-200"
                        >
                          <span className="truncate">
                            {idx + 1}. {entry.label}
                          </span>
                          <span className="text-gray-400">{entry.value.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                    <div className="text-xs text-gray-400 space-y-2">
                      <p>
                        Higher halo rates indicate residues that fall outside dense clusters more often under the
                        selected condition.
                      </p>
                      {haloRanking.length > 25 && (
                        <p className="text-[11px] text-gray-500">
                          Showing top 25 of {haloRanking.length} residues. Refine by residue name for details.
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
