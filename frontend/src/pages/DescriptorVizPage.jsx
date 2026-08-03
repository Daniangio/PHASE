import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { CircleHelp } from 'lucide-react';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import ClusterPieChart, { clusterPieColor } from '../components/common/ClusterPieChart';
import {
  fetchSystem,
  fetchStateDescriptors,
  createClusterPatch,
  confirmClusterPatch,
  discardClusterPatch,
} from '../api/projects';

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
const DEFAULT_DIHEDRAL_KEYS = ['phi', 'psi', 'omega', 'chi1', 'chi2'];

function isFiniteNumber(value) {
  return Number.isFinite(Number(value));
}

export default function DescriptorVizPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [error, setError] = useState(null);
  const [helpOpen, setHelpOpen] = useState(false);

  const [selectedStates, setSelectedStates] = useState([]);
  const [appliedStates, setAppliedStates] = useState([]);
  const [selectedMetastableIds, setSelectedMetastableIds] = useState([]);
  const [residueFilter, setResidueFilter] = useState('');
  const [selectedResidue, setSelectedResidue] = useState('');
  const [residueOptions, setResidueOptions] = useState([]);
  const [residueLabelCache, setResidueLabelCache] = useState({});
  const [axisX, setAxisX] = useState('phi');
  const [axisY, setAxisY] = useState('psi');
  const [axisZ, setAxisZ] = useState('omega');

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
  const [appliedMetaQuery, setAppliedMetaQuery] = useState('');
  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [selectedClusterVariantId, setSelectedClusterVariantId] = useState('original');
  const [clusterVariants, setClusterVariants] = useState([]);
  const [clusterLegend, setClusterLegend] = useState([]);
  const [clusterLabelMode, setClusterLabelMode] = useState('halo');
  const [haloSummary, setHaloSummary] = useState(null);
  const [selectedHaloCondition, setSelectedHaloCondition] = useState('');
  const [patchResiduesInput, setPatchResiduesInput] = useState('');
  const [patchClusterSelectionMode, setPatchClusterSelectionMode] = useState('maxclust');
  const [patchNClusters, setPatchNClusters] = useState('');
  const [patchInconsistentThreshold, setPatchInconsistentThreshold] = useState('1.0');
  const [patchInconsistentDepth, setPatchInconsistentDepth] = useState('2');
  const [patchMaxClusterFrames, setPatchMaxClusterFrames] = useState('');
  const [patchLinkage, setPatchLinkage] = useState('ward');
  const [patchCovariance, setPatchCovariance] = useState('full');
  const [patchHaloPercentile, setPatchHaloPercentile] = useState(5);
  const [patchBusy, setPatchBusy] = useState(false);
  const [patchError, setPatchError] = useState(null);
  const [clusterVariantPanelOpen, setClusterVariantPanelOpen] = useState(false);
  const [pieModal, setPieModal] = useState(null);

  const [anglesByState, setAnglesByState] = useState({});
  const [metaByState, setMetaByState] = useState({});
  const [loadingAngles, setLoadingAngles] = useState(false);
  const [anglesError, setAnglesError] = useState(null);
  const didHydrateQueryRef = useRef(false);
  const bootstrapLabelsRequestIdRef = useRef(0);
  const loadAnglesRequestIdRef = useRef(0);

  const axisLabel = useCallback((key) => {
    if (key === 'phi') return 'Phi';
    if (key === 'psi') return 'Psi';
    if (key === 'omega') return 'Omega';
    if (key === 'chi1') return 'Chi1';
    if (key === 'chi2') return 'Chi2';
    return String(key || '');
  }, []);

  const normalizeClusterId = useCallback((raw) => {
    if (raw === null || raw === undefined) return -1;
    if (typeof raw === 'number' && Number.isFinite(raw)) return Math.trunc(raw);
    const text = String(raw).trim();
    if (!text) return -1;
    const prefixed = text.match(/^c(-?\d+)$/i);
    if (prefixed) return Number(prefixed[1]);
    const parsed = Number(text);
    if (Number.isFinite(parsed)) return Math.trunc(parsed);
    return -1;
  }, []);

  useEffect(() => {
    const loadSystem = async () => {
      setLoadingSystem(true);
      setError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
        const descriptorStates = Object.values(data.states || {}).filter((s) => s.descriptor_file);
        if (descriptorStates.length) {
          const firstStateId = descriptorStates[0].state_id;
          setSelectedStates((prev) => (prev.length ? prev : [firstStateId]));
          setAppliedStates((prev) => (prev.length ? prev : [firstStateId]));
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

  // Hydrate from query params once on first mount.
  useEffect(() => {
    if (didHydrateQueryRef.current) return;
    didHydrateQueryRef.current = true;
    const params = new URLSearchParams(location.search || '');
    const clusterId = params.get('cluster_id');
    if (clusterId) {
      setSelectedClusterId(clusterId);
    }
    const variantId = params.get('cluster_variant_id');
    if (variantId) {
      setSelectedClusterVariantId(variantId);
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

  useEffect(() => {
    if (!system) return;
    const params = new URLSearchParams(location.search || '');
    const metaParam = params.get('metastable_ids');
    if (!metaParam) return;
    if (metaParam === appliedMetaQuery) return;
    const ids = metaParam
      .split(',')
      .map((v) => v.trim())
      .filter(Boolean);
    if (ids.length) {
      setSelectedMetastableIds(ids);
    }
    setAppliedMetaQuery(metaParam);
  }, [appliedMetaQuery, location.search, system]);


  const residueKeys = useMemo(() => sortResidues(residueOptions), [residueOptions, sortResidues]);

  const residueLabel = useCallback(
    (key) => {
      if (residueLabelCache[key]) return residueLabelCache[key];
      const metasInOrder = [
        ...appliedStates.map((stateId) => metaByState[stateId]).filter(Boolean),
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
    [appliedStates, metaByState, residueLabelCache]
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

  const selectedClusterVariant = useMemo(
    () =>
      (clusterVariants || []).find((v) => String(v.id) === String(selectedClusterVariantId)) || null,
    [clusterVariants, selectedClusterVariantId]
  );

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

      const clusterLabels = Array.isArray(data.cluster_labels)
        ? data.cluster_labels.map((v) => normalizeClusterId(v))
        : null;
      const clusterColors =
        clusterLabels && clusterLegend.length
          ? clusterLabels.map((c) => {
              if (c < 0) return '#9ca3af';
              return clusterColorMap[c] || colors[Math.abs(c) % colors.length];
            })
          : null;
      const clusterHover =
        clusterLabels && clusterLegend.length
          ? clusterLabels.map((c) => (c >= 0 ? clusterLabelLookup[c] || `Cluster ${c}` : 'No cluster'))
          : null;

      const pick = (arr, indices) => indices.map((idx) => arr[idx]);
      const finiteIndices = (indices) =>
        indices.filter((idx) => {
          if (!isFiniteNumber(xVals[idx]) || !isFiniteNumber(yVals[idx])) return false;
          if (axes.zKey && !isFiniteNumber(zVals[idx])) return false;
          return true;
        });
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
        const indices = finiteIndices(groups[groupKey] || []);
        if (!indices.length) return null;
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
          type: axes.zKey ? 'scatter3d' : 'scatter',
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
              ? `<br>${axisLabel(axes.xKey)}: %{x:.2f}°<br>${axisLabel(axes.yKey)}: %{y:.2f}°<br>${axisLabel(axes.zKey)}: %{z:.2f}°`
              : `<br>${axisLabel(axes.xKey)}: %{x:.2f}°<br>${axisLabel(axes.yKey)}: %{y:.2f}°`) +
            (clusterHover ? '<br>Cluster: %{customdata}' : '') +
            '<extra></extra>',
        };
      }).filter(Boolean);
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
      axisLabel,
      normalizeClusterId,
    ]
  );

  const dihedralKeys = useMemo(() => {
    const merged = new Set();
    Object.values(metaByState || {}).forEach((meta) => {
      (meta?.dihedral_keys || []).forEach((key) => merged.add(String(key)));
    });
    if (!merged.size) {
      DEFAULT_DIHEDRAL_KEYS.forEach((key) => merged.add(key));
    }
    return DEFAULT_DIHEDRAL_KEYS.filter((key) => merged.has(key));
  }, [metaByState]);

  useEffect(() => {
    if (!dihedralKeys.length) return;
    setAxisX((prev) => (dihedralKeys.includes(prev) ? prev : dihedralKeys[0]));
    setAxisY((prev) => {
      if (dihedralKeys.includes(prev)) return prev;
      return dihedralKeys.find((key) => key !== axisX) || dihedralKeys[0];
    });
    setAxisZ((prev) => {
      if (dihedralKeys.includes(prev)) return prev;
      return dihedralKeys.find((key) => key !== axisX && key !== axisY) || dihedralKeys[0];
    });
  }, [axisX, axisY, dihedralKeys]);


  const selectResidue = (key) => {
    setSelectedResidue(key);
  };

  // Preload residue labels/resnames so the list keeps informative names even before a residue is loaded
  useEffect(() => {
    const bootstrapLabels = async () => {
      if (!appliedStates.length) return;
      const requestId = ++bootstrapLabelsRequestIdRef.current;
      try {
        const stateId = appliedStates[0];
        const data = await fetchStateDescriptors(projectId, systemId, stateId, { max_points: 10 });
        if (requestId !== bootstrapLabelsRequestIdRef.current) return;
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
          setSelectedResidue((prev) => (prev || !data.residue_keys.length ? prev : sortResidues(data.residue_keys)[0]));
        }
      } catch (err) {
        // keep silent; fallback labels will be used
      }
    };
    bootstrapLabels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appliedStates, projectId, systemId, sortResidues]);

  const loadAngles = useCallback(async () => {
    const requestId = ++loadAnglesRequestIdRef.current;
    if (!appliedStates.length) {
      setAnglesByState({});
      setMetaByState({});
      setClusterLegend([]);
      setClusterVariants([]);
      setSelectedClusterVariantId('original');
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
        qs.cluster_variant_id = selectedClusterVariantId;
      }
      if (selectedMetastableIds.length) {
        qs.metastable_ids = selectedMetastableIds;
      }
      if (selectedResidue) {
        qs.residue_keys = selectedResidue;
      }

      const responses = await Promise.all(
        appliedStates.map(async (stateId) => {
          const data = await fetchStateDescriptors(projectId, systemId, stateId, qs);
          return { stateId, data };
        })
      );
      if (requestId !== loadAnglesRequestIdRef.current) return;

      const newAngles = {};
      const newMeta = {};
      const unionResidues = new Set();
      let nextHaloSummary = null;
      let nextClusterVariants = [];
      let nextVariantId = selectedClusterVariantId;

      responses.forEach(({ stateId, data }) => {
        newAngles[stateId] = data.angles || {};
        newMeta[stateId] = {
          residue_keys: data.residue_keys || [],
          dihedral_keys: data.dihedral_keys || DEFAULT_DIHEDRAL_KEYS,
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
        if (!nextClusterVariants.length && Array.isArray(data.cluster_variants)) {
          nextClusterVariants = data.cluster_variants;
          if (data.cluster_variant_id) {
            nextVariantId = String(data.cluster_variant_id);
          }
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

      const legendFromMetadata = new Map();
      responses.forEach(({ data }) => {
        (data.cluster_legend || []).forEach((entry) => {
          const id = normalizeClusterId(entry?.id);
          if (id < 0) return;
          const key = String(id);
          if (!legendFromMetadata.has(key)) {
            legendFromMetadata.set(key, entry?.label || `c${id}`);
          }
        });
      });

      // Build legend from the labels actually present in the fetched points. This guarantees
      // every visible non-halo cluster has a color and avoids rendering valid clusters as gray.
      const observed = new Set();
      responses.forEach(({ data }) => {
        Object.values(data.angles || {}).forEach((anglePayload) => {
          (anglePayload?.cluster_labels || []).forEach((raw) => {
            const id = normalizeClusterId(raw);
            if (id >= 0) observed.add(id);
          });
        });
      });
      const ids = observed.size
        ? Array.from(observed).sort((a, b) => a - b)
        : Array.from(legendFromMetadata.keys())
            .map((k) => Number(k))
            .filter((v) => Number.isFinite(v))
            .sort((a, b) => a - b);
      const mergedLegend = ids.map((id) => ({
        id,
        label: legendFromMetadata.get(String(id)) || `c${id}`,
      }));
      setClusterLegend(mergedLegend);
      setClusterVariants(nextClusterVariants);
      if (nextClusterVariants.length) {
        const allowed = new Set(nextClusterVariants.map((v) => String(v.id)));
        const suggested = String(nextVariantId || 'original');
        const current = String(selectedClusterVariantId || 'original');
        const resolved = allowed.has(current)
          ? current
          : allowed.has(suggested)
          ? suggested
          : 'original';
        if (resolved !== current) {
          setSelectedClusterVariantId(resolved);
        }
      }
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
      setSelectedResidue((prev) => {
        if (!sortedResidues.length) return '';
        if (!prev) return sortedResidues[0];
        if (!unionResidues.has(prev)) return sortedResidues[0];
        return prev;
      });

      if (!bootstrapOnly && selectedResidue) {
        setAnglesByState(newAngles);
      } else {
        // During bootstrap, only populate residue options; a follow-up call will load the selected residue
        setAnglesByState({});
      }
    } catch (err) {
      if (requestId !== loadAnglesRequestIdRef.current) return;
      setAnglesError(err.message);
    } finally {
      if (requestId === loadAnglesRequestIdRef.current) {
        setLoadingAngles(false);
      }
    }
  }, [
    maxPoints,
    clusterLabelMode,
    selectedClusterVariantId,
    normalizeClusterId,
    projectId,
    selectedMetastableIds,
    selectedClusterId,
    selectedResidue,
    appliedStates,
    sortResidues,
    systemId,
  ]);

  useEffect(() => {
    if (appliedStates.length) {
      loadAngles();
    } else {
      setAnglesByState({});
      setMetaByState({});
      setSelectedResidue('');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appliedStates, selectedClusterId, selectedResidue, clusterLabelMode, selectedClusterVariantId, selectedMetastableIds]);

  useEffect(() => {
    if (!patchResiduesInput && selectedResidue) {
      setPatchResiduesInput(selectedResidue);
    }
  }, [selectedResidue, patchResiduesInput]);

  useEffect(() => {
    if (!selectedClusterId) {
      setClusterVariantPanelOpen(false);
    }
  }, [selectedClusterId]);

  const parsePatchResidues = useCallback(() => {
    const raw = (patchResiduesInput || '').trim();
    if (!raw) return selectedResidue ? [selectedResidue] : [];
    const toks = raw
      .split(/[,\s]+/)
      .map((v) => v.trim())
      .filter(Boolean);
    return Array.from(new Set(toks));
  }, [patchResiduesInput, selectedResidue]);

  const handleCreatePatch = useCallback(async () => {
    if (!selectedClusterId) return;
    const residueKeys = parsePatchResidues();
    if (!residueKeys.length) {
      setPatchError('Select at least one residue key for patching.');
      return;
    }
    setPatchBusy(true);
    setPatchError(null);
    try {
      const haloPct = Number(patchHaloPercentile);
      const payload = {
        residue_keys: residueKeys,
        cluster_selection_mode: patchClusterSelectionMode,
        linkage_method: patchLinkage,
        covariance_type: patchCovariance,
        halo_percentile: Number.isFinite(haloPct) ? haloPct : 5,
      };
      if (patchClusterSelectionMode === 'inconsistent') {
        const thr = Number(patchInconsistentThreshold);
        if (!Number.isFinite(thr)) {
          setPatchError('Inconsistent threshold must be numeric.');
          setPatchBusy(false);
          return;
        }
        payload.inconsistent_threshold = thr;
        const depth = Number(patchInconsistentDepth);
        payload.inconsistent_depth = Number.isFinite(depth) && depth > 0 ? Math.floor(depth) : 2;
      } else {
        const ncl = Number(patchNClusters);
        if (Number.isFinite(ncl) && ncl > 0) {
          payload.n_clusters = Math.floor(ncl);
        }
      }
      const maxFrames = Number(patchMaxClusterFrames);
      if (Number.isFinite(maxFrames) && maxFrames > 0) {
        payload.max_cluster_frames = Math.floor(maxFrames);
      }
      const out = await createClusterPatch(projectId, systemId, selectedClusterId, payload);
      if (out?.patch_id) setSelectedClusterVariantId(String(out.patch_id));
      await loadAngles();
    } catch (err) {
      setPatchError(err.message || 'Failed to create patch.');
    } finally {
      setPatchBusy(false);
    }
  }, [
    selectedClusterId,
    parsePatchResidues,
    patchClusterSelectionMode,
    patchLinkage,
    patchCovariance,
    patchHaloPercentile,
    patchNClusters,
    patchInconsistentThreshold,
    patchInconsistentDepth,
    patchMaxClusterFrames,
    projectId,
    systemId,
    loadAngles,
  ]);

  const handleConfirmPatch = useCallback(async () => {
    if (!selectedClusterId || !selectedClusterVariantId || selectedClusterVariantId === 'original') return;
    setPatchBusy(true);
    setPatchError(null);
    try {
      await confirmClusterPatch(projectId, systemId, selectedClusterId, selectedClusterVariantId);
      setSelectedClusterVariantId('original');
      await loadAngles();
    } catch (err) {
      setPatchError(err.message || 'Failed to confirm patch.');
    } finally {
      setPatchBusy(false);
    }
  }, [projectId, systemId, selectedClusterId, selectedClusterVariantId, loadAngles]);

  const handleDiscardPatch = useCallback(async () => {
    if (!selectedClusterId || !selectedClusterVariantId || selectedClusterVariantId === 'original') return;
    setPatchBusy(true);
    setPatchError(null);
    try {
      await discardClusterPatch(projectId, systemId, selectedClusterId, selectedClusterVariantId);
      setSelectedClusterVariantId('original');
      await loadAngles();
    } catch (err) {
      setPatchError(err.message || 'Failed to discard patch.');
    } finally {
      setPatchBusy(false);
    }
  }, [projectId, systemId, selectedClusterId, selectedClusterVariantId, loadAngles]);

  const traces3d = useMemo(() => {
    const traces = [];
    const residuesToPlot = selectedResidue ? [selectedResidue] : [];
    selectedStates.forEach((stateId) => {
      const perState = anglesByState[stateId] || {};
      residuesToPlot.forEach((key) => {
        const data = perState[key];
        if (!data) return;
        traces.push(
          ...buildGroupedTraces(stateId, key, data, { xKey: axisX, yKey: axisY, zKey: axisZ })
        );
      });
    });
    // Add legend for clusters if present
    if (clusterLegend.length) {
      clusterLegend.forEach((c) => {
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: [0],
          y: [0],
          z: [0],
          name: c.label,
          showlegend: true,
          visible: 'legendonly',
          marker: { color: clusterColorMap[c.id] || '#9ca3af' },
          hoverinfo: 'none',
        });
      });
    }
    return traces;
  }, [anglesByState, axisX, axisY, axisZ, buildGroupedTraces, clusterColorMap, clusterLegend, selectedResidue, selectedStates]);

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

  const residuePieByState = useMemo(() => {
    if (!selectedResidue) return [];
    const rows = [];
    selectedStates.forEach((stateId) => {
      const payload = anglesByState?.[stateId]?.[selectedResidue];
      const labelsRaw = Array.isArray(payload?.cluster_labels) ? payload.cluster_labels : null;
      if (!labelsRaw || !labelsRaw.length) {
        rows.push({
          stateId,
          stateName: stateName(stateId),
          total: 0,
          valid: 0,
          slices: [],
        });
        return;
      }
      const labels = labelsRaw.map((v) => normalizeClusterId(v));
      const counts = new Map();
      labels.forEach((cid) => {
        if (cid < 0) return;
        counts.set(cid, (counts.get(cid) || 0) + 1);
      });
      const valid = Array.from(counts.values()).reduce((acc, v) => acc + v, 0);
      const slicesRaw = Array.from(counts.entries())
        .sort((a, b) => b[1] - a[1])
        .map(([cid, count], idx) => {
          const label = clusterLabelLookup[cid] || `c${cid}`;
          const color = clusterColorMap[cid] || colors[Math.abs(cid) % colors.length] || clusterPieColor(idx);
          const frac = valid > 0 ? count / valid : 0;
          return {
            label,
            value: frac,
            color,
            tooltip: `cluster ${label}`,
          };
        });
      const maxSlices = 10;
      const keep = slicesRaw.slice(0, maxSlices);
      const rest = slicesRaw.slice(maxSlices);
      const restValue = rest.reduce((acc, s) => acc + (Number(s.value) || 0), 0);
      const slices = restValue > 0 ? [...keep, { label: 'other', value: restValue, color: '#6b7280', tooltip: 'other clusters' }] : keep;
      rows.push({
        stateId,
        stateName: stateName(stateId),
        total: labels.length,
        valid,
        slices,
      });
    });
    return rows;
  }, [
    anglesByState,
    selectedResidue,
    selectedStates,
    stateName,
    normalizeClusterId,
    clusterColorMap,
    clusterLabelLookup,
  ]);

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
      appliedStates.map((stateId) => ({
        stateId,
        name: stateName(stateId),
        frames: metaByState[stateId]?.n_frames,
        stride: metaByState[stateId]?.sample_stride,
      })),
    [appliedStates, metaByState, stateName]
  );

  const hasPendingStateSelection = useMemo(() => {
    if (selectedStates.length !== appliedStates.length) return true;
    return selectedStates.some((stateId, idx) => stateId !== appliedStates[idx]);
  }, [appliedStates, selectedStates]);

  const handleRefreshData = useCallback(() => {
    setAppliedStates([...selectedStates]);
  }, [selectedStates]);

  if (loadingSystem) return <Loader message="Loading system..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!system) return null;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Residue Patch Clustering: Help"
        docPath="/docs/cluster_patching_help.md"
        onClose={() => setHelpOpen(false)}
      />
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
            Visualize per-residue dihedral angles (phi, psi, omega, chi1, chi2). Data are down-sampled for plotting.
          </p>
        </div>
        <div className="flex items-start gap-3">
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
          >
            <CircleHelp className="h-4 w-4" />
            Help
          </button>
          <div className="text-xs text-gray-400 text-right space-y-0.5">
            <div>
              Loaded states: {appliedStates.length ? stateSummaries.map((s) => s.name).join(', ') : '—'}
            </div>
            <div>
              Selected states: {selectedStates.length ? selectedStates.map((stateId) => stateName(stateId)).join(', ') : '—'}
            </div>
            <div>
              Metastable: {selectedMetastableIds.length ? selectedMetastableIds.length : 'All'}
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
      </div>

      {descriptorStates.length === 0 ? (
        <ErrorMessage message="No descriptor-ready states. Upload trajectories and build descriptors first." />
      ) : (
        <div className="lg:grid lg:grid-cols-[minmax(260px,24%)_1fr] gap-4">
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
              onClick={handleRefreshData}
              disabled={loadingAngles || !selectedStates.length}
              className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md disabled:opacity-50"
            >
              {loadingAngles ? 'Loading…' : 'Refresh data'}
            </button>
            {hasPendingStateSelection && (
              <p className="text-[11px] text-amber-300">
                State selection changed. Click `Refresh data` to apply it.
              </p>
            )}
            <div>
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-gray-400">Macro states</p>
                <div className="flex items-center gap-2 text-[11px] text-gray-400">
                  <button
                    type="button"
                    onClick={() => setSelectedStates(descriptorStates.map((s) => s.state_id))}
                    className="hover:text-cyan-300"
                  >
                    Select all
                  </button>
                  <button
                    type="button"
                    onClick={() => setSelectedStates([])}
                    className="hover:text-cyan-300"
                  >
                    Deselect all
                  </button>
                </div>
              </div>
              <div className="space-y-2 max-h-48 overflow-y-auto border border-gray-700 rounded-md p-2 bg-gray-900">
                {descriptorStates.map((state) => (
                  <label key={state.state_id} className="flex items-center gap-2 text-xs text-gray-200">
                    <input
                      type="checkbox"
                      checked={selectedStates.includes(state.state_id)}
                      onChange={() =>
                        setSelectedStates((prev) =>
                          prev.includes(state.state_id)
                            ? prev.filter((id) => id !== state.state_id)
                            : [...prev, state.state_id]
                        )
                      }
                      className="accent-cyan-500"
                    />
                    <span className="truncate">{state.name || state.state_id}</span>
                  </label>
                ))}
              </div>
            </div>
            {metastableStates.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-gray-400">Metastable states</p>
                  <div className="flex items-center gap-2 text-[11px] text-gray-400">
                    <button
                      type="button"
                      onClick={() =>
                        setSelectedMetastableIds(
                          metastableStates.map((m) => m.metastable_id).filter(Boolean)
                        )
                      }
                      className="hover:text-cyan-300"
                    >
                      Select all
                    </button>
                    <button
                      type="button"
                      onClick={() => setSelectedMetastableIds([])}
                      className="hover:text-cyan-300"
                    >
                      Deselect all
                    </button>
                  </div>
                </div>
                <div className="space-y-2 max-h-48 overflow-y-auto border border-gray-700 rounded-md p-2 bg-gray-900">
                  {metastableStates.map((meta) => {
                    const label = meta.name || meta.default_name || meta.metastable_id;
                    const macroLabel =
                      descriptorStates.find((s) => s.state_id === meta.macro_state_id)?.name ||
                      meta.macro_state_id;
                    return (
                      <label key={meta.metastable_id} className="flex items-center gap-2 text-xs text-gray-200">
                        <input
                          type="checkbox"
                          checked={selectedMetastableIds.includes(meta.metastable_id)}
                          onChange={() =>
                            setSelectedMetastableIds((prev) =>
                              prev.includes(meta.metastable_id)
                                ? prev.filter((id) => id !== meta.metastable_id)
                                : [...prev, meta.metastable_id]
                            )
                          }
                          className="accent-cyan-500"
                        />
                        <span className="truncate">
                          {label} <span className="text-[10px] text-gray-500">({macroLabel})</span>
                        </span>
                      </label>
                    );
                  })}
                </div>
              </div>
            )}
            <div>
              <label className="block text-xs text-gray-400 mb-1">Cluster (optional coloring)</label>
              <select
                value={selectedClusterId}
                onChange={(e) => {
                  setSelectedClusterId(e.target.value);
                  setSelectedClusterVariantId('original');
                }}
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
              {selectedClusterId && (
                <div className="mt-3">
                  <button
                    type="button"
                    onClick={() => setClusterVariantPanelOpen(true)}
                    className="w-full border border-gray-700 rounded-md px-3 py-2 text-xs text-gray-200 hover:border-cyan-500"
                  >
                    Open Cluster Variant & Patch Panel
                  </button>
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
                  <div className="grid md:grid-cols-3 gap-3">
                    {[
                      { label: 'X axis', value: axisX, setter: setAxisX },
                      { label: 'Y axis', value: axisY, setter: setAxisY },
                      { label: 'Z axis', value: axisZ, setter: setAxisZ },
                    ].map((entry) => (
                      <div key={entry.label}>
                        <label className="block text-xs text-gray-400 mb-1">{entry.label}</label>
                        <select
                          value={entry.value}
                          onChange={(e) => entry.setter(e.target.value)}
                          className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                        >
                          {dihedralKeys.map((key) => (
                            <option key={`${entry.label}:${key}`} value={key}>
                              {axisLabel(key)}
                            </option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                  <Plot
                    data={traces3d}
                    layout={{
                      height: 500,
                      paper_bgcolor: '#111827',
                      plot_bgcolor: '#111827',
                      font: { color: '#e5e7eb' },
                      scene: {
                        xaxis: { title: `${axisLabel(axisX)} (°)`, range: [-180, 180] },
                        yaxis: { title: `${axisLabel(axisY)} (°)`, range: [-180, 180] },
                        zaxis: { title: `${axisLabel(axisZ)} (°)`, range: [-180, 180] },
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
                    { x: axisX, y: axisY, title: `${axisLabel(axisX)} vs ${axisLabel(axisY)}` },
                    { x: axisX, y: axisZ, title: `${axisLabel(axisX)} vs ${axisLabel(axisZ)}` },
                    { x: axisY, y: axisZ, title: `${axisLabel(axisY)} vs ${axisLabel(axisZ)}` },
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
                          xaxis: { title: `${axisLabel(axes.x)} (°)`, range: [-180, 180] },
                          yaxis: { title: `${axisLabel(axes.y)} (°)`, range: [-180, 180] },
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

            {hasAngles && selectedClusterId && (
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold text-white">Selected Residue Cluster Pies</h3>
                    <p className="text-[11px] text-gray-400">
                      Distribution of cluster assignments for <span className="font-mono">{residueLabel(selectedResidue)}</span> in each selected macro-state.
                    </p>
                  </div>
                </div>
                <div className="grid xl:grid-cols-2 gap-3">
                  {residuePieByState.map((row) => (
                    <div key={`pie:${row.stateId}`} className="rounded-md border border-gray-700 bg-gray-900/50 p-3 space-y-2">
                      <div className="flex items-center justify-between gap-2">
                        <p className="text-xs text-gray-200 truncate">{row.stateName}</p>
                        <p className="text-[11px] text-gray-500">
                          valid {row.valid} / {row.total}
                        </p>
                      </div>
                      {row.slices.length === 0 ? (
                        <p className="text-xs text-gray-500">
                          No non-halo cluster labels in the loaded sample points for this state.
                        </p>
                      ) : (
                        <div className="flex items-start gap-3">
                          <ClusterPieChart
                            slices={row.slices}
                            size={124}
                            onClick={() =>
                              setPieModal({
                                title: `Cluster distribution · ${residueLabel(selectedResidue)}`,
                                subtitle: row.stateName,
                                slices: row.slices,
                              })
                            }
                          />
                          <div className="min-w-0 flex-1 space-y-1 text-[11px]">
                            {row.slices.map((s) => (
                              <div key={`pie-slice:${row.stateId}:${s.label}`} className="flex items-center gap-2 text-gray-300">
                                <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: s.color }} />
                                <span className="truncate">{s.label}</span>
                                <span className="ml-auto text-gray-400">{(100 * s.value).toFixed(1)}%</span>
                              </div>
                            ))}
                            <button
                              type="button"
                              onClick={() =>
                                setPieModal({
                                  title: `Cluster distribution · ${residueLabel(selectedResidue)}`,
                                  subtitle: row.stateName,
                                  slices: row.slices,
                                })
                              }
                              className="mt-1 text-[11px] px-2 py-1 rounded border border-gray-700 text-gray-200 hover:border-gray-500"
                            >
                              Enlarge pie
                            </button>
                          </div>
                        </div>
                      )}
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

          {selectedClusterId && clusterVariantPanelOpen && (
            <div className="fixed inset-0 z-40 bg-black/60 flex items-center justify-center p-4">
              <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-lg border border-gray-700 bg-gray-900 p-4 space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="text-sm font-semibold text-white">Cluster Variant & Patch Panel</h3>
                  <button
                    type="button"
                    onClick={() => setClusterVariantPanelOpen(false)}
                    className="border border-gray-700 rounded-md px-2 py-1 text-xs text-gray-200 hover:border-gray-500"
                  >
                    Close
                  </button>
                </div>

                <div>
                  <label className="block text-xs text-gray-400 mb-1">Cluster variant</label>
                  <select
                    value={selectedClusterVariantId}
                    onChange={(e) => setSelectedClusterVariantId(e.target.value || 'original')}
                    className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                  >
                    {(clusterVariants.length ? clusterVariants : [{ id: 'original', label: 'Original cluster' }]).map(
                      (v) => (
                        <option key={String(v.id)} value={String(v.id)}>
                          {v.label || v.id}
                        </option>
                      )
                    )}
                  </select>
                  {selectedClusterVariant && selectedClusterVariantId !== 'original' && (
                    <p className="text-[11px] text-gray-500 mt-1">
                      Variant status: {selectedClusterVariant.status || 'preview'}
                    </p>
                  )}
                </div>

                <div className="space-y-1">
                  <label className="block text-xs text-gray-400">Patch residues</label>
                  <input
                    type="text"
                    value={patchResiduesInput}
                    onChange={(e) => setPatchResiduesInput(e.target.value)}
                    placeholder="res_10,res_11"
                    className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                  />
                  <p className="text-[10px] text-gray-500">
                    Comma-separated residue keys. Selected residue is used if empty.
                  </p>
                </div>

                <div>
                  <label className="block text-xs text-gray-400 mb-1">Cluster selection mode</label>
                  <select
                    value={patchClusterSelectionMode}
                    onChange={(e) => setPatchClusterSelectionMode(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                  >
                    <option value="maxclust">Fixed K (maxclust)</option>
                    <option value="inconsistent">Auto by threshold (inconsistent)</option>
                  </select>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  {patchClusterSelectionMode === 'maxclust' ? (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">n_clusters (optional)</label>
                      <input
                        type="number"
                        min={1}
                        value={patchNClusters}
                        onChange={(e) => setPatchNClusters(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                      />
                    </div>
                  ) : (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Inconsistent threshold</label>
                      <input
                        type="number"
                        step={0.1}
                        value={patchInconsistentThreshold}
                        onChange={(e) => setPatchInconsistentThreshold(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                      />
                    </div>
                  )}
                  {patchClusterSelectionMode === 'inconsistent' && (
                    <div>
                      <label className="block text-xs text-gray-400 mb-1">Inconsistent depth</label>
                      <input
                        type="number"
                        min={1}
                        value={patchInconsistentDepth}
                        onChange={(e) => setPatchInconsistentDepth(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                      />
                    </div>
                  )}
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Max frames (optional)</label>
                    <input
                      type="number"
                      min={1}
                      value={patchMaxClusterFrames}
                      onChange={(e) => setPatchMaxClusterFrames(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Halo percentile</label>
                    <input
                      type="number"
                      min={0}
                      max={50}
                      step={0.5}
                      value={patchHaloPercentile}
                      onChange={(e) => setPatchHaloPercentile(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Linkage</label>
                    <select
                      value={patchLinkage}
                      onChange={(e) => setPatchLinkage(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                    >
                      <option value="ward">ward</option>
                      <option value="complete">complete</option>
                      <option value="average">average</option>
                      <option value="single">single</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Covariance</label>
                    <select
                      value={patchCovariance}
                      onChange={(e) => setPatchCovariance(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1.5 text-white text-xs"
                    >
                      <option value="full">full</option>
                      <option value="diag">diag</option>
                    </select>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={handleCreatePatch}
                  disabled={patchBusy}
                  className="w-full bg-cyan-700 hover:bg-cyan-600 text-white text-xs font-semibold py-1.5 rounded-md disabled:opacity-60"
                >
                  {patchBusy ? 'Working…' : 'Create preview patch'}
                </button>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={handleConfirmPatch}
                    disabled={patchBusy || selectedClusterVariantId === 'original'}
                    className="bg-emerald-700 hover:bg-emerald-600 text-white text-xs font-semibold py-1.5 rounded-md disabled:opacity-60"
                  >
                    Confirm swap
                  </button>
                  <button
                    type="button"
                    onClick={handleDiscardPatch}
                    disabled={patchBusy || selectedClusterVariantId === 'original'}
                    className="bg-rose-700 hover:bg-rose-600 text-white text-xs font-semibold py-1.5 rounded-md disabled:opacity-60"
                  >
                    Discard patch
                  </button>
                </div>
                {patchError && <p className="text-[11px] text-rose-400">{patchError}</p>}
              </div>
            </div>
          )}
        </div>
      )}
      {pieModal && (
        <div className="fixed inset-0 z-40 bg-black/70 flex items-center justify-center p-4">
          <div className="w-full max-w-xl rounded-lg border border-gray-700 bg-gray-900 p-4 space-y-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-white">{pieModal.title}</h3>
                {pieModal.subtitle && <p className="text-xs text-gray-400">{pieModal.subtitle}</p>}
              </div>
              <button
                type="button"
                onClick={() => setPieModal(null)}
                className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-200 hover:border-gray-500"
              >
                Close
              </button>
            </div>
            <div className="flex flex-col md:flex-row items-start gap-4">
              <ClusterPieChart slices={pieModal.slices || []} size={280} />
              <div className="flex-1 max-h-72 overflow-auto pr-1 space-y-1 text-xs">
                {(pieModal.slices || []).map((s) => (
                  <div key={`pie-modal-slice:${s.label}`} className="flex items-center gap-2 text-gray-200">
                    <span className="inline-block h-3 w-3 rounded-sm" style={{ backgroundColor: s.color }} />
                    <span className="truncate">{s.tooltip || s.label}</span>
                    <span className="ml-auto text-gray-400">{(100 * (Number(s.value) || 0)).toFixed(2)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
