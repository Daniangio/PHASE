import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useParams } from 'react-router-dom';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import { Color } from 'molstar/lib/mol-util/color';
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Script } from 'molstar/lib/mol-script/script';
import { Bond, Model, StructureElement, StructureProperties, StructureSelection } from 'molstar/lib/mol-model/structure';
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms';
import { ColorThemeCategory } from 'molstar/lib/mol-theme/color/categories';
import { clearStructureOverpaint, setStructureOverpaint } from 'molstar/lib/mol-plugin-state/helpers/structure-overpaint';
import 'molstar/build/viewer/molstar.css';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import ClusterPieChart, { clusterPieColor } from '../components/common/ClusterPieChart';
import EnergyDistributionPlot, { buildEnergyDistributionPlot } from '../components/common/EnergyDistributionPlot';
import {
  assignClusterStates,
  deleteClusterUiSetup,
  fetchClusterAnalyses,
  fetchClusterAnalysisData,
  fetchClusterUiSetups,
  fetchPottsClusterInfo,
  fetchStateTrajectoryOverlay,
  fetchSystem,
  saveClusterUiSetup,
  uploadStateTrajectory,
} from '../api/projects';

function energyModelKey(analysis) {
  const ids = Array.isArray(analysis?.model_ids) && analysis.model_ids.length
    ? analysis.model_ids.map(String)
    : [String(analysis?.model_id || '')].filter(Boolean);
  return `${ids.join('+')}|${analysis?.md_label_mode || 'assigned'}|${Boolean(analysis?.drop_invalid)}`;
}

function energyModelLabel(analysis) {
  const names = Array.isArray(analysis?.model_names) && analysis.model_names.length
    ? analysis.model_names
    : [analysis?.model_name || analysis?.model_id].filter(Boolean);
  return names.join(' + ') || 'Unnamed Potts model';
}

function inferCoordinateFormat(name = '') {
  const extension = String(name || '').split('?')[0].toLowerCase().split('.').pop();
  if (extension === 'dcd') return 'dcd';
  if (extension === 'trr') return 'trr';
  if (extension === 'nc' || extension === 'nctraj') return 'nctraj';
  return 'xtc';
}

function trajectoryFrameIndex(model) {
  if (!model) return 0;
  try {
    return Number(Model.TrajectoryInfo.get(model)?.index || 0);
  } catch {
    // Mol* briefly exposes an incomplete hierarchy while changing frames.
    return 0;
  }
}

function createTrajectoryClusterTheme(frameLabels, residueIndexByResid) {
  const labelsBySourceFrame = new Map(
    (frameLabels || []).map((row) => [Number(row.source_frame_index), row.clusters || {}])
  );
  const provider = {
    name: 'phase-trajectory-residue-clusters',
    label: 'PHASE trajectory residue clusters',
    category: ColorThemeCategory.Residue,
    getParams: () => ({}),
    defaultValues: {},
    isApplicable: (ctx) => Boolean(ctx.structure),
  };
  const factory = (ctx, props) => {
    const location = StructureElement.Location.create(ctx.structure);
    const colorFor = (unit, element) => {
      location.unit = unit;
      location.element = element;
      const resid = Number(StructureProperties.residue.auth_seq_id(location));
      const residueIndex = residueIndexByResid.get(resid);
      const sourceFrame = trajectoryFrameIndex(unit.model);
      const clusterId = Number(labelsBySourceFrame.get(sourceFrame)?.[String(residueIndex)] ?? -1);
      return Color(hexToInt(clusterId >= 0 ? clusterPieColor(clusterId) : '#64748b'));
    };
    return {
      factory,
      granularity: 'group',
      color: (loc) => {
        if (StructureElement.Location.is(loc)) return colorFor(loc.unit, loc.element);
        if (Bond.isLocation(loc)) return colorFor(loc.aUnit, loc.aUnit.elements[loc.aIndex]);
        return Color(hexToInt('#64748b'));
      },
      props,
      contextHash: trajectoryFrameIndex(ctx.structure?.models?.[0]),
      description: 'Colors selected residues by their PHASE cluster in the current trajectory frame.',
    };
  };
  provider.factory = factory;
  return provider;
}

function hexToInt(hex) {
  const clean = String(hex || '#22d3ee').replace('#', '');
  return parseInt(clean, 16);
}

function residFromKey(key) {
  const m = String(key || '').match(/(-?\d+)(?!.*\d)/);
  return m ? Number(m[1]) : null;
}

function normalizeBlock(block) {
  const start = Number(block.start);
  const end = Number(block.end);
  const residues = String(block.residuesText || block.residues || '')
    .split(/[\s,;]+/)
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isInteger(v));
  return {
    id: block.id || crypto.randomUUID(),
    name: block.name || 'block',
    start: Number.isInteger(start) ? start : null,
    end: Number.isInteger(end) ? end : null,
    residues,
  };
}

function selectedResiduesFromBlocks(blocks, residueKeys) {
  const available = new Set(residueKeys.map(residFromKey).filter((v) => Number.isInteger(v)));
  const out = new Set();
  (blocks || []).forEach((raw) => {
    const block = normalizeBlock(raw);
    if (Number.isInteger(block.start) && Number.isInteger(block.end)) {
      const a = Math.min(block.start, block.end);
      const b = Math.max(block.start, block.end);
      for (let r = a; r <= b; r += 1) if (available.has(r)) out.add(r);
    }
    block.residues.forEach((r) => { if (available.has(r)) out.add(r); });
  });
  return Array.from(out).sort((a, b) => a - b);
}

function parseFrameSelection(value, frameCount) {
  const selected = new Set();
  String(value || '').split(',').map((part) => part.trim()).filter(Boolean).forEach((part) => {
    const match = part.match(/^(\d+)\s*-\s*(\d+)$/);
    if (match) {
      const start = Number(match[1]);
      const end = Number(match[2]);
      // Ranges are half-open: 500-1000 means frames 500..999 (500 frames).
      for (let idx = Math.min(start, end); idx < Math.max(start, end); idx += 1) selected.add(idx);
      return;
    }
    if (/^\d+$/.test(part)) selected.add(Number(part));
  });
  const frames = Array.from(selected).filter((idx) => idx >= 0 && (!frameCount || idx < frameCount)).sort((a, b) => a - b);
  if (!frames.length) throw new Error('Enter at least one valid frame, for example 0-19 or 0,5,10.');
  if (frames.length > 500) throw new Error('Select at most 500 frames for one overlay.');
  return frames;
}

export default function ResidueSelectionPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const query = new URLSearchParams(location.search);

  const [system, setSystem] = useState(null);
  const [selectedClusterId, setSelectedClusterId] = useState(query.get('cluster_id') || '');
  const [clusterInfo, setClusterInfo] = useState(null);
  const [setups, setSetups] = useState([]);
  const [selectedSetupId, setSelectedSetupId] = useState('');
  const [name, setName] = useState('New residue selection');
  const [blocks, setBlocks] = useState([{ id: crypto.randomUUID(), name: 'block_A', start: '', end: '', residuesText: '' }]);
  const [selectedStateId, setSelectedStateId] = useState('');
  const [viewMode, setViewMode] = useState('reference');
  const [frameSelectionMode, setFrameSelectionMode] = useState('explicit');
  const [frameSelection, setFrameSelection] = useState('0-100');
  const [maxFramesPerCluster, setMaxFramesPerCluster] = useState('10');
  const [clusterResidueIndex, setClusterResidueIndex] = useState('0');
  const [hideLicoriceHydrogens, setHideLicoriceHydrogens] = useState(true);
  const [overlaySummary, setOverlaySummary] = useState(null);
  const [wholeTrajectoryLoaded, setWholeTrajectoryLoaded] = useState(false);
  const [trajectoryClusterHighlight, setTrajectoryClusterHighlight] = useState(null);
  const [currentTrajectoryFrame, setCurrentTrajectoryFrame] = useState(0);
  const [energyAnalyses, setEnergyAnalyses] = useState([]);
  const [deltaEnergyAnalyses, setDeltaEnergyAnalyses] = useState([]);
  const [energyAnalysisLoading, setEnergyAnalysisLoading] = useState(false);
  const [energyViewMode, setEnergyViewMode] = useState('none');
  const [selectedEnergyModelKey, setSelectedEnergyModelKey] = useState('');
  const [selectedDeltaEnergyId, setSelectedDeltaEnergyId] = useState('');
  const [animatedEnergyData, setAnimatedEnergyData] = useState(null);
  const [animatedEnergyLoading, setAnimatedEnergyLoading] = useState(false);
  const [animatedEnergyError, setAnimatedEnergyError] = useState(null);
  const [selectionPieProfiles, setSelectionPieProfiles] = useState([]);
  const [pieProfilesLoading, setPieProfilesLoading] = useState(false);
  const [pieProfilesError, setPieProfilesError] = useState(null);
  const [trajectoryFile, setTrajectoryFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [pickTarget, setPickTarget] = useState(null); // { blockId, field }
  const [status, setStatus] = useState('idle');
  const [viewerInitAttempt, setViewerInitAttempt] = useState(0);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  const trajectoryThemeProviderRef = useRef(null);
  const trajectoryHighlightComponentRef = useRef(null);
  const viewerGenerationRef = useRef(0);

  useEffect(() => {
    fetchSystem(projectId, systemId)
      .then((data) => {
        setSystem(data);
        const clusters = (data?.metastable_clusters || []).filter((c) => c.path && c.status !== 'failed');
        if (!selectedClusterId && clusters[0]?.cluster_id) setSelectedClusterId(clusters[0].cluster_id);
        const states = Object.entries(data?.states || {})
          .map(([stateId, state]) => ({ ...state, state_id: state?.state_id || stateId }))
          .filter((state) => state.pdb_file);
        if (states[0]?.state_id) setSelectedStateId(states[0].state_id);
      })
      .catch((err) => setError(err.message || 'Failed to load system.'));
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    fetchPottsClusterInfo(projectId, systemId, selectedClusterId)
      .then(setClusterInfo)
      .catch((err) => setError(err.message || 'Failed to load cluster residue metadata.'));
    fetchClusterUiSetups(projectId, systemId, selectedClusterId, { setupType: 'residue_selection' })
      .then((res) => setSetups(Array.isArray(res?.setups) ? res.setups : []))
      .catch(() => setSetups([]));
    setEnergyAnalysisLoading(true);
    Promise.all([
      fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'model_energy' }),
      fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_energy' }),
    ])
      .then(([energyResult, deltaResult]) => {
        setEnergyAnalyses(Array.isArray(energyResult?.analyses) ? energyResult.analyses : []);
        setDeltaEnergyAnalyses(Array.isArray(deltaResult?.analyses) ? deltaResult.analyses : []);
      })
      .catch((err) => setAnimatedEnergyError(err.message || 'Failed to list existing energy analyses.'))
      .finally(() => setEnergyAnalysisLoading(false));
  }, [projectId, systemId, selectedClusterId]);

  const viewerCanInitialize = Boolean(selectedStateId);

  useEffect(() => {
    if (!viewerCanInitialize) return undefined;
    let disposed = false;
    const generation = viewerGenerationRef.current + 1;
    viewerGenerationRef.current = generation;
    const init = async () => {
      if (!containerRef.current || pluginRef.current) return;
      setStatus('initializing');
      try {
        let timedOut = false;
        let timeoutId;
        const pluginPromise = createPluginUI({ target: containerRef.current, render: renderReact18 })
          .then((plugin) => {
            if (timedOut || disposed || generation !== viewerGenerationRef.current) plugin.dispose?.();
            return plugin;
          });
        const timeoutPromise = new Promise((_, reject) => {
          timeoutId = window.setTimeout(() => {
            timedOut = true;
            reject(new Error('Mol* initialization timed out.'));
          }, 30000);
        });
        const plugin = await Promise.race([pluginPromise, timeoutPromise]);
        window.clearTimeout(timeoutId);
        if (disposed || generation !== viewerGenerationRef.current) { plugin.dispose?.(); return; }
        pluginRef.current = plugin;
        setStatus('ready');
      } catch (err) {
        if (disposed || generation !== viewerGenerationRef.current) return;
        setStatus('error');
        setError(err.message || 'Mol* initialization failed.');
      }
    };
    // React StrictMode mounts effects twice in development. Deferring the
    // expensive initialization avoids constructing and disposing two viewers.
    const initTimer = window.setTimeout(init, 0);
    return () => {
      disposed = true;
      viewerGenerationRef.current += 1;
      window.clearTimeout(initTimer);
      try { pluginRef.current?.dispose?.(); } catch { /* noop */ }
      pluginRef.current = null;
    };
  }, [viewerCanInitialize, viewerInitAttempt]);

  const selectedResidues = useMemo(
    () => selectedResiduesFromBlocks(blocks, clusterInfo?.residue_keys || []),
    [blocks, clusterInfo]
  );

  const getBase = useCallback(() => {
    const components = pluginRef.current?.managers?.structure?.hierarchy?.current?.structures?.[0]?.components;
    if (!Array.isArray(components)) return null;
    // Overpaint helpers expect the hierarchy component wrapper, not its cell.
    return components.find((component) => Array.isArray(component?.representations)) || null;
  }, []);

  const applyHighlight = useCallback(async () => {
    const plugin = pluginRef.current;
    const base = getBase();
    if (!plugin || !base || status !== 'ready' || viewMode !== 'reference') return;
    try { await clearStructureOverpaint(plugin, [base]); } catch { /* noop */ }
    if (!selectedResidues.length) return;
    const propFn = MS.struct.atomProperty.macromolecular.auth_seq_id();
    const residueTests = selectedResidues.length === 1
      ? MS.core.rel.eq([propFn, selectedResidues[0]])
      : MS.core.set.has([MS.set(...selectedResidues), propFn]);
    const expression = MS.struct.generator.atomGroups({ 'residue-test': residueTests });
    const rootStructure = plugin.managers.structure.hierarchy.current.structures[0]?.cell?.obj?.data;
    if (rootStructure) {
      const sel = Script.getStructureSelection(expression, rootStructure);
      if (StructureSelection.unionStructure(sel).elementCount === 0) return;
    }
    try {
      await setStructureOverpaint(plugin, [base], hexToInt('#22d3ee'), async (structure) => {
        const sel = Script.getStructureSelection(expression, structure);
        return StructureSelection.toLociWithSourceUnits(sel);
      }, ['cartoon']);
    } catch (err) {
      // Ignore a stale hierarchy while Mol* is replacing the reference model.
      if (pluginRef.current === plugin && status === 'ready') {
        console.warn('Failed to update residue-selection overpaint', err);
      }
    }
  }, [getBase, selectedResidues, status, viewMode]);

  const loadStructure = useCallback(async () => {
    const plugin = pluginRef.current;
    if (!plugin || !selectedStateId) return;
    setBusy(true);
    setError(null);
    try {
      await plugin.clear();
      await plugin.dataTransaction(async () => {
        const url = `/api/v1/projects/${projectId}/systems/${systemId}/structures/${encodeURIComponent(selectedStateId)}`;
        const data = await plugin.builders.data.download({ url: Asset.Url(url), label: selectedStateId }, { state: { isGhost: true } });
        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
      });
      setOverlaySummary(null);
      setWholeTrajectoryLoaded(false);
      setTrajectoryClusterHighlight(null);
    } catch (err) {
      setError(err.message || 'Failed to load structure.');
    } finally {
      setBusy(false);
    }
  }, [projectId, selectedStateId, systemId]);

  useEffect(() => { if (status === 'ready' && selectedStateId && viewMode === 'reference') loadStructure(); }, [status, selectedStateId, viewMode, loadStructure]);
  useEffect(() => { applyHighlight(); }, [applyHighlight]);

  useEffect(() => {
    const plugin = pluginRef.current;
    if (status !== 'ready' || !plugin) return undefined;
    const sub = plugin.behaviors.interaction.click.subscribe((evt) => {
      if (!pickTarget) return;
      const loci = evt?.current?.loci;
      if (!StructureElement.Loci.is(loci)) return;
      const loc = StructureElement.Loci.getFirstLocation(loci);
      if (!loc) return;
      const auth = Number(StructureProperties.residue.auth_seq_id(loc));
      if (!Number.isInteger(auth)) return;
      setBlocks((prev) => prev.map((b) => (b.id === pickTarget.blockId ? { ...b, [pickTarget.field]: String(auth) } : b)));
      setPickTarget(null);
    });
    return () => { try { sub?.unsubscribe?.(); } catch { /* noop */ } };
  }, [pickTarget, status]);

  useEffect(() => {
    if (status !== 'ready' || viewMode !== 'trajectory' || !wholeTrajectoryLoaded) return undefined;
    let previous = -1;
    const updateFrame = () => {
      const model = pluginRef.current?.managers?.structure?.hierarchy?.current?.structures?.[0]?.cell?.obj?.data?.models?.[0];
      if (!model) return;
      const frame = trajectoryFrameIndex(model);
      if (Number.isInteger(frame) && frame !== previous) {
        previous = frame;
        setCurrentTrajectoryFrame(frame);
      }
    };
    updateFrame();
    const timer = window.setInterval(updateFrame, 120);
    return () => window.clearInterval(timer);
  }, [status, viewMode, wholeTrajectoryLoaded]);

  const clusters = (system?.metastable_clusters || []).filter((c) => c.path && c.status !== 'failed');
  const states = Object.entries(system?.states || {})
    .map(([stateId, state]) => ({ ...state, state_id: state?.state_id || stateId }))
    .filter((state) => state.pdb_file);
  const selectedState = states.find((state) => String(state.state_id) === String(selectedStateId)) || null;
  const selectedCluster = clusters.find((cluster) => String(cluster.cluster_id) === String(selectedClusterId)) || null;
  const selectedStateMdSample = useMemo(() => {
    const candidates = (selectedCluster?.samples || []).filter(
      (sample) => String(sample?.type || '').toLowerCase() === 'md_eval'
        && String(sample?.state_id || sample?.summary?.state_id || '') === String(selectedStateId)
    );
    return [...candidates].sort((a, b) => String(a?.created_at || '').localeCompare(String(b?.created_at || ''))).pop() || null;
  }, [selectedCluster, selectedStateId]);

  const eligibleEnergyModelGroups = useMemo(() => {
    if (!selectedStateMdSample?.sample_id) return [];
    const groups = new Map();
    energyAnalyses.forEach((analysis) => {
      if (String(analysis?.sample_id || '') !== String(selectedStateMdSample.sample_id)) return;
      const key = energyModelKey(analysis);
      if (!groups.has(key)) groups.set(key, { key, label: energyModelLabel(analysis), anchor: analysis });
    });
    return Array.from(groups.values()).sort((a, b) => a.label.localeCompare(b.label));
  }, [energyAnalyses, selectedStateMdSample]);

  const eligibleDeltaEnergyAnalyses = useMemo(() => {
    if (!selectedStateMdSample?.sample_id) return [];
    return deltaEnergyAnalyses.filter((analysis) => (
      Array.isArray(analysis?.sample_ids) && analysis.sample_ids.map(String).includes(String(selectedStateMdSample.sample_id))
    ));
  }, [deltaEnergyAnalyses, selectedStateMdSample]);

  useEffect(() => {
    if (energyViewMode === 'single') {
      const allowed = eligibleEnergyModelGroups.map((group) => group.key);
      if (!allowed.includes(selectedEnergyModelKey)) setSelectedEnergyModelKey(allowed[0] || '');
    }
    if (energyViewMode === 'delta') {
      const allowed = eligibleDeltaEnergyAnalyses.map((analysis) => String(analysis.analysis_id));
      if (!allowed.includes(selectedDeltaEnergyId)) setSelectedDeltaEnergyId(allowed[0] || '');
    }
  }, [eligibleDeltaEnergyAnalyses, eligibleEnergyModelGroups, energyViewMode, selectedDeltaEnergyId, selectedEnergyModelKey]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setAnimatedEnergyData(null);
      setAnimatedEnergyError(null);
      if (!selectedStateMdSample?.sample_id || energyViewMode === 'none') return;
      setAnimatedEnergyLoading(true);
      try {
        if (energyViewMode === 'single' && selectedEnergyModelKey) {
          const matching = energyAnalyses.filter((analysis) => energyModelKey(analysis) === selectedEnergyModelKey);
          const payloadRows = await Promise.all(matching.map(async (analysis) => ({
            analysis,
            payload: await fetchClusterAnalysisData(
              projectId,
              systemId,
              selectedClusterId,
              'model_energy',
              analysis.analysis_id,
              String(analysis.sample_id) === String(selectedStateMdSample.sample_id) ? {} : { maxRows: 1500, sampleSeed: 0 }
            ),
          })));
          if (cancelled) return;
          const series = payloadRows.map(({ analysis, payload }) => ({
            id: analysis.sample_id,
            label: analysis.sample_name || analysis.sample_id,
            kind: analysis.sample_type || 'sample',
            values: payload?.data?.energies || [],
          })).filter((entry) => entry.values.length);
          const current = payloadRows.find(({ analysis }) => String(analysis.sample_id) === String(selectedStateMdSample.sample_id));
          setAnimatedEnergyData({
            kind: 'single',
            title: energyModelLabel(current?.analysis || matching[0]),
            series,
            frameIndices: current?.payload?.data?.frame_indices || [],
            frameValues: current?.payload?.data?.energies || [],
          });
        } else if (energyViewMode === 'delta' && selectedDeltaEnergyId) {
          const analysis = eligibleDeltaEnergyAnalyses.find((item) => String(item.analysis_id) === String(selectedDeltaEnergyId));
          const payload = await fetchClusterAnalysisData(
            projectId,
            systemId,
            selectedClusterId,
            'delta_energy',
            selectedDeltaEnergyId,
            { includeFrameValues: true, sampleId: selectedStateMdSample.sample_id }
          );
          if (cancelled) return;
          const bins = payload?.data?.delta_energy_bins || [];
          const hist = payload?.data?.delta_energy_hist || [];
          const ids = payload?.data?.sample_ids || [];
          const labels = payload?.data?.sample_labels || [];
          const types = payload?.data?.sample_types || [];
          setAnimatedEnergyData({
            kind: 'delta',
            title: `${analysis?.model_a_name || analysis?.model_a_id || 'Model A'} − ${analysis?.model_b_name || analysis?.model_b_id || 'Model B'}`,
            series: hist.map((density, index) => ({
              id: ids[index] || `sample-${index}`,
              label: labels[index] || ids[index] || `sample ${index + 1}`,
              kind: types[index] || 'sample',
              bins,
              density,
            })),
            frameIndices: payload?.data?.selected_sample_frame_indices || [],
            frameValues: payload?.data?.selected_sample_delta_energy || [],
          });
        }
      } catch (err) {
        if (!cancelled) setAnimatedEnergyError(err.message || 'Failed to load framewise energy data.');
      } finally {
        if (!cancelled) setAnimatedEnergyLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [eligibleDeltaEnergyAnalyses, energyAnalyses, energyViewMode, projectId, selectedClusterId, selectedDeltaEnergyId, selectedEnergyModelKey, selectedStateMdSample, systemId]);

  const currentFrameEnergy = useMemo(() => {
    const frames = animatedEnergyData?.frameIndices || [];
    const values = animatedEnergyData?.frameValues || [];
    const row = frames.findIndex((frame) => Number(frame) === Number(currentTrajectoryFrame));
    return row >= 0 && Number.isFinite(Number(values[row])) ? Number(values[row]) : null;
  }, [animatedEnergyData, currentTrajectoryFrame]);

  const animatedEnergyPlot = useMemo(() => {
    if (!animatedEnergyData?.series?.length) return null;
    return buildEnergyDistributionPlot({
      series: animatedEnergyData.series,
      mode: 'curves',
      title: animatedEnergyData.kind === 'delta'
        ? `Delta energy · ${animatedEnergyData.title}`
        : `Energy · ${animatedEnergyData.title}`,
      xTitle: animatedEnergyData.kind === 'delta' ? 'ΔE = E_model_A - E_model_B' : 'Energy',
      height: 330,
      background: 'dark',
    });
  }, [animatedEnergyData]);
  useEffect(() => {
    let cancelled = false;
    const residueRows = selectedResidues
      .map((resid) => ({
        resid,
        residueIndex: (clusterInfo?.residue_keys || []).findIndex((key) => residFromKey(key) === resid),
      }))
      .filter((row) => row.residueIndex >= 0);
    if (!selectedClusterId || !selectedStateId || !residueRows.length) {
      setSelectionPieProfiles([]);
      setPieProfilesError(null);
      return () => { cancelled = true; };
    }
    setPieProfilesLoading(true);
    setPieProfilesError(null);
    fetchStateTrajectoryOverlay(projectId, systemId, selectedClusterId, selectedStateId, {
      labels_only: true,
      map_source_frames: false,
      frame_indices: [],
      frame_selection_mode: 'explicit',
      residue_indices: residueRows.map((row) => row.residueIndex),
    })
      .then((payload) => {
        if (cancelled) return;
        const frameLabels = payload?.frame_labels || [];
        const rows = residueRows.map((row) => {
          const values = frameLabels.map((frame) => Number(frame?.clusters?.[String(row.residueIndex)] ?? -1));
          const valid = values.filter((value) => value >= 0);
          const configuredK = Number(clusterInfo?.cluster_counts?.[row.residueIndex] || 0);
          const observedK = valid.length ? Math.max(...valid) + 1 : 0;
          const k = Math.max(configuredK, observedK);
          const counts = Array.from({ length: k }, () => 0);
          valid.forEach((clusterId) => { if (clusterId < counts.length) counts[clusterId] += 1; });
          const probabilities = counts.map((count) => (valid.length ? count / valid.length : 0));
          return {
            ...row,
            profile: {
              n_frames: frameLabels.length,
              node_valid_count: valid.length,
              node_probs: probabilities,
            },
          };
        });
        setSelectionPieProfiles(rows);
      })
      .catch((err) => { if (!cancelled) { setSelectionPieProfiles([]); setPieProfilesError(err.message || 'Failed to load residue cluster pies.'); } })
      .finally(() => { if (!cancelled) setPieProfilesLoading(false); });
    return () => { cancelled = true; };
  }, [clusterInfo, projectId, selectedClusterId, selectedResidues, selectedStateId, systemId]);

  const loadTrajectoryOverlay = useCallback(async ({ align = false } = {}) => {
    if (!selectedClusterId || !selectedStateId || !clusterInfo) return;
    const residueIndex = Number(clusterResidueIndex);
    if (!Number.isInteger(residueIndex) || residueIndex < 0) {
      setError('Select a residue to color.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const frames = frameSelectionMode === 'explicit'
        ? parseFrameSelection(frameSelection, Number(selectedState?.n_frames || 0))
        : [];
      const result = await fetchStateTrajectoryOverlay(projectId, systemId, selectedClusterId, selectedStateId, {
        frame_indices: frames,
        residue_indices: [residueIndex],
        frame_selection_mode: frameSelectionMode,
        max_frames_per_cluster: Number(maxFramesPerCluster),
        alignment_resids: align ? selectedResidues : [],
      });
      const plugin = pluginRef.current;
      if (!plugin) throw new Error('Mol* is not ready.');
      await plugin.clear();
      for (const item of result?.structures || []) {
        await plugin.dataTransaction(async () => {
          const data = await plugin.builders.data.rawData({
            data: item.pdb,
            label: `${selectedStateId} frame ${item.frame_index}`,
          });
          const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
          await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
        });
      }

      const roots = plugin.managers.structure.hierarchy.current.structures || [];
      if (!roots.length) throw new Error('Mol* did not create structures for the selected frames.');
      await plugin.managers.structure.component.clear(roots);
      const residueKey = clusterInfo.residue_keys?.[residueIndex];
      const resid = residFromKey(residueKey);
      if (!Number.isInteger(resid)) throw new Error(`Cannot determine residue number from '${residueKey}'.`);
      const residueExpression = MS.struct.generator.atomGroups({
        'residue-test': MS.core.rel.eq([MS.struct.atomProperty.macromolecular.auth_seq_id(), resid]),
      });
      for (let idx = 0; idx < roots.length; idx += 1) {
        const root = roots[idx];
        const item = result.structures[idx];
        const clusterId = Number(item?.clusters?.[String(residueIndex)] ?? -1);
        const color = clusterId >= 0 ? clusterPieColor(clusterId) : '#64748b';
        const base = await plugin.builders.structure.tryCreateComponentFromExpression(root.cell, MS.struct.generator.all(), `overlay-base-${idx}`);
        if (base) {
          await plugin.builders.structure.representation.addRepresentation(base, {
            type: 'cartoon',
            color: 'uniform',
            colorParams: { value: hexToInt('#d1d5db') },
          });
        }
        const residueComponent = await plugin.builders.structure.tryCreateComponentFromExpression(root.cell, residueExpression, `overlay-residue-${idx}`);
        if (residueComponent) {
          await plugin.builders.structure.representation.addRepresentation(residueComponent, {
            type: 'ball-and-stick',
            typeParams: { ignoreHydrogens: hideLicoriceHydrogens, ignoreHydrogensVariant: 'all' },
            color: 'uniform',
            colorParams: { value: hexToInt(color) },
          });
        }
      }
      setOverlaySummary({ ...result, residueIndex, residueKey });
      setWholeTrajectoryLoaded(false);
      setTrajectoryClusterHighlight(null);
    } catch (err) {
      setError(err.message || 'Failed to load trajectory overlay.');
    } finally {
      setBusy(false);
    }
  }, [clusterInfo, clusterResidueIndex, frameSelection, frameSelectionMode, hideLicoriceHydrogens, maxFramesPerCluster, projectId, selectedClusterId, selectedResidues, selectedState, selectedStateId, systemId]);

  const loadWholeTrajectory = useCallback(async () => {
    if (!selectedState?.trajectory_file) {
      setError('The selected state has no stored trajectory. Upload it first.');
      return;
    }
    const plugin = pluginRef.current;
    if (!plugin) return;
    setBusy(true);
    setError(null);
    try {
      await plugin.clear();
      const topologyUrl = `/api/v1/projects/${projectId}/systems/${systemId}/structures/${encodeURIComponent(selectedStateId)}`;
      const coordinatesUrl = `/api/v1/projects/${projectId}/systems/${systemId}/states/${encodeURIComponent(selectedStateId)}/trajectory/raw`;
      let combinedTrajectory = null;
      await plugin.dataTransaction(async () => {
        const modelData = await plugin.builders.data.download(
          { url: Asset.Url(topologyUrl), label: selectedState.pdb_file || 'structure.pdb' },
          { state: { isGhost: true } }
        );
        const modelTrajectory = await plugin.builders.structure.parseTrajectory(modelData, 'pdb');
        const model = await plugin.builders.structure.createModel(modelTrajectory);
        const coordinateData = await plugin.builders.data.download(
          { url: Asset.Url(coordinatesUrl), isBinary: true, label: selectedState.source_traj || selectedState.trajectory_file },
          { state: { isGhost: true } }
        );
        const format = inferCoordinateFormat(selectedState.source_traj || selectedState.trajectory_file);
        const provider = plugin.dataFormats.get(format);
        if (!provider) throw new Error(`Mol* does not support trajectory format '${format}'.`);
        const coordinates = await provider.parse(plugin, coordinateData);
        combinedTrajectory = await plugin.build().toRoot()
          .apply(StateTransforms.Model.TrajectoryFromModelAndCoordinates, {
            modelRef: model.ref,
            coordinatesRef: coordinates.ref,
          }, { dependsOn: [model.ref, coordinates.ref] })
          .commit();
      });
      // Apply the visual preset only after the trajectory transaction commits.
      // Doing this inside the transaction races Mol*'s hierarchy manager for
      // large coordinate sets and can leave a valid trajectory with no visible
      // structure.
      if (!combinedTrajectory) throw new Error('Mol* did not create the combined trajectory.');
      await plugin.builders.structure.hierarchy.applyPreset(combinedTrajectory, 'default');
      await new Promise((resolve) => window.setTimeout(resolve, 0));
      const loadedStructure = plugin.managers.structure.hierarchy.current.structures?.[0]?.cell?.obj?.data;
      if (!loadedStructure || !loadedStructure.elementCount) {
        throw new Error('Mol* loaded the trajectory coordinates but did not create a visible structure.');
      }
      plugin.managers.camera.reset(undefined, 0);
      plugin.canvas3d?.requestDraw?.();
      setWholeTrajectoryLoaded(true);
      setCurrentTrajectoryFrame(0);
      setOverlaySummary(null);
      setTrajectoryClusterHighlight(null);
    } catch (err) {
      setError(err.message || 'Failed to load the complete trajectory into Mol*.');
      setWholeTrajectoryLoaded(false);
    } finally {
      setBusy(false);
    }
  }, [projectId, selectedState, selectedStateId, systemId]);

  const highlightAnimatedTrajectory = useCallback(async () => {
    if (!wholeTrajectoryLoaded || !selectedResidues.length || !selectedClusterId) return;
    const plugin = pluginRef.current;
    if (!plugin) return;
    const residueRows = selectedResidues
      .map((resid) => ({
        resid,
        residueIndex: (clusterInfo?.residue_keys || []).findIndex((key) => residFromKey(key) === resid),
      }))
      .filter((row) => row.residueIndex >= 0);
    if (!residueRows.length) {
      setError('None of the currently selected residues belongs to this cluster definition.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const labelsPayload = await fetchStateTrajectoryOverlay(projectId, systemId, selectedClusterId, selectedStateId, {
        labels_only: true,
        frame_indices: [],
        frame_selection_mode: 'explicit',
        residue_indices: residueRows.map((row) => row.residueIndex),
      });
      if (trajectoryHighlightComponentRef.current) {
        await plugin.build().delete(trajectoryHighlightComponentRef.current).commit();
        trajectoryHighlightComponentRef.current = null;
      }
      const registry = plugin.representation.structure.themes.colorThemeRegistry;
      if (trajectoryThemeProviderRef.current) registry.remove(trajectoryThemeProviderRef.current);
      const residueIndexByResid = new Map(residueRows.map((row) => [row.resid, row.residueIndex]));
      const themeProvider = createTrajectoryClusterTheme(labelsPayload.frame_labels || [], residueIndexByResid);
      registry.add(themeProvider);
      trajectoryThemeProviderRef.current = themeProvider;

      const root = plugin.managers.structure.hierarchy.current.structures?.[0];
      if (!root?.cell) throw new Error('No animated trajectory structure is loaded.');
      const residValues = residueRows.map((row) => row.resid);
      const residueTest = residValues.length === 1
        ? MS.core.rel.eq([MS.struct.atomProperty.macromolecular.auth_seq_id(), residValues[0]])
        : MS.core.set.has([MS.set(...residValues), MS.struct.atomProperty.macromolecular.auth_seq_id()]);
      const component = await plugin.builders.structure.tryCreateComponentFromExpression(
        root.cell,
        MS.struct.generator.atomGroups({ 'residue-test': residueTest }),
        'phase-animated-clusters'
      );
      if (!component) throw new Error('Failed to create the animated residue selection.');
      trajectoryHighlightComponentRef.current = component.ref;
      await plugin.builders.structure.representation.addRepresentation(component, {
        type: 'ball-and-stick',
        typeParams: { ignoreHydrogens: hideLicoriceHydrogens, ignoreHydrogensVariant: 'all' },
        color: themeProvider.name,
      });
      setTrajectoryClusterHighlight({ residueRows, frameCount: labelsPayload.frame_labels?.length || 0 });
    } catch (err) {
      setError(err.message || 'Failed to activate trajectory cluster coloring.');
    } finally {
      setBusy(false);
    }
  }, [clusterInfo, hideLicoriceHydrogens, projectId, selectedClusterId, selectedResidues, selectedStateId, systemId, wholeTrajectoryLoaded]);

  const uploadTrajectory = useCallback(async () => {
    if (!trajectoryFile || !selectedStateId) return;
    setBusy(true);
    setError(null);
    setUploadProgress(0);
    try {
      const form = new FormData();
      form.append('trajectory', trajectoryFile);
      form.append('stride', '1');
      form.append('build_descriptors_after_upload', 'true');
      await uploadStateTrajectory(projectId, systemId, selectedStateId, form, {
        onUploadProgress: setUploadProgress,
      });
      if (selectedClusterId) await assignClusterStates(projectId, systemId, selectedClusterId, [selectedStateId]);
      const refreshed = await fetchSystem(projectId, systemId);
      setSystem(refreshed);
      setTrajectoryFile(null);
      setUploadProgress(null);
    } catch (err) {
      setError(err.message || 'Failed to store and assign trajectory.');
    } finally {
      setBusy(false);
    }
  }, [projectId, selectedClusterId, selectedStateId, systemId, trajectoryFile]);

  const updateBlock = (id, patch) => setBlocks((prev) => prev.map((b) => (b.id === id ? { ...b, ...patch } : b)));
  const loadSetup = (setup) => {
    setSelectedSetupId(setup.setup_id);
    setName(setup.name || setup.setup_id);
    const loadedBlocks = Array.isArray(setup?.payload?.blocks) ? setup.payload.blocks : [];
    setBlocks(loadedBlocks.length ? loadedBlocks.map((b) => ({ ...b, id: b.id || crypto.randomUUID(), residuesText: (b.residues || []).join(',') })) : []);
  };

  const saveSelection = async () => {
    if (!selectedClusterId) return;
    setBusy(true);
    setError(null);
    try {
      const normalized = blocks.map(normalizeBlock);
      const payload = {
        blocks: normalized,
        selected_residues: selectedResidues,
        selected_residue_indices: selectedResidues.map((resid) => (clusterInfo?.residue_keys || []).findIndex((k) => residFromKey(k) === resid)).filter((i) => i >= 0),
      };
      const saved = await saveClusterUiSetup(projectId, systemId, selectedClusterId, {
        setup_id: selectedSetupId || undefined,
        name,
        setup_type: 'residue_selection',
        page: 'residue_selection',
        payload,
      });
      setSelectedSetupId(saved.setup_id);
      const res = await fetchClusterUiSetups(projectId, systemId, selectedClusterId, { setupType: 'residue_selection' });
      setSetups(Array.isArray(res?.setups) ? res.setups : []);
    } catch (err) {
      setError(err.message || 'Failed to save selection.');
    } finally {
      setBusy(false);
    }
  };

  const deleteSelection = async () => {
    if (!selectedClusterId || !selectedSetupId) return;
    setBusy(true);
    try {
      await deleteClusterUiSetup(projectId, systemId, selectedClusterId, selectedSetupId);
      setSelectedSetupId('');
      setName('New residue selection');
      setBlocks([{ id: crypto.randomUUID(), name: 'block_A', start: '', end: '', residuesText: '' }]);
      const res = await fetchClusterUiSetups(projectId, systemId, selectedClusterId, { setupType: 'residue_selection' });
      setSetups(Array.isArray(res?.setups) ? res.setups : []);
    } catch (err) {
      setError(err.message || 'Failed to delete selection.');
    } finally {
      setBusy(false);
    }
  };

  if (!system) return <Loader message="Loading residue selection editor..." />;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Structures & residue selections</h1>
          <p className="text-sm text-gray-400">Build reusable selections or overlay trajectory frames with residues colored by their assigned clusters.</p>
        </div>
        <button type="button" onClick={saveSelection} disabled={busy || !name.trim()} className="rounded-md bg-cyan-500 px-4 py-2 text-sm font-semibold text-black disabled:opacity-50">Save selection</button>
      </div>
      {error ? <ErrorMessage message={error} /> : null}
      <div className="grid grid-cols-1 xl:grid-cols-[360px_1fr] gap-4">
        <aside className="space-y-3 rounded-lg border border-gray-800 bg-gray-900/60 p-3">
          <label className="block text-xs text-gray-400">Cluster</label>
          <select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
            {clusters.map((c) => <option key={c.cluster_id} value={c.cluster_id}>{c.name || c.cluster_id}</option>)}
          </select>
          <label className="block text-xs text-gray-400">Saved selections</label>
          <select value={selectedSetupId} onChange={(e) => { const setup = setups.find((s) => s.setup_id === e.target.value); if (setup) loadSetup(setup); else setSelectedSetupId(''); }} className="w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
            <option value="">New selection</option>
            {setups.map((s) => <option key={s.setup_id} value={s.setup_id}>{s.name || s.setup_id}</option>)}
          </select>
          <input value={name} onChange={(e) => setName(e.target.value)} className="w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100" placeholder="Selection name" />
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold uppercase tracking-wide text-gray-400">OR blocks</div>
              <button type="button" onClick={() => setBlocks((prev) => [...prev, { id: crypto.randomUUID(), name: `block_${prev.length + 1}`, start: '', end: '', residuesText: '' }])} className="text-xs text-cyan-300">Add block</button>
            </div>
            {blocks.map((block) => (
              <div key={block.id} className="rounded border border-gray-800 bg-gray-950/60 p-2 space-y-2">
                <input value={block.name} onChange={(e) => updateBlock(block.id, { name: e.target.value })} className="w-full rounded bg-gray-900 border border-gray-700 px-2 py-1.5 text-xs text-gray-100" />
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="text-[11px] text-gray-500">Start resid</label>
                    <div className="flex gap-1"><input value={block.start} onChange={(e) => updateBlock(block.id, { start: e.target.value })} className="min-w-0 flex-1 rounded bg-gray-900 border border-gray-700 px-2 py-1.5 text-xs text-gray-100" /><button type="button" onClick={() => setPickTarget({ blockId: block.id, field: 'start' })} className="rounded border border-gray-700 px-2 text-xs text-gray-300">pick</button></div>
                  </div>
                  <div>
                    <label className="text-[11px] text-gray-500">End resid</label>
                    <div className="flex gap-1"><input value={block.end} onChange={(e) => updateBlock(block.id, { end: e.target.value })} className="min-w-0 flex-1 rounded bg-gray-900 border border-gray-700 px-2 py-1.5 text-xs text-gray-100" /><button type="button" onClick={() => setPickTarget({ blockId: block.id, field: 'end' })} className="rounded border border-gray-700 px-2 text-xs text-gray-300">pick</button></div>
                  </div>
                </div>
                <input value={block.residuesText || ''} onChange={(e) => updateBlock(block.id, { residuesText: e.target.value })} className="w-full rounded bg-gray-900 border border-gray-700 px-2 py-1.5 text-xs text-gray-100" placeholder="Extra residues, e.g. 45, 89, 131" />
                <button type="button" onClick={() => setBlocks((prev) => prev.filter((b) => b.id !== block.id))} className="text-xs text-red-300">Remove block</button>
              </div>
            ))}
          </div>
          <div className="rounded border border-gray-800 bg-gray-950/50 p-2 text-xs text-gray-300">
            Selected residues: {selectedResidues.length ? selectedResidues.join(', ') : 'none'}
          </div>
          {selectedSetupId ? <button type="button" onClick={deleteSelection} disabled={busy} className="rounded border border-red-500/60 px-3 py-2 text-xs text-red-300">Delete selection</button> : null}
        </aside>
        <main className="space-y-3">
          <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-3">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => setViewMode('reference')} className={`rounded border px-3 py-2 text-xs ${viewMode === 'reference' ? 'border-cyan-500 bg-cyan-950/50 text-cyan-400' : 'border-gray-700 text-gray-300'}`}>Reference structure</button>
              <button type="button" onClick={() => setViewMode('overlay')} className={`rounded border px-3 py-2 text-xs ${viewMode === 'overlay' ? 'border-cyan-500 bg-cyan-950/50 text-cyan-400' : 'border-gray-700 text-gray-300'}`}>Trajectory overlay (manual load)</button>
              <button type="button" onClick={() => setViewMode('trajectory')} className={`rounded border px-3 py-2 text-xs ${viewMode === 'trajectory' ? 'border-cyan-500 bg-cyan-950/50 text-cyan-400' : 'border-gray-700 text-gray-300'}`}>Animated full trajectory</button>
              <span className="text-[11px] text-gray-500">Initially only the state PDB is loaded. The complete XTC is never downloaded into Mol* automatically.</span>
            </div>
            <div className="mb-3 grid gap-3 rounded border border-gray-800 bg-gray-950/40 p-3 lg:grid-cols-2">
              <label className="text-xs text-gray-400">
                State / topology
                <select value={selectedStateId} onChange={(e) => { setSelectedStateId(e.target.value); setOverlaySummary(null); setWholeTrajectoryLoaded(false); setTrajectoryClusterHighlight(null); }} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
                  {states.map((s) => <option key={s.state_id} value={s.state_id}>{s.name || s.state_id} {s.trajectory_file ? `(${s.n_frames || '?'} frames)` : '(PDB only)'}</option>)}
                </select>
              </label>
              {viewMode === 'reference' ? (
                <div className="flex items-end gap-2">
                  <button type="button" onClick={loadStructure} className="rounded border border-gray-700 px-3 py-2 text-xs text-gray-100">Reload structure</button>
                  {pickTarget ? <span className="text-xs text-cyan-300">Click a residue in Mol* to fill {pickTarget.field}.</span> : null}
                </div>
              ) : viewMode === 'overlay' ? (
                <>
                  <label className="text-xs text-gray-400">
                    Frame selection
                    <select value={frameSelectionMode} onChange={(e) => setFrameSelectionMode(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
                      <option value="explicit">Explicit frame range</option>
                      <option value="per_cluster">Up to X frames per residue cluster</option>
                    </select>
                  </label>
                  {frameSelectionMode === 'explicit' ? (
                    <label className="text-xs text-gray-400">
                      Frames (end-exclusive, maximum 500)
                      <input value={frameSelection} onChange={(e) => setFrameSelection(e.target.value)} placeholder="0-100 or 0,5,10" className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100" />
                      <span className="mt-1 block text-[11px] text-gray-500">`0-100` loads 100 frames; `500-1000` loads 500.</span>
                    </label>
                  ) : (
                    <label className="text-xs text-gray-400">
                      Maximum frames per cluster
                      <input type="number" min="1" max="500" value={maxFramesPerCluster} onChange={(e) => setMaxFramesPerCluster(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100" />
                      <span className="mt-1 block text-[11px] text-gray-500">Frames are spread across each assigned cluster; the complete overlay remains capped at 500.</span>
                    </label>
                  )}
                  <label className="text-xs text-gray-400">
                    Residue colored by cluster
                    <select value={clusterResidueIndex} onChange={(e) => setClusterResidueIndex(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
                      {(clusterInfo?.residue_keys || []).map((key, index) => (
                        <option key={`${key}-${index}`} value={index}>
                          {clusterInfo?.residue_display_labels?.[index] || key} · {clusterInfo?.cluster_counts?.[index] || '?'} clusters
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-gray-300">
                    <input type="checkbox" checked={hideLicoriceHydrogens} onChange={(e) => setHideLicoriceHydrogens(e.target.checked)} className="h-4 w-4 rounded border-gray-600 bg-gray-900 text-cyan-500" />
                    Hide hydrogens in residue licorice
                  </label>
                  <div className="flex flex-wrap items-end gap-2">
                    <button type="button" onClick={() => loadTrajectoryOverlay()} disabled={busy} className="rounded bg-cyan-600 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-500 disabled:opacity-50">Refresh overlay</button>
                    <button type="button" onClick={() => loadTrajectoryOverlay({ align: true })} disabled={busy || selectedResidues.length === 0} title={selectedResidues.length ? `Align all frames on ${selectedResidues.length} currently selected residue(s).` : 'Create or load a residue selection first.'} className="rounded border border-cyan-500/70 px-3 py-2 text-xs text-cyan-200 disabled:opacity-40">Align on selection & refresh</button>
                  </div>
                </>
              ) : (
                <>
                  <div className="space-y-1 text-xs text-gray-400">
                    <p>This explicitly downloads the complete stored trajectory into Mol*. Use Mol*'s native <span className="font-semibold text-gray-200">Animate</span> control to play it.</p>
                    {wholeTrajectoryLoaded ? <p className="text-emerald-300">Complete trajectory loaded.</p> : null}
                  </div>
                  <label className="flex items-center gap-2 text-xs text-gray-300">
                    <input type="checkbox" checked={hideLicoriceHydrogens} onChange={(e) => setHideLicoriceHydrogens(e.target.checked)} className="h-4 w-4 rounded border-gray-600 bg-gray-900 text-cyan-500" />
                    Hide hydrogens in highlighted residues
                  </label>
                  <div className="flex flex-wrap items-end gap-2">
                    <button type="button" onClick={loadWholeTrajectory} disabled={busy || !selectedState?.trajectory_file} className="rounded bg-cyan-600 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-500 disabled:opacity-50">Load complete trajectory</button>
                    <button type="button" onClick={highlightAnimatedTrajectory} disabled={busy || !wholeTrajectoryLoaded || selectedResidues.length === 0} title={selectedResidues.length ? 'Color the current selection by its cluster at every animated frame.' : 'Create or load a residue selection first.'} className="rounded border border-cyan-500/70 px-3 py-2 text-xs text-cyan-200 disabled:opacity-40">Highlight selection by cluster</button>
                  </div>
                  {trajectoryClusterHighlight ? <p className="text-xs text-cyan-300">Dynamic coloring active for {trajectoryClusterHighlight.residueRows.length} residue(s), with assignments for {trajectoryClusterHighlight.frameCount} descriptor frames.</p> : null}
                  <label className="text-xs text-gray-400">
                    Current-frame energy overlay
                    <select value={energyViewMode} onChange={(e) => setEnergyViewMode(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
                      <option value="none">None</option>
                      <option value="single" disabled={!eligibleEnergyModelGroups.length}>Potts energy</option>
                      <option value="delta" disabled={!eligibleDeltaEnergyAnalyses.length}>Pairwise delta energy</option>
                    </select>
                  </label>
                  {energyViewMode === 'single' ? (
                    <label className="text-xs text-gray-400">
                      Existing Sampling Explorer energy model
                      <select value={selectedEnergyModelKey} onChange={(e) => setSelectedEnergyModelKey(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
                        {eligibleEnergyModelGroups.map((group) => <option key={group.key} value={group.key}>{group.label}</option>)}
                      </select>
                    </label>
                  ) : null}
                  {energyViewMode === 'delta' ? (
                    <label className="text-xs text-gray-400">
                      Existing model-pair energy analysis
                      <select value={selectedDeltaEnergyId} onChange={(e) => setSelectedDeltaEnergyId(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-2 text-sm text-gray-100">
                        {eligibleDeltaEnergyAnalyses.map((analysis) => (
                          <option key={analysis.analysis_id} value={analysis.analysis_id}>
                            {analysis.model_a_name || analysis.model_a_id || 'Model A'} − {analysis.model_b_name || analysis.model_b_id || 'Model B'}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : null}
                  {!energyAnalysisLoading && !eligibleEnergyModelGroups.length && !eligibleDeltaEnergyAnalyses.length ? (
                    <p className="text-xs text-gray-500">No existing energy analysis includes this state's assigned MD sample. Run Sampling Explorer or model-pair delta energy first.</p>
                  ) : null}
                </>
              )}
            </div>
            {viewMode !== 'reference' && !selectedState?.trajectory_file ? (
              <div className="mb-3 rounded border border-amber-500/40 bg-amber-950/20 p-3">
                <p className="text-xs text-amber-200">This state only has its PDB frame. Upload a matching trajectory; PHASE will store it, rebuild descriptors, and assign its frames to the selected cluster.</p>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <input type="file" accept=".xtc,.dcd,.trr,.nc,.nctraj,.pdb" onChange={(e) => setTrajectoryFile(e.target.files?.[0] || null)} className="min-w-0 flex-1 text-xs text-gray-300" />
                  <button type="button" disabled={!trajectoryFile || busy} onClick={uploadTrajectory} className="rounded border border-amber-400/60 px-3 py-2 text-xs text-amber-100 disabled:opacity-50">Upload, build & assign</button>
                  {uploadProgress !== null ? <span className="text-xs text-gray-300">{uploadProgress}% uploaded</span> : null}
                </div>
              </div>
            ) : null}
            {viewMode === 'overlay' && overlaySummary ? (
              <div className="mb-3 rounded border border-gray-800 bg-gray-950/50 p-3 text-xs text-gray-300">
                <div className="font-semibold text-gray-100">{overlaySummary.residueKey} across {overlaySummary.structures?.length || 0} structures</div>
                {overlaySummary.frame_selection_mode === 'per_cluster' ? <p className="mt-1 text-gray-400">Balanced frame selection: {Object.entries(overlaySummary.selected_cluster_counts || {}).map(([clusterId, count]) => `c${clusterId}: ${count}`).join(' · ')}</p> : null}
                {overlaySummary.alignment_resids?.length ? <p className="mt-1 text-cyan-300">Aligned on current residue selection: {overlaySummary.alignment_resids.join(', ')}</p> : null}
                <div className="mt-2 flex flex-wrap gap-3">
                  {Array.from(new Set((overlaySummary.structures || []).map((item) => Number(item?.clusters?.[String(overlaySummary.residueIndex)] ?? -1)))).sort((a, b) => a - b).map((clusterId) => (
                    <span key={clusterId} className="inline-flex items-center gap-1.5">
                      <span className="h-3 w-3 rounded-full" style={{ backgroundColor: clusterId >= 0 ? clusterPieColor(clusterId) : '#64748b' }} />
                      {clusterId >= 0 ? `cluster ${clusterId}` : 'unassigned'}
                    </span>
                  ))}
                </div>
                <p className="mt-2 text-gray-500">Each frame is a separate superposed structure. The selected residue uses ball-and-stick (licorice-style) rendering; the protein backbone remains an opaque cartoon.</p>
              </div>
            ) : null}
            <div className="relative h-[70vh] min-h-[520px] overflow-hidden rounded border border-gray-800 bg-black/30">
              {(status === 'initializing' || busy) ? <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50"><Loader message={busy ? (uploadProgress !== null ? 'Uploading and assigning trajectory...' : (viewMode === 'reference' ? 'Loading reference PDB...' : 'Loading selected trajectory frames...')) : 'Initializing Mol*...'} /></div> : null}
              {status === 'error' ? (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-black/70 px-6 text-center">
                  <p className="max-w-xl text-sm text-red-200">Mol* could not initialize. No trajectory has been requested at this stage.</p>
                  <button type="button" onClick={() => { setError(null); setStatus('initializing'); setViewerInitAttempt((value) => value + 1); }} className="rounded border border-cyan-500 px-3 py-2 text-xs text-cyan-200">Retry viewer</button>
                </div>
              ) : null}
              <div ref={containerRef} className="h-full w-full" />
            </div>
          </div>
          {viewMode === 'trajectory' && energyViewMode !== 'none' ? (
            <section className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold text-white">Animated-frame energy</h2>
                  <p className="mt-1 text-xs text-gray-400">
                    Mol* frame {currentTrajectoryFrame}. The dashed amber marker follows the currently displayed trajectory frame.
                  </p>
                </div>
                <div className="rounded border border-amber-500/40 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">
                  {currentFrameEnergy === null ? 'This frame has no stored energy value' : `${animatedEnergyData?.kind === 'delta' ? 'ΔE' : 'E'} = ${currentFrameEnergy.toFixed(4)}`}
                </div>
              </div>
              {animatedEnergyLoading ? <div className="py-8"><Loader message="Loading existing energy analysis..." /></div> : null}
              {animatedEnergyError ? <div className="mt-3"><ErrorMessage message={animatedEnergyError} /></div> : null}
              {!animatedEnergyLoading && animatedEnergyPlot ? (
                <div className="mt-3 overflow-hidden rounded border border-gray-800 bg-gray-950/40">
                  <EnergyDistributionPlot
                    plot={animatedEnergyPlot}
                    height={330}
                    frameMarker={currentFrameEnergy === null ? null : {
                      value: currentFrameEnergy,
                      label: `Frame ${currentTrajectoryFrame}: ${animatedEnergyData?.kind === 'delta' ? 'ΔE' : 'E'} ${currentFrameEnergy.toFixed(4)}`,
                    }}
                  />
                </div>
              ) : null}
            </section>
          ) : null}
          <section className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
            <div>
              <h2 className="text-sm font-semibold text-white">Selected-residue cluster distributions</h2>
              <p className="mt-1 text-xs text-gray-400">Assigned-cluster populations in {selectedState?.name || selectedStateId} for every residue in the current selection.</p>
            </div>
            {pieProfilesLoading ? <div className="py-6"><Loader message="Loading cluster distributions..." /></div> : null}
            {!pieProfilesLoading && selectedResidues.length === 0 ? <p className="mt-3 text-xs text-gray-500">Create or load a residue selection to show pie charts.</p> : null}
            {!pieProfilesLoading && pieProfilesError ? <p className="mt-3 text-xs text-amber-300">{pieProfilesError}</p> : null}
            {!pieProfilesLoading && selectionPieProfiles.length ? (
              <div className="mt-4 grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                {selectionPieProfiles.map(({ resid, residueIndex, profile }) => {
                  const slices = (profile?.node_probs || []).map((value, clusterId) => ({
                    label: `c${clusterId}`,
                    clusterId,
                    value: Number(value) || 0,
                    color: clusterPieColor(clusterId),
                  })).filter((slice) => slice.value > 0);
                  const label = clusterInfo?.residue_display_labels?.[residueIndex] || clusterInfo?.residue_keys?.[residueIndex] || `res_${resid}`;
                  return (
                    <article key={`${resid}:${residueIndex}`} className="rounded border border-gray-800 bg-gray-950/50 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <h3 className="text-xs font-semibold text-gray-100">{label}</h3>
                        <span className="text-[11px] text-gray-500">valid {profile?.node_valid_count || 0} / {profile?.n_frames || 0}</span>
                      </div>
                      <div className="mt-2 flex items-start gap-3">
                        <ClusterPieChart slices={slices} size={112} />
                        <div className="min-w-0 flex-1 space-y-1 text-[11px]">
                          {slices.map((slice) => (
                            <div key={`${residueIndex}:${slice.clusterId}`} className="flex items-center gap-2 text-gray-300">
                              <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: slice.color }} />
                              <span>{slice.label}</span>
                              <span className="ml-auto text-gray-400">{(100 * slice.value).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </article>
                  );
                })}
              </div>
            ) : null}
          </section>
        </main>
      </div>
    </div>
  );
}
