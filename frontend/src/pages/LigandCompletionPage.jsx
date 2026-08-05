import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { CircleHelp, Play, Plus, RefreshCw, X } from 'lucide-react';
import Plot from 'react-plotly.js';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import {
  deleteClusterAnalysis,
  fetchClusterAnalyses,
  fetchClusterAnalysisData,
  fetchClusterUiSetups,
  fetchSystem,
} from '../api/projects';
import { fetchJobStatus, submitLigandCompletionJob } from '../api/jobs';

function parseCsvMixed(input) {
  return String(input || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map((token) => {
      if (/^-?\d+$/.test(token)) return Number(token);
      return token;
    });
}

function parseCsvFloats(input) {
  return String(input || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map((token) => Number(token))
    .filter((v) => Number.isFinite(v));
}

function mean(values) {
  const arr = (values || []).map((v) => Number(v)).filter((v) => Number.isFinite(v));
  if (!arr.length) return null;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(values) {
  const arr = (values || []).map((v) => Number(v)).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!arr.length) return null;
  const m = Math.floor(arr.length / 2);
  return arr.length % 2 ? arr[m] : 0.5 * (arr[m - 1] + arr[m]);
}

export default function LigandCompletionPage() {
  const { projectId, systemId } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState(searchParams.get('cluster_id') || '');
  const [modelAId, setModelAId] = useState('');
  const [modelBId, setModelBId] = useState('');
  const [mdSampleId, setMdSampleId] = useState('');
  const [refSampleAId, setRefSampleAId] = useState('');
  const [refSampleBId, setRefSampleBId] = useState('');
  const [constrainedResidues, setConstrainedResidues] = useState('');

  const [sampler, setSampler] = useState('sa');
  const [lambdaValues, setLambdaValues] = useState('0,0.25,0.5,1,2,4,8');
  const [nStartFrames, setNStartFrames] = useState(100);
  const [nSamplesPerFrame, setNSamplesPerFrame] = useState(100);
  const [nSteps, setNSteps] = useState(1000);
  const [tailSteps, setTailSteps] = useState(200);
  const [targetWindowSize, setTargetWindowSize] = useState(11);
  const [targetPseudocount, setTargetPseudocount] = useState(1e-3);
  const [epsilonLogpenalty, setEpsilonLogpenalty] = useState(1e-8);
  const [constraintWeightMode, setConstraintWeightMode] = useState('uniform');
  const [constraintWeights, setConstraintWeights] = useState('');
  const [constraintWeightMin, setConstraintWeightMin] = useState(0.0);
  const [constraintWeightMax, setConstraintWeightMax] = useState(1.0);
  const [constraintSourceMode, setConstraintSourceMode] = useState('manual');
  const [deltaJsExperimentId, setDeltaJsExperimentId] = useState('');
  const [deltaJsFilterSetupId, setDeltaJsFilterSetupId] = useState('');
  const [deltaJsFilterEdgeAlpha, setDeltaJsFilterEdgeAlpha] = useState(0.75);
  const [constraintDeltaJsSampleId, setConstraintDeltaJsSampleId] = useState('');
  const [constraintAutoTopK, setConstraintAutoTopK] = useState(12);
  const [constraintAutoEdgeAlpha, setConstraintAutoEdgeAlpha] = useState(0.3);
  const [constraintAutoExcludeSuccess, setConstraintAutoExcludeSuccess] = useState(true);
  const [gibbsBeta, setGibbsBeta] = useState(1.0);
  const [saBetaHot, setSaBetaHot] = useState(0.8);
  const [saBetaCold, setSaBetaCold] = useState(50.0);
  const [saSchedule, setSaSchedule] = useState('geom');
  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [keepInvalid, setKeepInvalid] = useState(false);
  const [successMetricMode, setSuccessMetricMode] = useState('deltae');
  const [deltaJsDResidueMin, setDeltaJsDResidueMin] = useState(0.0);
  const [deltaJsDResidueMax, setDeltaJsDResidueMax] = useState('');
  const [deltaJsDEdgeMin, setDeltaJsDEdgeMin] = useState(0.0);
  const [deltaJsDEdgeMax, setDeltaJsDEdgeMax] = useState('');
  const [deltaJsNodeEdgeAlpha, setDeltaJsNodeEdgeAlpha] = useState('');
  const [jsSuccessThreshold, setJsSuccessThreshold] = useState(0.15);
  const [jsSuccessMargin, setJsSuccessMargin] = useState(0.02);
  const [deltaEMargin, setDeltaEMargin] = useState(0.0);
  const [completionTargetSuccess, setCompletionTargetSuccess] = useState(0.7);
  const [seed, setSeed] = useState(0);

  const [analyses, setAnalyses] = useState([]);
  const [analysesLoading, setAnalysesLoading] = useState(false);
  const [analysesError, setAnalysesError] = useState(null);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState(searchParams.get('analysis_id') || '');
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisDataLoading, setAnalysisDataLoading] = useState(false);
  const [analysisDataError, setAnalysisDataError] = useState(null);
  const analysisDataCacheRef = useRef({});
  const analysisDataInFlightRef = useRef({});
  const [deltaJsAnalyses, setDeltaJsAnalyses] = useState([]);
  const [deltaJsAnalysesError, setDeltaJsAnalysesError] = useState(null);
  const [deltaJsFilterSetups, setDeltaJsFilterSetups] = useState([]);
  const [deltaJsFilterSetupsError, setDeltaJsFilterSetupsError] = useState(null);

  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const [runPanelOpen, setRunPanelOpen] = useState(false);
  const [deletingAnalysis, setDeletingAnalysis] = useState(false);

  const [wCompletion, setWCompletion] = useState(1.0);
  const [wRaw, setWRaw] = useState(0.5);
  const [wNovelty, setWNovelty] = useState(0.5);

  useEffect(() => {
    const loadSystem = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        const data = await fetchSystem(projectId, systemId);
        setSystem(data);
      } catch (err) {
        setSystemError(err.message || 'Failed to load system.');
      } finally {
        setLoadingSystem(false);
      }
    };
    loadSystem();
  }, [projectId, systemId]);

  const clusterOptions = useMemo(
    () => (system?.metastable_clusters || []).filter((run) => run.path && run.status !== 'failed'),
    [system]
  );
  const selectedCluster = useMemo(
    () => clusterOptions.find((c) => c.cluster_id === selectedClusterId) || null,
    [clusterOptions, selectedClusterId]
  );
  const modelOptions = useMemo(() => selectedCluster?.potts_models || [], [selectedCluster]);
  const mdSamples = useMemo(
    () => (selectedCluster?.samples || []).filter((s) => String(s?.type || '') === 'md_eval'),
    [selectedCluster]
  );

  useEffect(() => {
    if (!clusterOptions.length) return;
    if (!selectedClusterId || !clusterOptions.some((c) => c.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusterOptions[0].cluster_id);
    }
  }, [clusterOptions, selectedClusterId]);

  useEffect(() => {
    if (!modelOptions.length) {
      setModelAId('');
      setModelBId('');
      return;
    }
    if (!modelAId || !modelOptions.some((m) => m.model_id === modelAId)) {
      setModelAId(modelOptions[0].model_id);
    }
    if (!modelBId || !modelOptions.some((m) => m.model_id === modelBId)) {
      setModelBId(modelOptions[Math.min(1, modelOptions.length - 1)].model_id);
    }
  }, [modelOptions, modelAId, modelBId]);

  useEffect(() => {
    if (!mdSamples.length) {
      setMdSampleId('');
      setRefSampleAId('');
      setRefSampleBId('');
      setConstraintDeltaJsSampleId('');
      return;
    }
    if (!mdSampleId || !mdSamples.some((s) => s.sample_id === mdSampleId)) {
      setMdSampleId(mdSamples[0].sample_id);
    }
    if (refSampleAId && !mdSamples.some((s) => s.sample_id === refSampleAId)) setRefSampleAId('');
    if (refSampleBId && !mdSamples.some((s) => s.sample_id === refSampleBId)) setRefSampleBId('');
    if (
      !constraintDeltaJsSampleId ||
      !mdSamples.some((s) => s.sample_id === constraintDeltaJsSampleId)
    ) {
      setConstraintDeltaJsSampleId(mdSampleId || mdSamples[0].sample_id);
    }
  }, [mdSamples, mdSampleId, refSampleAId, refSampleBId, constraintDeltaJsSampleId]);

  useEffect(() => {
    const params = new URLSearchParams(searchParams);
    if (selectedClusterId) params.set('cluster_id', selectedClusterId);
    else params.delete('cluster_id');
    if (selectedAnalysisId) params.set('analysis_id', selectedAnalysisId);
    else params.delete('analysis_id');
    if (params.toString() !== searchParams.toString()) {
      setSearchParams(params, { replace: true });
    }
  }, [selectedClusterId, selectedAnalysisId, searchParams, setSearchParams]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return [];
    setAnalysesLoading(true);
    setAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'ligand_completion' });
      const list = Array.isArray(data?.analyses) ? data.analyses : [];
      setAnalyses(list);
      setSelectedAnalysisId((prev) => {
        if (prev && list.some((a) => a.analysis_id === prev)) return prev;
        return list[0]?.analysis_id || '';
      });
      return list;
    } catch (err) {
      setAnalysesError(err.message || 'Failed to load analyses.');
      setAnalyses([]);
      return [];
    } finally {
      setAnalysesLoading(false);
    }
  }, [projectId, systemId, selectedClusterId]);

  const loadDeltaJsAnalyses = useCallback(async () => {
    if (!selectedClusterId) return [];
    setDeltaJsAnalysesError(null);
    try {
      const data = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'delta_js' });
      const list = Array.isArray(data?.analyses) ? data.analyses : [];
      setDeltaJsAnalyses(list);
      setDeltaJsExperimentId((prev) => {
        if (prev && list.some((a) => a.analysis_id === prev)) return prev;
        return list[0]?.analysis_id || '';
      });
      return list;
    } catch (err) {
      setDeltaJsAnalyses([]);
      setDeltaJsAnalysesError(err.message || 'Failed to load Delta JS analyses.');
      return [];
    }
  }, [projectId, systemId, selectedClusterId]);

  const loadDeltaJsFilterSetups = useCallback(async () => {
    if (!selectedClusterId) return [];
    setDeltaJsFilterSetupsError(null);
    try {
      const res = await fetchClusterUiSetups(projectId, systemId, selectedClusterId, {
        setupType: 'js_range_filters',
        page: 'delta_js',
      });
      const list = Array.isArray(res?.setups) ? res.setups : [];
      setDeltaJsFilterSetups(list);
      setDeltaJsFilterSetupId((prev) => {
        if (prev && list.some((s) => String(s?.setup_id) === String(prev))) return prev;
        return '';
      });
      return list;
    } catch (err) {
      setDeltaJsFilterSetups([]);
      setDeltaJsFilterSetupsError(err.message || 'Failed to load Delta JS filter setups.');
      return [];
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    analysisDataCacheRef.current = {};
    analysisDataInFlightRef.current = {};
    setSelectedAnalysisId('');
    setAnalysisData(null);
    loadAnalyses();
    loadDeltaJsAnalyses();
    loadDeltaJsFilterSetups();
  }, [selectedClusterId, loadAnalyses, loadDeltaJsAnalyses, loadDeltaJsFilterSetups]);

  const loadAnalysisData = useCallback(
    async (analysisId) => {
      if (!analysisId) return null;
      const cacheKey = `ligand_completion:${analysisId}`;
      const cached = analysisDataCacheRef.current;
      if (Object.prototype.hasOwnProperty.call(cached, cacheKey)) return cached[cacheKey];
      const inflight = analysisDataInFlightRef.current;
      if (inflight[cacheKey]) return inflight[cacheKey];

      const p = fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'ligand_completion', analysisId)
        .then((payload) => {
          analysisDataCacheRef.current = { ...analysisDataCacheRef.current, [cacheKey]: payload };
          delete analysisDataInFlightRef.current[cacheKey];
          return payload;
        })
        .catch((err) => {
          delete analysisDataInFlightRef.current[cacheKey];
          throw err;
        });
      inflight[cacheKey] = p;
      return p;
    },
    [projectId, systemId, selectedClusterId]
  );

  useEffect(() => {
    const run = async () => {
      setAnalysisDataError(null);
      setAnalysisData(null);
      if (!selectedAnalysisId) return;
      setAnalysisDataLoading(true);
      try {
        const payload = await loadAnalysisData(selectedAnalysisId);
        setAnalysisData(payload);
      } catch (err) {
        setAnalysisDataError(err.message || 'Failed to load analysis data.');
      } finally {
        setAnalysisDataLoading(false);
      }
    };
    run();
  }, [selectedAnalysisId, loadAnalysisData]);

  const handleSubmit = useCallback(async () => {
    if (!selectedClusterId || !modelAId || !modelBId || !mdSampleId) return;
    const constrained = parseCsvMixed(constrainedResidues);
    if (constraintSourceMode === 'manual' && !constrained.length) {
      setJobError('Please provide constrained residues.');
      return;
    }
    const requiresDeltaJsExperiment = constraintSourceMode === 'delta_js_auto' || successMetricMode === 'delta_js_edge';
    if (requiresDeltaJsExperiment && !deltaJsExperimentId) {
      setJobError('Please select one Delta JS experiment to use for auto-constraints/success metric.');
      return;
    }
    if (constraintSourceMode === 'delta_js_auto' && !deltaJsExperimentId) {
      setJobError('Please select a Delta JS analysis for auto-constraint mode.');
      return;
    }
    if (constraintSourceMode === 'delta_js_auto' && constraintWeightMode === 'custom') {
      setJobError('Custom constraint weights are not supported in delta_js_auto mode.');
      return;
    }
    const lambdaGrid = parseCsvFloats(lambdaValues);
    if (lambdaGrid.length < 2) {
      setJobError('Please provide at least two lambda values.');
      return;
    }
    if (successMetricMode === 'delta_js_edge' && !deltaJsExperimentId) {
      setJobError('Please select a Delta JS analysis for JS-based success metric.');
      return;
    }

    setJobError(null);
    setJob(null);
    setJobStatus(null);
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        model_a_id: modelAId,
        model_b_id: modelBId,
        md_sample_id: mdSampleId,
        constrained_residues: constraintSourceMode === 'manual' ? constrained : [],
        sampler,
        lambda_values: lambdaGrid,
        n_start_frames: Number(nStartFrames),
        n_samples_per_frame: Number(nSamplesPerFrame),
        n_steps: Number(nSteps),
        tail_steps: Number(tailSteps),
        target_window_size: Number(targetWindowSize),
        target_pseudocount: Number(targetPseudocount),
        epsilon_logpenalty: Number(epsilonLogpenalty),
        constraint_source_mode: constraintSourceMode,
        constraint_weight_mode: constraintSourceMode === 'delta_js_auto' ? 'uniform' : constraintWeightMode,
        constraint_weight_min: Number(constraintWeightMin),
        constraint_weight_max: Number(constraintWeightMax),
        gibbs_beta: Number(gibbsBeta),
        sa_beta_hot: Number(saBetaHot),
        sa_beta_cold: Number(saBetaCold),
        sa_schedule: saSchedule,
        md_label_mode: mdLabelMode,
        keep_invalid: keepInvalid,
        success_metric_mode: successMetricMode,
        delta_js_experiment_id: deltaJsExperimentId || undefined,
        delta_js_analysis_id: successMetricMode === 'delta_js_edge' ? deltaJsExperimentId : undefined,
        delta_js_d_residue_min: Number(deltaJsDResidueMin),
        delta_js_d_edge_min: Number(deltaJsDEdgeMin),
        js_success_threshold: Number(jsSuccessThreshold),
        js_success_margin: Number(jsSuccessMargin),
        deltae_margin: Number(deltaEMargin),
        completion_target_success: Number(completionTargetSuccess),
        seed: Number(seed),
      };
      if (constraintSourceMode === 'delta_js_auto') {
        payload.constraint_delta_js_analysis_id = deltaJsExperimentId;
        if (constraintDeltaJsSampleId) payload.constraint_delta_js_sample_id = constraintDeltaJsSampleId;
        payload.constraint_auto_top_k = Number(constraintAutoTopK);
        payload.constraint_auto_edge_alpha = Number(constraintAutoEdgeAlpha);
        payload.constraint_auto_exclude_success = Boolean(constraintAutoExcludeSuccess);
      }
      if (deltaJsFilterSetupId) payload.delta_js_filter_setup_id = deltaJsFilterSetupId;
      payload.delta_js_filter_edge_alpha = Number(deltaJsFilterEdgeAlpha);
      const dResMax = String(deltaJsDResidueMax || '').trim();
      const dEdgeMax = String(deltaJsDEdgeMax || '').trim();
      const alphaOpt = String(deltaJsNodeEdgeAlpha || '').trim();
      if (successMetricMode === 'delta_js_edge') {
        if (dResMax !== '') payload.delta_js_d_residue_max = Number(dResMax);
        if (dEdgeMax !== '') payload.delta_js_d_edge_max = Number(dEdgeMax);
        if (alphaOpt !== '') payload.delta_js_node_edge_alpha = Number(alphaOpt);
      }
      if (constraintSourceMode === 'manual' && constraintWeightMode === 'custom') {
        const ws = parseCsvFloats(constraintWeights);
        if (ws.length !== constrained.length) {
          setJobError('Custom constraint weights must match the constrained residue count.');
          return;
        }
        payload.constraint_weights = ws;
      }
      if (refSampleAId) payload.reference_sample_id_a = refSampleAId;
      if (refSampleBId) payload.reference_sample_id_b = refSampleBId;
      const res = await submitLigandCompletionJob(payload);
      setJob(res);
      setRunPanelOpen(false);
    } catch (err) {
      setJobError(err.message || 'Failed to submit ligand completion analysis.');
    }
  }, [
    selectedClusterId,
    modelAId,
    modelBId,
    mdSampleId,
    constrainedResidues,
    lambdaValues,
    projectId,
    systemId,
    sampler,
    nStartFrames,
    nSamplesPerFrame,
    nSteps,
    tailSteps,
    targetWindowSize,
    targetPseudocount,
    epsilonLogpenalty,
    constraintWeightMode,
    constraintWeightMin,
    constraintWeightMax,
    constraintSourceMode,
    deltaJsExperimentId,
    deltaJsFilterSetupId,
    deltaJsFilterEdgeAlpha,
    constraintDeltaJsSampleId,
    constraintAutoTopK,
    constraintAutoEdgeAlpha,
    constraintAutoExcludeSuccess,
    gibbsBeta,
    saBetaHot,
    saBetaCold,
    saSchedule,
    mdLabelMode,
    keepInvalid,
    successMetricMode,
    deltaJsDResidueMin,
    deltaJsDResidueMax,
    deltaJsDEdgeMin,
    deltaJsDEdgeMax,
    deltaJsNodeEdgeAlpha,
    jsSuccessThreshold,
    jsSuccessMargin,
    deltaEMargin,
    completionTargetSuccess,
    seed,
    constraintWeights,
    refSampleAId,
    refSampleBId,
    setRunPanelOpen,
  ]);

  useEffect(() => {
    if (!job?.job_id) return;
    let cancelled = false;
    const terminal = new Set(['finished', 'failed', 'canceled']);
    const poll = async () => {
      try {
        const status = await fetchJobStatus(job.job_id);
        if (cancelled) return;
        setJobStatus(status);
        if (terminal.has(status?.status)) {
          clearInterval(timer);
          if (status?.status === 'finished') {
            const list = await loadAnalyses();
            if (!cancelled && Array.isArray(list) && list.length) setSelectedAnalysisId(list[0].analysis_id);
            const data = await fetchSystem(projectId, systemId);
            if (!cancelled) setSystem(data);
          }
        }
      } catch (err) {
        if (!cancelled) setJobError(err.message || 'Failed to poll job.');
      }
    };
    const timer = setInterval(poll, 2500);
    poll();
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job, loadAnalyses, projectId, systemId]);

  const handleDeleteSelectedAnalysis = useCallback(async () => {
    if (!selectedClusterId || !selectedAnalysisId || deletingAnalysis) return;
    const ok = window.confirm('Delete selected Ligand Completion analysis? This cannot be undone.');
    if (!ok) return;
    setDeletingAnalysis(true);
    setJobError(null);
    try {
      await deleteClusterAnalysis(projectId, systemId, selectedClusterId, 'ligand_completion', selectedAnalysisId);
      analysisDataCacheRef.current = {};
      analysisDataInFlightRef.current = {};
      setAnalysisData(null);
      const list = await loadAnalyses();
      if (!Array.isArray(list) || !list.length) setSelectedAnalysisId('');
    } catch (err) {
      setJobError(err.message || 'Failed to delete analysis.');
    } finally {
      setDeletingAnalysis(false);
    }
  }, [selectedClusterId, selectedAnalysisId, deletingAnalysis, projectId, systemId, loadAnalyses]);

  const selectedAnalysisMeta = useMemo(
    () => analyses.find((a) => a.analysis_id === selectedAnalysisId) || null,
    [analyses, selectedAnalysisId]
  );
  const selectedAnalysisMetadataRows = useMemo(() => {
    if (!selectedAnalysisMeta) return [];
    const m = selectedAnalysisMeta;
    return [
      ['Analysis ID', m.analysis_id],
      ['Created', m.created_at],
      ['Updated', m.updated_at],
      ['Cluster', m.cluster_id],
      ['Model A', m.model_a_name || m.model_a_id],
      ['Model B', m.model_b_name || m.model_b_id],
      ['Start MD sample', m.md_sample_name || m.md_sample_id],
      ['Reference A', m.reference_sample_name_a || m.reference_sample_id_a || 'auto'],
      ['Reference B', m.reference_sample_name_b || m.reference_sample_id_b || 'auto'],
      ['Sampler', m.sampler],
      ['Frames used', m.n_start_frames_used],
      ['Frames requested', m.n_start_frames_requested],
      ['Samples/frame', m.n_samples_per_frame],
      ['Steps', m.n_steps],
      ['Tail steps', m.tail_steps],
      ['ΔE margin', m.deltae_margin],
      ['Target success', m.completion_target_success],
      ['Constraint source', m.constraint_source_mode || 'manual'],
      ['Constraint Delta JS', m.constraint_delta_js_analysis_id],
      ['Constraint Delta JS sample', m.constraint_delta_js_sample_id],
      ['Constraint auto top-K', m.constraint_auto_top_k],
      ['Constraint auto edge α', m.constraint_auto_edge_alpha],
      ['Constraint auto exclude success', m.constraint_auto_exclude_success !== undefined ? String(Boolean(m.constraint_auto_exclude_success)) : undefined],
      ['Weight mode', m.constraint_weight_mode],
      ['Constraint W min/max', `${m.constraint_weight_min ?? 'n/a'} / ${m.constraint_weight_max ?? 'n/a'}`],
      ['MD label mode', m.md_label_mode],
      ['Drop invalid', m.drop_invalid !== undefined ? String(Boolean(m.drop_invalid)) : undefined],
      ['Success metric', m.success_metric_mode || 'deltae'],
      ['JS success threshold', m.js_success_threshold],
      ['JS success margin', m.js_success_margin],
      ['Delta JS experiment', m.delta_js_experiment_id],
      ['Delta JS analysis', m.delta_js_analysis_id],
      ['Delta JS filter setup', m.delta_js_filter_setup_id],
      ['Delta JS filter edge α', m.delta_js_filter_edge_alpha],
      ['Delta-JS residue D filter', `${m.delta_js_d_residue_min ?? 'n/a'} .. ${m.delta_js_d_residue_max ?? 'max'}`],
      ['Delta-JS edge D filter', `${m.delta_js_d_edge_min ?? 'n/a'} .. ${m.delta_js_d_edge_max ?? 'max'}`],
      ['Delta-JS alpha', m.delta_js_node_edge_alpha],
      ['Delta-JS selected residues', Array.isArray(m.delta_js_selected_residue_keys) ? m.delta_js_selected_residue_keys.join(', ') : undefined],
      ['Seed', m.seed],
      ['Workers', m.workers],
      ['Constrained residues', Array.isArray(m.constrained_keys) ? m.constrained_keys.join(', ') : undefined],
      ['Lambda grid', Array.isArray(m.lambda_values) ? m.lambda_values.join(', ') : undefined],
    ].filter((row) => row[1] !== undefined && row[1] !== null && String(row[1]).trim() !== '');
  }, [selectedAnalysisMeta]);

  const parsed = useMemo(() => {
    const d = analysisData?.data || {};
    const toNumArray = (v) => (Array.isArray(v) ? v.map((x) => Number(x)) : []);
    const toNumMatrix = (v) => (Array.isArray(v) ? v.map((row) => (Array.isArray(row) ? row.map((x) => Number(x)) : [])) : []);
    return {
      lambdas: toNumArray(d.lambda_values),
      successAMean: toNumArray(d.success_a_mean),
      successBMean: toNumArray(d.success_b_mean),
      successAStd: toNumArray(d.success_a_std),
      successBStd: toNumArray(d.success_b_std),
      aucA: toNumArray(d.auc_a),
      aucB: toNumArray(d.auc_b),
      aucDir: toNumArray(d.auc_dir),
      deltaEMeanA: toNumMatrix(d.deltae_mean_under_a),
      deltaEMeanB: toNumMatrix(d.deltae_mean_under_b),
      jsAUnderA: toNumMatrix(d.js_a_under_a),
      jsBUnderB: toNumMatrix(d.js_b_under_b),
      lacsComp: toNumArray(d.lacs_component_completion),
      lacsRaw: toNumArray(d.lacs_component_raw),
      lacsNovelty: toNumArray(d.lacs_component_novelty),
    };
  }, [analysisData]);

  const deltaEMeanUnderA = useMemo(
    () => (parsed.deltaEMeanA.length ? parsed.deltaEMeanA[0].map((_, i) => mean(parsed.deltaEMeanA.map((r) => r[i]))) : []),
    [parsed.deltaEMeanA]
  );
  const deltaEMeanUnderB = useMemo(
    () => (parsed.deltaEMeanB.length ? parsed.deltaEMeanB[0].map((_, i) => mean(parsed.deltaEMeanB.map((r) => r[i]))) : []),
    [parsed.deltaEMeanB]
  );
  const jsAUnderAMean = useMemo(
    () => (parsed.jsAUnderA.length ? parsed.jsAUnderA[0].map((_, i) => mean(parsed.jsAUnderA.map((r) => r[i]))) : []),
    [parsed.jsAUnderA]
  );
  const jsBUnderBMean = useMemo(
    () => (parsed.jsBUnderB.length ? parsed.jsBUnderB[0].map((_, i) => mean(parsed.jsBUnderB.map((r) => r[i]))) : []),
    [parsed.jsBUnderB]
  );

  const lacsFrame = useMemo(() => {
    const n = Math.min(parsed.lacsComp.length, parsed.lacsRaw.length, parsed.lacsNovelty.length);
    const vals = [];
    for (let i = 0; i < n; i += 1) {
      const c = Number(parsed.lacsComp[i]);
      const r = Number(parsed.lacsRaw[i]);
      const nval = Number(parsed.lacsNovelty[i]);
      if (!Number.isFinite(c) || !Number.isFinite(r) || !Number.isFinite(nval)) continue;
      vals.push(wCompletion * c + wRaw * r - wNovelty * nval);
    }
    return vals;
  }, [parsed.lacsComp, parsed.lacsRaw, parsed.lacsNovelty, wCompletion, wRaw, wNovelty]);

  if (loadingSystem) return <Loader message="Loading ligand completion..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <HelpDrawer
        open={helpOpen}
        title="Ligand Completion: Help"
        docPath="/docs/ligand_completion_help.md"
        onClose={() => setHelpOpen(false)}
      />

      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Ligand Completion</h1>
          <p className="text-sm text-gray-400">
            Conditional completion under endpoint models A/B from selected MD starts with target-distribution constraints.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setRunPanelOpen(true)}
            className="text-xs px-3 py-2 rounded-md border border-cyan-700 text-cyan-200 hover:border-cyan-500 inline-flex items-center gap-2"
          >
            <Plus className="h-4 w-4" />
            New analysis
          </button>
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-2"
          >
            <CircleHelp className="h-4 w-4" />
            Help
          </button>
          <button
            type="button"
            onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/visualize`)}
            className="text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
          >
            Back to sampling
          </button>
        </div>
      </div>

      {runPanelOpen && (
        <div
          className="fixed inset-0 z-50 bg-black/70 p-4 sm:p-8 overflow-y-auto"
          onClick={() => setRunPanelOpen(false)}
        >
          <div
            className="mx-auto max-w-3xl rounded-lg border border-gray-700 bg-gray-900 p-4 space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-white">Run New Ligand Completion Analysis</h2>
              <button
                type="button"
                onClick={() => setRunPanelOpen(false)}
                className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-gray-700 text-gray-300 hover:border-gray-500"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Cluster</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                {clusterOptions.map((c) => (
                  <option key={c.cluster_id} value={c.cluster_id}>
                    {c.name || c.cluster_id}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-1 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Model A</label>
                <select
                  value={modelAId}
                  onChange={(e) => setModelAId(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  {modelOptions.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.name || m.model_id}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Model B</label>
                <select
                  value={modelBId}
                  onChange={(e) => setModelBId(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  {modelOptions.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.name || m.model_id}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Start MD sample</label>
              <select
                value={mdSampleId}
                onChange={(e) => setMdSampleId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                {mdSamples.map((s) => (
                  <option key={s.sample_id} value={s.sample_id}>
                    {s.name || s.sample_id}
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Ref A (optional)</label>
                <select
                  value={refSampleAId}
                  onChange={(e) => setRefSampleAId(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  <option value="">Auto</option>
                  {mdSamples.map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {s.name || s.sample_id}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Ref B (optional)</label>
                <select
                  value={refSampleBId}
                  onChange={(e) => setRefSampleBId(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  <option value="">Auto</option>
                  {mdSamples.map((s) => (
                    <option key={s.sample_id} value={s.sample_id}>
                      {s.name || s.sample_id}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Constraint source mode</label>
              <select
                value={constraintSourceMode}
                onChange={(e) => setConstraintSourceMode(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                <option value="manual">manual</option>
                <option value="delta_js_auto">delta_js_auto</option>
              </select>
            </div>
            {constraintSourceMode === 'manual' ? (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Constrained residues</label>
                <input
                  value={constrainedResidues}
                  onChange={(e) => setConstrainedResidues(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                  placeholder="e.g. res_279,res_281 or 279,281"
                />
              </div>
            ) : (
              <div className="space-y-2 rounded-md border border-gray-800 bg-gray-950/40 p-2">
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Sample for impact</label>
                    <select
                      value={constraintDeltaJsSampleId}
                      onChange={(e) => setConstraintDeltaJsSampleId(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                    >
                      {mdSamples.map((s) => (
                        <option key={s.sample_id} value={s.sample_id}>
                          {s.name || s.sample_id}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Top-K residues</label>
                    <input type="number" min="1" value={constraintAutoTopK} onChange={(e) => setConstraintAutoTopK(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Edge α</label>
                    <input type="number" min="0" max="1" step="0.05" value={constraintAutoEdgeAlpha} onChange={(e) => setConstraintAutoEdgeAlpha(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
                  </div>
                </div>
                <label className="inline-flex items-center gap-2 text-xs text-gray-300">
                  <input
                    type="checkbox"
                    checked={constraintAutoExcludeSuccess}
                    onChange={(e) => setConstraintAutoExcludeSuccess(e.target.checked)}
                  />
                  Exclude residues used by delta_js_edge success set
                </label>
              </div>
            )}
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Sampler</label>
                <select
                  value={sampler}
                  onChange={(e) => setSampler(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                >
                  <option value="sa">SA (default)</option>
                  <option value="gibbs">Gibbs</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Lambda grid</label>
                <input
                  value={lambdaValues}
                  onChange={(e) => setLambdaValues(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
                />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Frames</label>
                <input type="number" value={nStartFrames} onChange={(e) => setNStartFrames(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Samples/frame</label>
                <input type="number" value={nSamplesPerFrame} onChange={(e) => setNSamplesPerFrame(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Steps</label>
                <input type="number" value={nSteps} onChange={(e) => setNSteps(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Tail steps</label>
                <input type="number" value={tailSteps} onChange={(e) => setTailSteps(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Window</label>
                <input type="number" value={targetWindowSize} onChange={(e) => setTargetWindowSize(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Target p* </label>
                <input type="number" step="0.05" value={completionTargetSuccess} onChange={(e) => setCompletionTargetSuccess(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Pseudo-count</label>
                <input type="number" step="0.0001" value={targetPseudocount} onChange={(e) => setTargetPseudocount(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Log epsilon</label>
                <input type="number" step="0.0000001" value={epsilonLogpenalty} onChange={(e) => setEpsilonLogpenalty(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Weight mode</label>
                <select value={constraintWeightMode} onChange={(e) => setConstraintWeightMode(e.target.value)} disabled={constraintSourceMode !== 'manual'} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white disabled:opacity-40">
                  <option value="uniform">uniform</option>
                  <option value="js_abs">js_abs</option>
                  <option value="custom">custom</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Custom weights</label>
                <input value={constraintWeights} onChange={(e) => setConstraintWeights(e.target.value)} disabled={constraintWeightMode !== 'custom' || constraintSourceMode !== 'manual'} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white disabled:opacity-40" placeholder="w1,w2,..." />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">W min</label>
                <input type="number" step="0.01" value={constraintWeightMin} onChange={(e) => setConstraintWeightMin(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">W max</label>
                <input type="number" step="0.01" value={constraintWeightMax} onChange={(e) => setConstraintWeightMax(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Success metric</label>
                <select value={successMetricMode} onChange={(e) => setSuccessMetricMode(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white">
                  <option value="deltae">deltae</option>
                  <option value="delta_js_edge">delta_js_edge</option>
                </select>
              </div>
            </div>
            {(constraintSourceMode === 'delta_js_auto' || successMetricMode === 'delta_js_edge') && (
              <div className="space-y-2 rounded-md border border-gray-800 bg-gray-950/40 p-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Delta JS experiment (shared)</label>
                  <select
                    value={deltaJsExperimentId}
                    onChange={(e) => setDeltaJsExperimentId(e.target.value)}
                    className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                  >
                    <option value="">Select analysis</option>
                    {deltaJsAnalyses.map((a) => (
                      <option key={a.analysis_id} value={a.analysis_id}>
                        {(a.created_at || a.analysis_id)} :: {(a.model_a_name || a.model_a_id || 'A')} vs {(a.model_b_name || a.model_b_id || 'B')}
                      </option>
                    ))}
                  </select>
                  {deltaJsAnalysesError && <p className="text-[11px] text-red-300 mt-1">{deltaJsAnalysesError}</p>}
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Filter setup (optional)</label>
                    <select
                      value={deltaJsFilterSetupId}
                      onChange={(e) => setDeltaJsFilterSetupId(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                    >
                      <option value="">None</option>
                      {deltaJsFilterSetups.map((s) => (
                        <option key={String(s.setup_id)} value={String(s.setup_id)}>
                          {s.name || s.setup_id}
                        </option>
                      ))}
                    </select>
                    {deltaJsFilterSetupsError && <p className="text-[11px] text-red-300 mt-1">{deltaJsFilterSetupsError}</p>}
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Filter edge α</label>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.05"
                      value={deltaJsFilterEdgeAlpha}
                      onChange={(e) => setDeltaJsFilterEdgeAlpha(e.target.value)}
                      className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white"
                    />
                  </div>
                </div>
              </div>
            )}
            {successMetricMode === 'delta_js_edge' && (
              <>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Residue D min</label>
                    <input type="number" step="0.001" value={deltaJsDResidueMin} onChange={(e) => setDeltaJsDResidueMin(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Residue D max (opt)</label>
                    <input type="number" step="0.001" value={deltaJsDResidueMax} onChange={(e) => setDeltaJsDResidueMax(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Edge D min</label>
                    <input type="number" step="0.001" value={deltaJsDEdgeMin} onChange={(e) => setDeltaJsDEdgeMin(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Edge D max (opt)</label>
                    <input type="number" step="0.001" value={deltaJsDEdgeMax} onChange={(e) => setDeltaJsDEdgeMax(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">JS threshold</label>
                    <input type="number" step="0.001" value={jsSuccessThreshold} onChange={(e) => setJsSuccessThreshold(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">JS margin</label>
                    <input type="number" step="0.001" value={jsSuccessMargin} onChange={(e) => setJsSuccessMargin(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
                  </div>
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">Node-edge α (opt)</label>
                    <input type="number" step="0.05" value={deltaJsNodeEdgeAlpha} onChange={(e) => setDeltaJsNodeEdgeAlpha(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
                  </div>
                </div>
              </>
            )}
            <div>
              <label className="block text-xs text-gray-400 mb-1">ΔE margin</label>
              <input type="number" step="0.01" value={deltaEMargin} onChange={(e) => setDeltaEMargin(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              <p className="text-[11px] text-gray-500 mt-1">Used only when success metric = deltae (kept for diagnostics in all modes).</p>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Gibbs beta</label>
                <input type="number" step="0.05" value={gibbsBeta} onChange={(e) => setGibbsBeta(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">SA schedule</label>
                <select value={saSchedule} onChange={(e) => setSaSchedule(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white">
                  <option value="geom">geom</option>
                  <option value="lin">lin</option>
                </select>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">SA hot</label>
                <input type="number" step="0.1" value={saBetaHot} onChange={(e) => setSaBetaHot(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">SA cold</label>
                <input type="number" step="0.5" value={saBetaCold} onChange={(e) => setSaBetaCold(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Seed</label>
                <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-2 text-white" />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <label className="inline-flex items-center gap-2 text-xs text-gray-300">
                <input type="checkbox" checked={keepInvalid} onChange={(e) => setKeepInvalid(e.target.checked)} />
                Keep invalid
              </label>
              <div className="flex items-center gap-2">
                <label className="text-xs text-gray-400">Label mode</label>
                <select value={mdLabelMode} onChange={(e) => setMdLabelMode(e.target.value)} className="bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-white text-xs">
                  <option value="assigned">assigned</option>
                  <option value="halo">halo</option>
                </select>
              </div>
            </div>

            {jobError && <ErrorMessage message={jobError} />}
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!selectedClusterId || !modelAId || !modelBId || !mdSampleId}
              className="w-full px-3 py-2 rounded-md border border-cyan-600 text-cyan-200 hover:border-cyan-400 disabled:opacity-40 inline-flex items-center justify-center gap-2"
            >
              <Play className="h-4 w-4" />
              Run analysis
            </button>
            {job && (
              <div className="text-xs text-gray-300">
                <div>job: {job.job_id}</div>
                <div>status: {jobStatus?.status || 'queued'}</div>
                <div>{jobStatus?.meta?.status || ''}</div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[390px_1fr] gap-4">
        <aside className="space-y-3">

          <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-white">Saved analyses</h2>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleDeleteSelectedAnalysis}
                  disabled={!selectedAnalysisId || deletingAnalysis}
                  className="text-xs px-2 py-1 rounded border border-red-800 text-red-200 hover:border-red-600 disabled:opacity-40"
                >
                  {deletingAnalysis ? 'Deleting...' : 'Delete'}
                </button>
                <button type="button" onClick={loadAnalyses} className="text-xs px-2 py-1 rounded border border-gray-700 text-gray-200 hover:border-gray-500 inline-flex items-center gap-1">
                  <RefreshCw className="h-3 w-3" />
                  Refresh
                </button>
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Cluster</label>
              <select
                value={selectedClusterId}
                onChange={(e) => setSelectedClusterId(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              >
                {clusterOptions.map((c) => (
                  <option key={c.cluster_id} value={c.cluster_id}>
                    {c.name || c.cluster_id}
                  </option>
                ))}
              </select>
            </div>
            {job && (
              <div className="text-xs text-gray-300 rounded-md border border-gray-800 bg-gray-950/60 p-2">
                <div>Latest job: {job.job_id}</div>
                <div>Status: {jobStatus?.status || 'queued'}</div>
                {jobStatus?.meta?.status && <div>{jobStatus.meta.status}</div>}
              </div>
            )}
            {jobError && <ErrorMessage message={jobError} />}
            {analysesError && <ErrorMessage message={analysesError} />}
            {analysesLoading && <p className="text-xs text-gray-400">Loading...</p>}
            <select
              value={selectedAnalysisId}
              onChange={(e) => setSelectedAnalysisId(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
            >
              {analyses.map((a) => (
                <option key={a.analysis_id} value={a.analysis_id}>
                  {a.created_at || a.analysis_id}
                </option>
              ))}
            </select>
            {selectedAnalysisMeta && (
              <div className="text-xs text-gray-400 space-y-1">
                <div>models: {selectedAnalysisMeta.model_a_name || selectedAnalysisMeta.model_a_id} vs {selectedAnalysisMeta.model_b_name || selectedAnalysisMeta.model_b_id}</div>
                <div>frames: {selectedAnalysisMeta.n_start_frames_used || selectedAnalysisMeta.summary?.n_start_frames_used || selectedAnalysisMeta.summary?.n_frames || 'n/a'}</div>
              </div>
            )}
          </div>
        </aside>

        <section className="space-y-3">
          {analysisDataLoading && <Loader message="Loading analysis..." />}
          {analysisDataError && <ErrorMessage message={analysisDataError} />}
          {!analysisDataLoading && !analysisDataError && !selectedAnalysisId && (
            <p className="text-sm text-gray-400">No analysis selected.</p>
          )}
          {!analysisDataLoading && !analysisDataError && selectedAnalysisId && (
            <>
              <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
                <h3 className="text-sm font-semibold text-white mb-2">Selected Analysis Metadata</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1 text-xs">
                  {selectedAnalysisMetadataRows.map(([label, value]) => (
                    <div key={String(label)} className="flex items-start justify-between gap-3 border-b border-gray-800/70 py-1">
                      <span className="text-gray-400">{label}</span>
                      <span className="text-gray-200 text-right break-all">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
                <h3 className="text-sm font-semibold text-white mb-2">Success vs λ (MD→A vs MD→B)</h3>
                <Plot
                  data={[
                    {
                      x: parsed.lambdas,
                      y: parsed.successAMean,
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Success under A',
                      line: { color: '#ef4444', width: 2 },
                    },
                    {
                      x: parsed.lambdas,
                      y: parsed.successBMean,
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Success under B',
                      line: { color: '#3b82f6', width: 2 },
                    },
                    {
                      x: parsed.lambdas,
                      y: parsed.successAMean.map((v, i) => Number(v) + Number(parsed.successAStd[i] || 0)),
                      type: 'scatter',
                      mode: 'lines',
                      line: { width: 0 },
                      showlegend: false,
                      hoverinfo: 'skip',
                    },
                    {
                      x: parsed.lambdas,
                      y: parsed.successAMean.map((v, i) => Number(v) - Number(parsed.successAStd[i] || 0)),
                      type: 'scatter',
                      mode: 'lines',
                      fill: 'tonexty',
                      fillcolor: 'rgba(239,68,68,0.15)',
                      line: { width: 0 },
                      showlegend: false,
                      hoverinfo: 'skip',
                    },
                    {
                      x: parsed.lambdas,
                      y: parsed.successBMean.map((v, i) => Number(v) + Number(parsed.successBStd[i] || 0)),
                      type: 'scatter',
                      mode: 'lines',
                      line: { width: 0 },
                      showlegend: false,
                      hoverinfo: 'skip',
                    },
                    {
                      x: parsed.lambdas,
                      y: parsed.successBMean.map((v, i) => Number(v) - Number(parsed.successBStd[i] || 0)),
                      type: 'scatter',
                      mode: 'lines',
                      fill: 'tonexty',
                      fillcolor: 'rgba(59,130,246,0.15)',
                      line: { width: 0 },
                      showlegend: false,
                      hoverinfo: 'skip',
                    },
                  ]}
                  layout={{
                    autosize: true,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    margin: { l: 46, r: 10, t: 8, b: 42 },
                    font: { color: '#d1d5db', size: 12 },
                    xaxis: { title: 'λ' },
                    yaxis: { title: 'Success rate', range: [0, 1] },
                    legend: { orientation: 'h', x: 0, y: 1.15 },
                  }}
                  style={{ width: '100%', height: 350 }}
                  config={{ displaylogo: false, responsive: true }}
                />
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
                  <h3 className="text-sm font-semibold text-white mb-2">AUC distributions</h3>
                  <Plot
                    data={[
                      { y: parsed.aucA, type: 'box', name: 'AUC_A', marker: { color: '#ef4444' } },
                      { y: parsed.aucB, type: 'box', name: 'AUC_B', marker: { color: '#3b82f6' } },
                      { y: parsed.aucDir, type: 'box', name: 'AUC_B - AUC_A', marker: { color: '#22c55e' } },
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(0,0,0,0)',
                      margin: { l: 46, r: 10, t: 8, b: 42 },
                      font: { color: '#d1d5db', size: 12 },
                      yaxis: { title: 'AUC' },
                    }}
                    style={{ width: '100%', height: 330 }}
                    config={{ displaylogo: false, responsive: true }}
                  />
                </div>

                <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 space-y-2">
                  <h3 className="text-sm font-semibold text-white">LACS recombination</h3>
                  <div className="grid grid-cols-3 gap-2 text-xs text-gray-300">
                    <label className="space-y-1">
                      <span>w completion</span>
                      <input type="number" step="0.1" value={wCompletion} onChange={(e) => setWCompletion(Number(e.target.value))} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-white" />
                    </label>
                    <label className="space-y-1">
                      <span>w raw</span>
                      <input type="number" step="0.1" value={wRaw} onChange={(e) => setWRaw(Number(e.target.value))} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-white" />
                    </label>
                    <label className="space-y-1">
                      <span>w novelty</span>
                      <input type="number" step="0.1" value={wNovelty} onChange={(e) => setWNovelty(Number(e.target.value))} className="w-full bg-gray-900 border border-gray-700 rounded-md px-2 py-1 text-white" />
                    </label>
                  </div>
                  <div className="text-xs text-gray-300">
                    <div>LACS mean: {mean(lacsFrame)?.toFixed(4) ?? 'n/a'}</div>
                    <div>LACS median: {median(lacsFrame)?.toFixed(4) ?? 'n/a'}</div>
                  </div>
                  <Plot
                    data={[{ x: lacsFrame, type: 'histogram', marker: { color: '#22c55e' }, nbinsx: 30 }]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(0,0,0,0)',
                      margin: { l: 46, r: 10, t: 8, b: 42 },
                      font: { color: '#d1d5db', size: 12 },
                      xaxis: { title: 'LACS' },
                      yaxis: { title: 'Count' },
                    }}
                    style={{ width: '100%', height: 240 }}
                    config={{ displaylogo: false, responsive: true }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
                  <h3 className="text-sm font-semibold text-white mb-2">Mean ΔE_BA on sampled tails</h3>
                  <Plot
                    data={[
                      { x: parsed.lambdas, y: deltaEMeanUnderA, type: 'scatter', mode: 'lines+markers', name: 'under A', line: { color: '#ef4444' } },
                      { x: parsed.lambdas, y: deltaEMeanUnderB, type: 'scatter', mode: 'lines+markers', name: 'under B', line: { color: '#3b82f6' } },
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(0,0,0,0)',
                      margin: { l: 46, r: 10, t: 8, b: 42 },
                      font: { color: '#d1d5db', size: 12 },
                      xaxis: { title: 'λ' },
                      yaxis: { title: 'mean(E_B - E_A)' },
                    }}
                    style={{ width: '100%', height: 320 }}
                    config={{ displaylogo: false, responsive: true }}
                  />
                </div>
                <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-3">
                  <h3 className="text-sm font-semibold text-white mb-2">Arrival JS (lower is closer)</h3>
                  <Plot
                    data={[
                      { x: parsed.lambdas, y: jsAUnderAMean, type: 'scatter', mode: 'lines+markers', name: 'JS to A (under A)', line: { color: '#ef4444' } },
                      { x: parsed.lambdas, y: jsBUnderBMean, type: 'scatter', mode: 'lines+markers', name: 'JS to B (under B)', line: { color: '#3b82f6' } },
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: 'rgba(0,0,0,0)',
                      plot_bgcolor: 'rgba(0,0,0,0)',
                      margin: { l: 46, r: 10, t: 8, b: 42 },
                      font: { color: '#d1d5db', size: 12 },
                      xaxis: { title: 'λ' },
                      yaxis: { title: 'mean node JS' },
                    }}
                    style={{ width: '100%', height: 320 }}
                    config={{ displaylogo: false, responsive: true }}
                  />
                </div>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
