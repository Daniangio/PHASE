import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { CircleHelp, RefreshCw } from 'lucide-react';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitSpectralIntersectionJob } from '../api/jobs';

const INTERSECTION_TYPE = 'hamiltonian_spectral_intersection';
const SINGLE_TYPE = 'hamiltonian_spectral_single';
const PAIR_TYPE = 'hamiltonian_spectral_pair';
const CLASS_NAMES = ['Other', 'Structural scaffold', 'Transient switches', 'Allosteric piston'];

function safeArray(x) { return Array.isArray(x) ? x : []; }

function analysisTitle(meta) {
  if (!meta) return '';
  if (meta.mode === 'single') return `${meta.state_name || meta.state_id}`;
  if (meta.mode === 'pair') return `${meta.state_a_name || meta.state_a_id} -> ${meta.state_b_name || meta.state_b_id}`;
  if (meta.mode === 'intersection') return `${meta.single_state_name || meta.single_analysis_id} x ${meta.pair_state_a_name || meta.pair_state_a_id} -> ${meta.pair_state_b_name || meta.pair_state_b_id}`;
  return meta.analysis_id || '';
}

function parsePistonMembers(raw) {
  const first = Array.isArray(raw) ? raw[0] : raw;
  if (!first) return [];
  try {
    const parsed = JSON.parse(String(first));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function buildCompositeHeatmap(data) {
  const sIds = safeArray(data.composite_structural_community_ids).map(Number);
  const fIds = safeArray(data.composite_functional_community_ids).map(Number);
  const sizes = safeArray(data.composite_group_sizes).map(Number);
  const sx = Array.from(new Set(sIds)).sort((a, b) => a - b);
  const fy = Array.from(new Set(fIds)).sort((a, b) => a - b);
  const sIndex = new Map(sx.map((x, i) => [x, i]));
  const fIndex = new Map(fy.map((x, i) => [x, i]));
  const z = fy.map(() => sx.map(() => 0));
  for (let i = 0; i < sizes.length; i += 1) {
    const row = fIndex.get(fIds[i]);
    const col = sIndex.get(sIds[i]);
    if (row !== undefined && col !== undefined) z[row][col] = sizes[i];
  }
  return { x: sx.map((x) => `S${x}`), y: fy.map((x) => `F${x}`), z };
}

export default function SpectralIntersectionPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [error, setError] = useState(null);
  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [singleAnalyses, setSingleAnalyses] = useState([]);
  const [pairAnalyses, setPairAnalyses] = useState([]);
  const [intersections, setIntersections] = useState([]);
  const [selectedSingleId, setSelectedSingleId] = useState('');
  const [selectedPairId, setSelectedPairId] = useState('');
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [loadingData, setLoadingData] = useState(false);
  const [runOpen, setRunOpen] = useState(false);
  const [minGroupSize, setMinGroupSize] = useState(3);
  const [overwrite, setOverwrite] = useState(false);
  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoadingSystem(true);
      try {
        const payload = await fetchSystem(projectId, systemId);
        if (!cancelled) setSystem(payload);
      } catch (err) {
        if (!cancelled) setError(err.message || 'Failed to load system.');
      } finally {
        if (!cancelled) setLoadingSystem(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [projectId, systemId]);

  const clusters = useMemo(() => safeArray(system?.metastable_clusters).filter((c) => c.cluster_id), [system]);

  useEffect(() => {
    if (!clusters.length) return;
    const qs = new URLSearchParams(location.search || '');
    const requested = String(qs.get('cluster_id') || '').trim();
    if (requested && clusters.some((c) => c.cluster_id === requested)) setSelectedClusterId(requested);
    else if (!selectedClusterId || !clusters.some((c) => c.cluster_id === selectedClusterId)) setSelectedClusterId(clusters[0].cluster_id);
  }, [clusters, selectedClusterId, location.search]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setError(null);
    try {
      const [singlePayload, pairPayload, intersectionPayload] = await Promise.all([
        fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: SINGLE_TYPE }),
        fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: PAIR_TYPE }),
        fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: INTERSECTION_TYPE }),
      ]);
      const singles = safeArray(singlePayload?.analyses);
      const pairs = safeArray(pairPayload?.analyses);
      const rows = safeArray(intersectionPayload?.analyses);
      setSingleAnalyses(singles);
      setPairAnalyses(pairs);
      setIntersections(rows);
      setSelectedSingleId((prev) => (prev && singles.some((a) => String(a.analysis_id) === prev) ? prev : String(singles[0]?.analysis_id || '')));
      setSelectedPairId((prev) => (prev && pairs.some((a) => String(a.analysis_id) === prev) ? prev : String(pairs[0]?.analysis_id || '')));
      setSelectedAnalysisId((prev) => (prev && rows.some((a) => String(a.analysis_id) === prev) ? prev : String(rows[0]?.analysis_id || '')));
    } catch (err) {
      setError(err.message || 'Failed to load spectral analyses.');
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => { loadAnalyses(); }, [loadAnalyses]);

  useEffect(() => {
    if (!selectedClusterId || !selectedAnalysisId) { setAnalysisData(null); return; }
    let cancelled = false;
    const load = async () => {
      setLoadingData(true);
      try {
        const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, INTERSECTION_TYPE, selectedAnalysisId);
        if (!cancelled) setAnalysisData(payload);
      } catch (err) {
        if (!cancelled) setError(err.message || 'Failed to load intersection data.');
      } finally {
        if (!cancelled) setLoadingData(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [projectId, systemId, selectedClusterId, selectedAnalysisId]);

  useEffect(() => {
    if (!job?.job_id) return undefined;
    let cancelled = false;
    const poll = async () => {
      try {
        const status = await fetchJobStatus(job.job_id);
        if (cancelled) return;
        setJobStatus(status);
        const done = ['finished', 'failed', 'canceled', 'cancelled'].includes(String(status?.status || status?.meta?.status || '').toLowerCase());
        if (done) loadAnalyses();
      } catch (err) {
        if (!cancelled) setError(err.message || 'Failed to poll job.');
      }
    };
    poll();
    const timer = setInterval(poll, 1500);
    return () => { cancelled = true; clearInterval(timer); };
  }, [job, loadAnalyses]);

  const submitRun = useCallback(async () => {
    setError(null);
    try {
      const res = await submitSpectralIntersectionJob({
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        single_analysis_id: selectedSingleId,
        pair_analysis_id: selectedPairId,
        min_group_size: Math.max(1, Number(minGroupSize) || 3),
        overwrite,
      });
      setJob(res);
      setJobStatus(null);
      setRunOpen(false);
    } catch (err) {
      setError(err.message || 'Failed to submit spectral intersection analysis.');
    }
  }, [projectId, systemId, selectedClusterId, selectedSingleId, selectedPairId, minGroupSize, overwrite]);

  const data = useMemo(() => analysisData?.data || {}, [analysisData]);
  const meta = analysisData?.metadata || null;
  const pistonMembers = useMemo(() => parsePistonMembers(data.piston_members_json), [data.piston_members_json]);
  const classCounts = useMemo(() => safeArray(data.class_counts).map((row) => safeArray(row).map(Number)), [data.class_counts]);
  const heatmap = useMemo(() => buildCompositeHeatmap(data), [data]);

  const classPlot = useMemo(() => ({
    data: [{
      x: classCounts.map((row) => CLASS_NAMES[row[0]] || `class ${row[0]}`),
      y: classCounts.map((row) => row[1]),
      type: 'bar',
      marker: { color: classCounts.map((row) => ['#6b7280', '#22c55e', '#f59e0b', '#ef4444'][row[0]] || '#9ca3af') },
    }],
    layout: { title: 'Residue roles', paper_bgcolor: '#111827', plot_bgcolor: '#111827', font: { color: '#e5e7eb' }, height: 280, margin: { l: 55, r: 20, t: 45, b: 70 }, yaxis: { title: 'residues' } },
    config: { responsive: true, displayModeBar: false },
  }), [classCounts]);

  const heatmapPlot = useMemo(() => ({
    data: [{ z: heatmap.z, x: heatmap.x, y: heatmap.y, type: 'heatmap', colorscale: 'YlOrRd', colorbar: { title: 'residues' }, hovertemplate: 'struct=%{x}<br>func=%{y}<br>n=%{z}<extra></extra>' }],
    layout: { title: 'Composite groups Cstruct x Cfunc', paper_bgcolor: '#111827', plot_bgcolor: '#111827', font: { color: '#e5e7eb' }, height: 360, margin: { l: 70, r: 20, t: 45, b: 70 }, xaxis: { title: 'structural community' }, yaxis: { title: 'functional community' } },
    config: { responsive: true, displayModeBar: true },
  }), [heatmap]);

  if (loadingSystem) return <Loader message="Loading spectral intersections..." />;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-6 overflow-y-auto">
      <HelpDrawer open={helpOpen} onClose={() => setHelpOpen(false)} title="Spectral intersection help" docPath="/docs/spectral_intersection.md" />
      {runOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="w-full max-w-2xl rounded-lg border border-gray-700 bg-gray-900 p-4 shadow-xl">
            <div className="flex items-start justify-between border-b border-gray-800 pb-3">
              <div><h2 className="text-lg font-semibold text-white">Run spectral set-intersection</h2><p className="mt-1 text-xs text-gray-400">Intersect one structural single-state community analysis with one functional pair community analysis.</p></div>
              <button type="button" onClick={() => setRunOpen(false)} className="text-sm text-gray-400 hover:text-gray-100">Close</button>
            </div>
            <div className="space-y-3 pt-4">
              <label className="block text-xs text-gray-400">Structural single-state analysis<select value={selectedSingleId} onChange={(e) => setSelectedSingleId(e.target.value)} className="mt-1 w-full rounded border border-gray-800 bg-gray-950 px-2 py-2 text-sm text-gray-100">{singleAnalyses.map((a) => <option key={a.analysis_id} value={a.analysis_id}>{analysisTitle(a)}</option>)}</select></label>
              <label className="block text-xs text-gray-400">Functional pair analysis<select value={selectedPairId} onChange={(e) => setSelectedPairId(e.target.value)} className="mt-1 w-full rounded border border-gray-800 bg-gray-950 px-2 py-2 text-sm text-gray-100">{pairAnalyses.map((a) => <option key={a.analysis_id} value={a.analysis_id}>{analysisTitle(a)}</option>)}</select></label>
              <label className="block text-xs text-gray-400">Minimum piston group size<input type="number" min={1} step={1} value={minGroupSize} onChange={(e) => setMinGroupSize(Number(e.target.value))} className="mt-1 w-full rounded border border-gray-800 bg-gray-950 px-2 py-2 text-sm text-gray-100" /></label>
              <label className="flex items-center gap-2 text-sm text-gray-200"><input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} />Overwrite existing matching analysis</label>
              <button type="button" onClick={submitRun} disabled={!selectedSingleId || !selectedPairId} className="w-full rounded-md bg-cyan-500 px-3 py-2 text-sm font-semibold text-black hover:bg-cyan-400 disabled:opacity-50">Run intersection</button>
            </div>
          </div>
        </div>
      ) : null}
      <div className="mx-auto max-w-7xl space-y-4 pb-16">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-cyan-400">PHASE pistons</div>
            <h1 className="mt-2 text-2xl font-semibold text-white">Spectral set-intersection</h1>
            <p className="mt-1 max-w-3xl text-sm text-gray-400">Intersects structural communities from a single-state Laplacian with functional communities from a pair ΔF Laplacian to identify allosteric pistons.</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => setHelpOpen(true)} className="inline-flex items-center gap-2 rounded-md border border-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-800"><CircleHelp className="h-4 w-4" /> Help</button>
            <button type="button" onClick={loadAnalyses} className="inline-flex items-center gap-2 rounded-md border border-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-800"><RefreshCw className="h-4 w-4" /> Refresh</button>
            <button type="button" onClick={() => navigate(`/projects/${projectId}/systems/${systemId}/sampling/spectral_intersection_3d${selectedClusterId ? `?cluster_id=${encodeURIComponent(selectedClusterId)}` : ''}`)} className="rounded-md border border-gray-700 px-3 py-2 text-sm text-gray-200 hover:bg-gray-800">3D view</button>
            <button type="button" onClick={() => setRunOpen(true)} className="rounded-md bg-cyan-500 px-3 py-2 text-sm font-semibold text-black hover:bg-cyan-400">Run analysis</button>
          </div>
        </div>
        {error ? <ErrorMessage message={error} /> : null}
        {job ? <div className="rounded-lg border border-cyan-800 bg-cyan-950/30 p-3 text-sm text-cyan-100">Job {job.job_id}: {jobStatus?.meta?.status || jobStatus?.status || 'queued'} · progress {Math.round(Number(jobStatus?.meta?.progress || jobStatus?.progress || 0))}%</div> : null}
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[330px_minmax(0,1fr)]">
          <aside className="space-y-3 rounded-lg border border-gray-800 bg-gray-900 p-3">
            <label className="block text-xs text-gray-400">Cluster<select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="mt-1 w-full rounded border border-gray-800 bg-gray-950 px-2 py-2 text-sm text-gray-100">{clusters.map((c) => <option key={c.cluster_id} value={c.cluster_id}>{c.name || c.cluster_id}</option>)}</select></label>
            <div className="text-xs uppercase tracking-[0.15em] text-gray-500">Available intersections</div>
            <div className="space-y-2">
              {intersections.map((a) => <button key={a.analysis_id} type="button" onClick={() => setSelectedAnalysisId(String(a.analysis_id))} className={`w-full rounded-md border px-3 py-2 text-left ${String(selectedAnalysisId) === String(a.analysis_id) ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-700 bg-gray-950/50 hover:bg-gray-800'}`}><div className="text-sm text-gray-100">{analysisTitle(a)}</div><div className="mt-1 text-xs text-gray-400">pistons {a?.summary?.n_pistons ?? 'n/a'} · min {a?.min_group_size ?? 'n/a'}</div></button>)}
              {!intersections.length ? <p className="text-xs text-gray-500">No intersections yet. Run one after creating v3 spectral communities.</p> : null}
            </div>
          </aside>
          <section className="min-w-0 space-y-4">
            {loadingData ? <Loader message="Loading intersection analysis..." /> : null}
            {!loadingData && !analysisData ? <div className="rounded-lg border border-gray-800 bg-gray-900 p-6 text-sm text-gray-300">Select an existing intersection or run a new one.</div> : null}
            {!loadingData && analysisData ? (
              <>
                <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
                  <div className="text-xs uppercase tracking-[0.15em] text-gray-500">Selected analysis</div>
                  <h2 className="mt-2 text-lg font-semibold text-white">{analysisTitle(meta)}</h2>
                  <p className="mt-1 text-sm text-gray-400">Allosteric pistons are composite groups `(Cstruct, Cfunc)` with at least {meta?.min_group_size ?? 3} residues. Scaffold and switch labels are heuristic role tags for non-piston residues.</p>
                </div>
                <div className="grid grid-cols-1 gap-4 2xl:grid-cols-2">
                  <div className="overflow-hidden rounded-lg border border-gray-800 bg-gray-900 p-3"><Plot data={classPlot.data} layout={classPlot.layout} config={classPlot.config} style={{ width: '100%' }} /></div>
                  <div className="overflow-hidden rounded-lg border border-gray-800 bg-gray-900 p-3"><Plot data={heatmapPlot.data} layout={heatmapPlot.layout} config={heatmapPlot.config} style={{ width: '100%' }} /></div>
                </div>
                <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 overflow-x-auto">
                  <h3 className="mb-2 text-sm font-semibold text-gray-100">Allosteric pistons</h3>
                  <table className="min-w-full text-sm">
                    <thead className="text-xs uppercase tracking-[0.12em] text-gray-500"><tr><th className="px-2 py-2 text-left">Piston</th><th className="px-2 py-2 text-left">Cstruct</th><th className="px-2 py-2 text-left">Cfunc</th><th className="px-2 py-2 text-right">Size</th><th className="px-2 py-2 text-left">Residues</th></tr></thead>
                    <tbody>
                      {pistonMembers.map((p) => <tr key={p.piston_id} className="border-t border-gray-800"><td className="px-2 py-2 text-gray-200">P{p.piston_id}</td><td className="px-2 py-2 text-gray-300">{p.structural_community_id}</td><td className="px-2 py-2 text-gray-300">{p.functional_community_id}</td><td className="px-2 py-2 text-right text-gray-300">{p.size}</td><td className="px-2 py-2 text-gray-300">{safeArray(p.residue_keys).slice(0, 24).join(', ')}{safeArray(p.residue_keys).length > 24 ? ' ...' : ''}</td></tr>)}
                      {!pistonMembers.length ? <tr><td colSpan={5} className="px-2 py-4 text-sm text-gray-500">No piston groups passed the minimum size threshold.</td></tr> : null}
                    </tbody>
                  </table>
                </div>
              </>
            ) : null}
          </section>
        </div>
      </div>
    </div>
  );
}
