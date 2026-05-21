import { useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { CircleHelp, RefreshCw } from 'lucide-react';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import HelpDrawer from '../components/common/HelpDrawer';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';
import { fetchJobStatus, submitTransientStatesJob } from '../api/jobs';

function fmtPct(x) {
  const v = Number(x);
  return Number.isFinite(v) ? `${(100 * v).toFixed(2)}%` : 'n/a';
}

function fmt(x, digits = 3) {
  const v = Number(x);
  return Number.isFinite(v) ? v.toFixed(digits) : 'n/a';
}

function safeArray(x) {
  return Array.isArray(x) ? x : [];
}

function analysisLabel(meta) {
  const ts = String(meta?.updated_at || meta?.created_at || '').slice(0, 19) || 'n/a';
  const s = meta?.summary || {};
  return `${ts} · samples=${s.n_samples ?? 0} · node hits=${s.n_node_hits ?? 0} · edge hits=${s.n_edge_hits ?? 0}`;
}

export default function TransientStatesPage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);
  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [analyses, setAnalyses] = useState([]);
  const [selectedAnalysisId, setSelectedAnalysisId] = useState('');
  const [analysisData, setAnalysisData] = useState(null);
  const [dataError, setDataError] = useState(null);
  const [loadingData, setLoadingData] = useState(false);

  const [selectedSampleIds, setSelectedSampleIds] = useState([]);
  const [mdLabelMode, setMdLabelMode] = useState('assigned');
  const [pMin, setPMin] = useState(0.005);
  const [pMax, setPMax] = useState(0.05);
  const [enrichmentMin, setEnrichmentMin] = useState(1.0);
  const [topKNodes, setTopKNodes] = useState(500);
  const [includeEdges, setIncludeEdges] = useState(true);
  const [edgeMode, setEdgeMode] = useState('cluster');
  const [deltaPmiMin, setDeltaPmiMin] = useState('');
  const [topKEdges, setTopKEdges] = useState(1000);
  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [jobError, setJobError] = useState(null);
  const [showRunPanel, setShowRunPanel] = useState(false);
  const [nodeSampleFilter, setNodeSampleFilter] = useState('all');
  const [edgeSampleFilter, setEdgeSampleFilter] = useState('all');
  const [helpOpen, setHelpOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        const payload = await fetchSystem(projectId, systemId);
        if (!cancelled) setSystem(payload);
      } catch (err) {
        if (!cancelled) setSystemError(err.message || 'Failed to load system.');
      } finally {
        if (!cancelled) setLoadingSystem(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [projectId, systemId]);

  const clusters = useMemo(() => (system?.metastable_clusters || []).filter((c) => c.cluster_id), [system]);
  const selectedCluster = useMemo(() => clusters.find((c) => c.cluster_id === selectedClusterId) || null, [clusters, selectedClusterId]);
  const samples = useMemo(() => selectedCluster?.samples || [], [selectedCluster]);

  useEffect(() => {
    if (!clusters.length) return;
    const qs = new URLSearchParams(location.search || '');
    const requested = String(qs.get('cluster_id') || '').trim();
    if (requested && clusters.some((c) => c.cluster_id === requested)) {
      setSelectedClusterId(requested);
      return;
    }
    if (!selectedClusterId || !clusters.some((c) => c.cluster_id === selectedClusterId)) setSelectedClusterId(clusters[0].cluster_id);
  }, [clusters, selectedClusterId, location.search]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    try {
      const payload = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'transient_states' });
      const arr = payload?.analyses || [];
      setAnalyses(arr);
      setSelectedAnalysisId((prev) => (prev && arr.some((a) => a.analysis_id === prev) ? prev : String(arr[0]?.analysis_id || '')));
    } catch (err) {
      setDataError(err.message || 'Failed to load analyses.');
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => { loadAnalyses(); }, [loadAnalyses]);

  useEffect(() => {
    if (!selectedAnalysisId || !selectedClusterId) {
      setAnalysisData(null);
      return;
    }
    let cancelled = false;
    const load = async () => {
      setLoadingData(true);
      setDataError(null);
      try {
        const payload = await fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'transient_states', selectedAnalysisId);
        if (!cancelled) setAnalysisData(payload);
      } catch (err) {
        if (!cancelled) setDataError(err.message || 'Failed to load analysis data.');
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
    const terminal = new Set(['finished', 'failed', 'stopped']);
    const poll = async () => {
      try {
        const status = await fetchJobStatus(job.job_id);
        if (cancelled) return;
        setJobStatus(status);
        if (terminal.has(status?.status)) {
          if (status.status === 'finished') await loadAnalyses();
          return;
        }
        setTimeout(poll, 2000);
      } catch (err) {
        if (!cancelled) setJobError(err.message || 'Failed to poll job.');
      }
    };
    poll();
    return () => { cancelled = true; };
  }, [job, loadAnalyses]);

  const submit = useCallback(async () => {
    setJobError(null);
    if (!selectedClusterId) return;
    if (selectedSampleIds.length < 2) {
      setJobError('Select at least two samples.');
      return;
    }
    try {
      const payload = {
        project_id: projectId,
        system_id: systemId,
        cluster_id: selectedClusterId,
        sample_ids: selectedSampleIds,
        md_label_mode: mdLabelMode,
        p_min: Number(pMin),
        p_max: Number(pMax),
        enrichment_min: Number(enrichmentMin),
        top_k_nodes: Number(topKNodes),
        include_edges: includeEdges,
        edge_mode: edgeMode,
        top_k_edges: Number(topKEdges),
      };
      if (String(deltaPmiMin).trim() !== '') payload.delta_pmi_min = Number(deltaPmiMin);
      const res = await submitTransientStatesJob(payload);
      setJob(res);
      setJobStatus(null);
      setShowRunPanel(false);
    } catch (err) {
      setJobError(err.message || 'Failed to submit transient-state analysis.');
    }
  }, [projectId, systemId, selectedClusterId, selectedSampleIds, mdLabelMode, pMin, pMax, enrichmentMin, topKNodes, includeEdges, edgeMode, topKEdges, deltaPmiMin]);

  const data = useMemo(() => analysisData?.data || {}, [analysisData]);
  const sampleLabels = safeArray(data.sample_labels);
  const sampleOptions = useMemo(() => sampleLabels.map((label, idx) => ({ idx, label: String(label || idx) })), [sampleLabels]);

  const nodeRows = useMemo(() => {
    const n = safeArray(data.node_score).length;
    const rows = [];
    for (let i = 0; i < n; i += 1) {
      const sampleIndex = Number(data.node_sample_index?.[i]);
      if (nodeSampleFilter !== 'all' && String(sampleIndex) !== String(nodeSampleFilter)) continue;
      rows.push({
        sampleIndex,
        sample: String(data.node_sample_label?.[i] || sampleLabels[sampleIndex] || sampleIndex),
        residue: String(data.node_residue_label?.[i] || data.node_residue_index?.[i] || ''),
        cluster: Number(data.node_cluster?.[i]),
        occupancy: Number(data.node_occupancy?.[i]),
        background: Number(data.node_background?.[i]),
        enrichment: Number(data.node_log2_enrichment?.[i]),
        episodes: Number(data.node_episodes?.[i]),
        meanDwell: Number(data.node_mean_dwell?.[i]),
        maxDwell: Number(data.node_max_dwell?.[i]),
        score: Number(data.node_score?.[i]),
      });
    }
    return rows.sort((a, b) => b.score - a.score);
  }, [data, sampleLabels, nodeSampleFilter]);

  const edgeRows = useMemo(() => {
    const n = safeArray(data.edge_score).length;
    const rows = [];
    for (let i = 0; i < n; i += 1) {
      const sampleIndex = Number(data.edge_sample_index?.[i]);
      if (edgeSampleFilter !== 'all' && String(sampleIndex) !== String(edgeSampleFilter)) continue;
      rows.push({
        sampleIndex,
        sample: String(data.edge_sample_label?.[i] || sampleLabels[sampleIndex] || sampleIndex),
        edge: String(data.edge_label?.[i] || ''),
        ci: Number(data.edge_cluster_i?.[i]),
        cj: Number(data.edge_cluster_j?.[i]),
        occupancy: Number(data.edge_occupancy?.[i]),
        background: Number(data.edge_background?.[i]),
        enrichment: Number(data.edge_log2_enrichment?.[i]),
        deltaPmi: Number(data.edge_delta_pmi?.[i]),
        episodes: Number(data.edge_episodes?.[i]),
        meanDwell: Number(data.edge_mean_dwell?.[i]),
        maxDwell: Number(data.edge_max_dwell?.[i]),
        score: Number(data.edge_score?.[i]),
      });
    }
    return rows.sort((a, b) => b.score - a.score);
  }, [data, sampleLabels, edgeSampleFilter]);

  if (loadingSystem) return <Loader message="Loading transient-state analysis..." />;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <HelpDrawer
        open={helpOpen}
        onClose={() => setHelpOpen(false)}
        title="Transient-State Analysis Help"
        docPath="/docs/transient_states_help.md"
      />
      <main className="max-w-[1500px] mx-auto px-6 py-6 space-y-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <button type="button" onClick={() => navigate(`/projects/${projectId}/systems/${systemId}`)} className="text-xs text-cyan-300 hover:text-cyan-200">← Back to system</button>
            <h1 className="text-2xl font-semibold text-white mt-2">Transient-State Analysis</h1>
            <p className="text-sm text-gray-400 max-w-4xl">
              Detect low-occupancy residue clusters and joint edge states that are selectively enriched in one trajectory relative to the leave-one-out background.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => setHelpOpen(true)} className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800"><CircleHelp className="h-4 w-4" /> Help</button>
            <button type="button" onClick={loadAnalyses} className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-gray-700 text-sm text-gray-200 hover:bg-gray-800"><RefreshCw className="h-4 w-4" /> Refresh</button>
          </div>
        </div>
        {systemError && <ErrorMessage message={systemError} />}
        {dataError && <ErrorMessage message={dataError} />}
        {jobError && <ErrorMessage message={jobError} />}

        <div className="grid grid-cols-1 xl:grid-cols-[360px_minmax(0,1fr)] gap-5">
          <aside className="space-y-4">
            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
              <label className="block text-xs text-gray-400">Cluster</label>
              <select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="w-full rounded-md bg-gray-950 border border-gray-700 px-2 py-2 text-sm">
                {clusters.map((c) => <option key={c.cluster_id} value={c.cluster_id}>{c.name || c.cluster_id}</option>)}
              </select>
              <button type="button" onClick={() => setShowRunPanel((v) => !v)} className="w-full rounded-md bg-cyan-600 hover:bg-cyan-500 text-white px-3 py-2 text-sm">Run new analysis</button>
              {jobStatus && <div className="text-xs text-gray-300">Job: {jobStatus.status} · {jobStatus.progress ?? 0}% · {jobStatus.status_text || jobStatus.meta?.status || ''}</div>}
            </div>

            {showRunPanel && (
              <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
                <div className="text-sm font-semibold">Run setup</div>
                <div className="text-xs text-gray-400">Samples to compare</div>
                <div className="max-h-56 overflow-auto rounded-md border border-gray-800">
                  {samples.map((s) => {
                    const sid = String(s.sample_id);
                    const checked = selectedSampleIds.includes(sid);
                    return (
                      <label key={sid} className="flex items-center gap-2 px-2 py-1.5 text-xs border-b border-gray-900">
                        <input type="checkbox" checked={checked} onChange={(e) => setSelectedSampleIds((prev) => e.target.checked ? [...prev, sid] : prev.filter((x) => x !== sid))} />
                        <span>{s.name || sid}</span>
                      </label>
                    );
                  })}
                </div>
                <label className="block text-xs text-gray-400">MD label mode</label>
                <select value={mdLabelMode} onChange={(e) => setMdLabelMode(e.target.value)} className="w-full rounded-md bg-gray-950 border border-gray-700 px-2 py-1.5 text-sm"><option value="assigned">assigned</option><option value="halo">halo</option></select>
                <div className="grid grid-cols-3 gap-2">
                  <label className="text-xs text-gray-400">p min<input type="number" step="0.001" value={pMin} onChange={(e) => setPMin(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-1" /></label>
                  <label className="text-xs text-gray-400">p max<input type="number" step="0.001" value={pMax} onChange={(e) => setPMax(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-1" /></label>
                  <label className="text-xs text-gray-400">log2 enrich<input type="number" step="0.1" value={enrichmentMin} onChange={(e) => setEnrichmentMin(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-1" /></label>
                </div>
                <label className="text-xs text-gray-400 block">Top node hits<input type="number" value={topKNodes} onChange={(e) => setTopKNodes(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-1" /></label>
                <label className="flex items-center gap-2 text-xs text-gray-300"><input type="checkbox" checked={includeEdges} onChange={(e) => setIncludeEdges(e.target.checked)} /> Compute edge states</label>
                {includeEdges && <>
                  <select value={edgeMode} onChange={(e) => setEdgeMode(e.target.value)} className="w-full rounded-md bg-gray-950 border border-gray-700 px-2 py-1.5 text-sm"><option value="cluster">cluster edges</option><option value="all_vs_all">all-vs-all</option></select>
                  <label className="text-xs text-gray-400 block">ΔPMI min (optional)<input type="number" step="0.1" value={deltaPmiMin} onChange={(e) => setDeltaPmiMin(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-1" /></label>
                  <label className="text-xs text-gray-400 block">Top edge hits<input type="number" value={topKEdges} onChange={(e) => setTopKEdges(e.target.value)} className="mt-1 w-full rounded bg-gray-950 border border-gray-700 px-2 py-1" /></label>
                </>}
                <button type="button" onClick={submit} className="w-full rounded-md bg-cyan-600 hover:bg-cyan-500 text-white px-3 py-2 text-sm">Submit</button>
              </div>
            )}

            <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-2">
              <div className="text-sm font-semibold">Existing analyses</div>
              <div className="max-h-72 overflow-auto rounded-md border border-gray-800">
                {!analyses.length && <div className="p-3 text-xs text-gray-500">No transient-state analyses yet.</div>}
                {analyses.map((a) => {
                  const active = String(a.analysis_id) === String(selectedAnalysisId);
                  return <button key={a.analysis_id} type="button" onClick={() => setSelectedAnalysisId(String(a.analysis_id))} className={`w-full text-left px-3 py-2 border-b border-gray-900 hover:bg-gray-800/40 ${active ? 'bg-cyan-950/40 border-l-2 border-l-cyan-500' : ''}`}><div className="text-xs text-gray-100">{a.analysis_id}</div><div className="text-[11px] text-gray-400">{analysisLabel(a)}</div></button>;
                })}
              </div>
            </div>
          </aside>

          <section className="space-y-5 min-w-0">
            {loadingData && <Loader message="Loading analysis data..." />}
            {!loadingData && analysisData && <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {Object.entries(analysisData.metadata?.summary || {}).map(([k, v]) => <div key={k} className="rounded-lg border border-gray-800 bg-gray-900/40 p-3"><div className="text-[11px] text-gray-500">{k}</div><div className="text-lg text-white">{String(v)}</div></div>)}
              </div>

              <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
                <div className="flex items-center justify-between gap-3"><h2 className="font-semibold">Top transient residue states</h2><select value={nodeSampleFilter} onChange={(e) => setNodeSampleFilter(e.target.value)} className="rounded bg-gray-950 border border-gray-700 px-2 py-1 text-xs"><option value="all">all samples</option>{sampleOptions.map((s) => <option key={s.idx} value={s.idx}>{s.label}</option>)}</select></div>
                {!!nodeRows.length && <Plot data={[{ type: 'bar', x: nodeRows.slice(0, 30).map((r) => `${r.residue} c${r.cluster} · ${r.sample}`), y: nodeRows.slice(0, 30).map((r) => r.score), marker: { color: nodeRows.slice(0, 30).map((r) => r.enrichment), colorscale: 'YlOrRd', showscale: true, colorbar: { title: 'log2 enrich' } }, customdata: nodeRows.slice(0, 30).map((r) => [fmtPct(r.occupancy), fmtPct(r.background), r.episodes, fmt(r.meanDwell, 1), r.maxDwell]), hovertemplate: '%{x}<br>score=%{y:.2f}<br>occupancy=%{customdata[0]}<br>background=%{customdata[1]}<br>episodes=%{customdata[2]}<br>mean dwell=%{customdata[3]}<br>max dwell=%{customdata[4]}<extra></extra>' }]} layout={{ paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#d1d5db' }, margin: { l: 60, r: 30, t: 10, b: 160 }, height: 430, xaxis: { tickangle: -45, automargin: true } }} config={{ responsive: true, displaylogo: false }} style={{ width: '100%' }} />}
                <div className="max-h-96 overflow-auto rounded-md border border-gray-800"><table className="w-full text-xs"><thead className="sticky top-0 bg-gray-950"><tr className="text-gray-300"><th className="px-2 py-2 text-left">Residue</th><th>Sample</th><th>Cluster</th><th>Occ.</th><th>Bg.</th><th>log2 enrich</th><th>Episodes</th><th>Mean dwell</th><th>Max dwell</th><th>Score</th></tr></thead><tbody>{nodeRows.map((r, i) => <tr key={`n${i}`} className="border-t border-gray-900"><td className="px-2 py-1.5 text-gray-100">{r.residue}</td><td>{r.sample}</td><td className="text-center">c{r.cluster}</td><td className="text-right">{fmtPct(r.occupancy)}</td><td className="text-right">{fmtPct(r.background)}</td><td className="text-right">{fmt(r.enrichment, 2)}</td><td className="text-right">{r.episodes}</td><td className="text-right">{fmt(r.meanDwell, 1)}</td><td className="text-right">{r.maxDwell}</td><td className="text-right pr-2">{fmt(r.score, 2)}</td></tr>)}</tbody></table></div>
              </div>

              <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4 space-y-3">
                <div className="flex items-center justify-between gap-3"><h2 className="font-semibold">Top transient edge states</h2><select value={edgeSampleFilter} onChange={(e) => setEdgeSampleFilter(e.target.value)} className="rounded bg-gray-950 border border-gray-700 px-2 py-1 text-xs"><option value="all">all samples</option>{sampleOptions.map((s) => <option key={s.idx} value={s.idx}>{s.label}</option>)}</select></div>
                <div className="max-h-96 overflow-auto rounded-md border border-gray-800"><table className="w-full text-xs"><thead className="sticky top-0 bg-gray-950"><tr className="text-gray-300"><th className="px-2 py-2 text-left">Edge</th><th>Sample</th><th>Clusters</th><th>Occ.</th><th>Bg.</th><th>log2 enrich</th><th>ΔPMI</th><th>Episodes</th><th>Mean dwell</th><th>Max dwell</th><th>Score</th></tr></thead><tbody>{edgeRows.map((r, i) => <tr key={`e${i}`} className="border-t border-gray-900"><td className="px-2 py-1.5 text-gray-100">{r.edge}</td><td>{r.sample}</td><td className="text-center">c{r.ci}/c{r.cj}</td><td className="text-right">{fmtPct(r.occupancy)}</td><td className="text-right">{fmtPct(r.background)}</td><td className="text-right">{fmt(r.enrichment, 2)}</td><td className="text-right">{fmt(r.deltaPmi, 2)}</td><td className="text-right">{r.episodes}</td><td className="text-right">{fmt(r.meanDwell, 1)}</td><td className="text-right">{r.maxDwell}</td><td className="text-right pr-2">{fmt(r.score, 2)}</td></tr>)}</tbody></table></div>
              </div>
            </>}
          </section>
        </div>
      </main>
    </div>
  );
}
