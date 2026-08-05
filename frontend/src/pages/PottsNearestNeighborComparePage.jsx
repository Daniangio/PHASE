import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useParams } from 'react-router-dom';
import Plot from 'react-plotly.js';
import { RefreshCw } from 'lucide-react';

import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { buildKdeCurve } from '../components/common/EnergyDistributionPlot';
import NearestNeighborWorkspaceTabs from '../components/potts/NearestNeighborWorkspaceTabs';
import { fetchClusterAnalyses, fetchClusterAnalysisData, fetchSystem } from '../api/projects';

const palette = ['#22d3ee', '#f97316', '#10b981', '#f43f5e', '#60a5fa', '#f59e0b', '#a78bfa', '#facc15'];

export default function PottsNearestNeighborComparePage() {
  const { projectId, systemId } = useParams();
  const location = useLocation();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [systemError, setSystemError] = useState(null);

  const [selectedClusterId, setSelectedClusterId] = useState('');
  const [analyses, setAnalyses] = useState([]);
  const [analysesError, setAnalysesError] = useState(null);
  const [analysesLoading, setAnalysesLoading] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]);
  const [rowCap, setRowCap] = useState('1500');
  const [distanceGraphMode, setDistanceGraphMode] = useState('histogram');

  const [seriesById, setSeriesById] = useState({});
  const [seriesError, setSeriesError] = useState(null);
  const [seriesLoading, setSeriesLoading] = useState(false);

  const cacheRef = useRef({});
  const inFlightRef = useRef({});

  const clusters = useMemo(
    () => (system?.metastable_clusters || []).filter((cluster) => cluster?.cluster_id),
    [system]
  );

  const selectedCluster = useMemo(
    () => clusters.find((cluster) => cluster.cluster_id === selectedClusterId) || null,
    [clusters, selectedClusterId]
  );

  useEffect(() => {
    const run = async () => {
      setLoadingSystem(true);
      setSystemError(null);
      try {
        setSystem(await fetchSystem(projectId, systemId));
      } catch (err) {
        setSystemError(err.message || 'Failed to load system.');
      } finally {
        setLoadingSystem(false);
      }
    };
    run();
  }, [projectId, systemId]);

  useEffect(() => {
    if (!clusters.length) return;
    const requestedClusterId = new URLSearchParams(location.search || '').get('cluster_id') || '';
    if (requestedClusterId && clusters.some((cluster) => cluster.cluster_id === requestedClusterId)) {
      setSelectedClusterId(requestedClusterId);
      return;
    }
    if (!selectedClusterId || !clusters.some((cluster) => cluster.cluster_id === selectedClusterId)) {
      setSelectedClusterId(clusters[0].cluster_id);
    }
  }, [clusters, selectedClusterId, location.search]);

  const loadAnalyses = useCallback(async () => {
    if (!selectedClusterId) return;
    setAnalysesLoading(true);
    setAnalysesError(null);
    try {
      const payload = await fetchClusterAnalyses(projectId, systemId, selectedClusterId, { analysisType: 'potts_nn_mapping' });
      const list = Array.isArray(payload?.analyses) ? payload.analyses : [];
      setAnalyses(list);
      setSelectedIds((prev) => {
        const filtered = prev.filter((id) => list.some((row) => row.analysis_id === id));
        if (filtered.length) return filtered;
        return list.slice(0, 2).map((row) => row.analysis_id);
      });
    } catch (err) {
      setAnalyses([]);
      setSelectedIds([]);
      setAnalysesError(err.message || 'Failed to load analyses.');
    } finally {
      setAnalysesLoading(false);
    }
  }, [projectId, systemId, selectedClusterId]);

  useEffect(() => {
    if (!selectedClusterId) return;
    cacheRef.current = {};
    inFlightRef.current = {};
    setSeriesById({});
    setSeriesError(null);
    loadAnalyses();
  }, [selectedClusterId, loadAnalyses]);

  const loadAnalysisData = useCallback(async (analysisId) => {
    const maxRows = Number(rowCap) > 0 ? Number(rowCap) : null;
    const cacheKey = `${selectedClusterId}:${analysisId}:${maxRows ?? 'all'}`;
    if (Object.prototype.hasOwnProperty.call(cacheRef.current, cacheKey)) return cacheRef.current[cacheKey];
    if (inFlightRef.current[cacheKey]) return inFlightRef.current[cacheKey];
    const promise = fetchClusterAnalysisData(projectId, systemId, selectedClusterId, 'potts_nn_mapping', analysisId, {
      maxRows,
      sampleSeed: 0,
    }).then((payload) => {
      cacheRef.current = { ...cacheRef.current, [cacheKey]: payload };
      delete inFlightRef.current[cacheKey];
      return payload;
    }).catch((err) => {
      delete inFlightRef.current[cacheKey];
      throw err;
    });
    inFlightRef.current[cacheKey] = promise;
    return promise;
  }, [projectId, systemId, selectedClusterId, rowCap]);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!selectedIds.length) {
        setSeriesById({});
        return;
      }
      setSeriesLoading(true);
      setSeriesError(null);
      try {
        const rows = await Promise.all(selectedIds.map(async (analysisId) => {
          const payload = await loadAnalysisData(analysisId);
          const distances = Array.isArray(payload?.data?.nn_dist_global) ? payload.data.nn_dist_global.map(Number) : [];
          const counts = Array.isArray(payload?.data?.sample_unique_counts) ? payload.data.sample_unique_counts.map(Number) : [];
          return [analysisId, { payload, distances, counts }];
        }));
        if (cancelled) return;
        setSeriesById(Object.fromEntries(rows));
      } catch (err) {
        if (cancelled) return;
        setSeriesById({});
        setSeriesError(err.message || 'Failed to load selected analyses.');
      } finally {
        if (!cancelled) setSeriesLoading(false);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [selectedIds, loadAnalysisData]);

  const histogram = useMemo(() => {
    if (!selectedIds.length) return null;
    const availableIds = selectedIds.filter((analysisId) => seriesById && Object.prototype.hasOwnProperty.call(seriesById, analysisId));
    if (!availableIds.length) return null;
    const data = availableIds.reduce((allTraces, analysisId, idx) => {
      const item = seriesById?.[analysisId];
      const meta = analyses.find((row) => row.analysis_id === analysisId);
      const name = meta ? `${meta.sample_name || meta.sample_id} → ${meta.md_sample_name || meta.md_sample_id}` : analysisId;
      const distances = Array.isArray(item?.distances) ? item.distances : [];
      const counts = Array.isArray(item?.counts) ? item.counts : [];
      const color = palette[idx % palette.length];
      const traces = [];
      if (distanceGraphMode === 'histogram') {
        traces.push({
          type: 'histogram',
          histnorm: 'probability density',
          x: distances,
          weights: counts.length === distances.length ? counts : undefined,
          opacity: 0.42,
          marker: { color },
          name: `${name} histogram`,
          showlegend: false,
        });
      }
      const curve = buildKdeCurve(distances, { weights: counts.length === distances.length ? counts : null });
      if (curve.x.length) {
        traces.push({ type: 'scatter', mode: 'lines', x: curve.x, y: curve.y, line: { color, width: 2.5 }, name });
      }
      return [...allTraces, ...traces];
    }, []);
    return {
      data,
      layout: {
        barmode: 'overlay',
        paper_bgcolor: '#111827',
        plot_bgcolor: '#111827',
        font: { color: '#e5e7eb' },
        margin: { t: 36, r: 16, b: 48, l: 54 },
        title: distanceGraphMode === 'histogram' ? 'Nearest-neighbor distance distributions (overlay)' : 'Nearest-neighbor fitted distance curves',
        xaxis: { title: 'Distance' },
        yaxis: { title: 'Probability density' },
        legend: { orientation: 'h', y: -0.2 },
      },
      config: { responsive: true, displaylogo: false },
    };
  }, [selectedIds, seriesById, analyses, distanceGraphMode]);

  const toggleSelection = (analysisId) => {
    setSelectedIds((prev) => (prev.includes(analysisId) ? prev.filter((id) => id !== analysisId) : [...prev, analysisId]));
  };

  if (loadingSystem) return <Loader message="Loading Potts NN compare..." />;
  if (systemError) return <ErrorMessage message={systemError} />;

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-white">Potts NN Mapping Compare</h1>
          <p className="text-sm text-gray-400">Overlay distance distributions from multiple Potts NN mapping analyses.</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={loadAnalyses}
            className="inline-flex items-center gap-2 text-xs px-3 py-2 rounded-md border border-gray-700 text-gray-200 hover:bg-gray-700/40"
          >
            <RefreshCw className="h-4 w-4" /> Refresh
          </button>
        </div>
      </div>

      <NearestNeighborWorkspaceTabs
        projectId={projectId}
        systemId={systemId}
        clusterId={selectedClusterId}
        active="compare"
      />

      <div className="grid grid-cols-1 xl:grid-cols-[360px,minmax(0,1fr)] gap-4">
        <aside className="space-y-3 rounded-lg border border-gray-800 bg-gray-900/70 p-4 h-fit">
          <label className="block text-sm text-gray-300">
            Cluster
            <select value={selectedClusterId} onChange={(e) => setSelectedClusterId(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white">
              {clusters.map((cluster) => (
                <option key={cluster.cluster_id} value={cluster.cluster_id}>{cluster.name || cluster.cluster_id}</option>
              ))}
            </select>
          </label>
          <label className="block text-sm text-gray-300">
            Max unique rows to load
            <select value={rowCap} onChange={(e) => setRowCap(e.target.value)} className="mt-1 w-full rounded-md bg-gray-950 border border-gray-700 px-3 py-2 text-sm text-white">
              <option value="500">500</option>
              <option value="1000">1000</option>
              <option value="1500">1500</option>
              <option value="3000">3000</option>
              <option value="5000">5000</option>
              <option value="0">All</option>
            </select>
          </label>
          <div className="border-t border-gray-800 pt-3">
            <p className="text-xs uppercase tracking-[0.2em] text-gray-500 mb-2">Experiments</p>
            {analysesLoading && <p className="text-xs text-gray-400">Loading…</p>}
            {!analysesLoading && analyses.length === 0 && <p className="text-xs text-gray-500">No analyses yet.</p>}
            <div className="space-y-2 max-h-[60vh] overflow-auto pr-1">
              {analyses.map((analysis) => {
                const selected = selectedIds.includes(analysis.analysis_id);
                return (
                  <label key={analysis.analysis_id} className={`block rounded-md border px-3 py-2 cursor-pointer ${selected ? 'border-cyan-500 bg-cyan-500/10' : 'border-gray-800 bg-gray-950/60'}`}>
                    <div className="flex items-start gap-2">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleSelection(analysis.analysis_id)}
                        className="mt-0.5"
                      />
                      <div className="min-w-0">
                        <p className="text-sm text-white truncate">{analysis.sample_name || analysis.sample_id} → {analysis.md_sample_name || analysis.md_sample_id}</p>
                        <p className="text-[11px] text-gray-500">{analysis.model_name || analysis.model_id}</p>
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
          </div>
          {analysesError && <ErrorMessage message={analysesError} />}
          {seriesError && <ErrorMessage message={seriesError} />}
        </aside>

        <main className="space-y-4 min-w-0">
          {!selectedIds.length && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-6 text-sm text-gray-400">
              Select at least one experiment.
            </div>
          )}
          {seriesLoading && <Loader message="Loading selected experiments..." />}
          {histogram && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-3">
              <div className="mb-2 flex items-center justify-end gap-2">
                <label className="text-xs text-gray-400" htmlFor="nn-compare-distance-graph-mode">Distance display</label>
                <select
                  id="nn-compare-distance-graph-mode"
                  value={distanceGraphMode}
                  onChange={(event) => setDistanceGraphMode(event.target.value)}
                  className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1 text-xs text-gray-100"
                >
                  <option value="histogram">Histogram + fitted curves</option>
                  <option value="curves">Fitted curves only</option>
                </select>
              </div>
              <Plot data={histogram.data} layout={histogram.layout} config={histogram.config} useResizeHandler style={{ width: '100%', height: 520 }} />
            </div>
          )}
          {selectedCluster && (
            <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-3 text-xs text-gray-500">
              Cluster: {selectedCluster.name || selectedCluster.cluster_id}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
