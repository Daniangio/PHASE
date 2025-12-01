import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
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
  const navigate = useNavigate();

  const [system, setSystem] = useState(null);
  const [loadingSystem, setLoadingSystem] = useState(true);
  const [error, setError] = useState(null);

  const [selectedStates, setSelectedStates] = useState([]);
  const [residueFilter, setResidueFilter] = useState('');
  const [selectedResidues, setSelectedResidues] = useState([]);
  const [maxPoints, setMaxPoints] = useState(2000);

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
          setSelectedStates([descriptorStates[0].state_id]);
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

  const residueKeys = useMemo(() => {
    const keys = new Set();
    Object.values(metaByState).forEach((meta) =>
      (meta?.residue_keys || []).forEach((key) => keys.add(key))
    );
    return Array.from(keys).sort();
  }, [metaByState]);

  const residueLabel = useCallback(
    (key) => {
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
    [metaByState, selectedStates]
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

  const stateColors = useMemo(() => {
    const mapping = {};
    selectedStates.forEach((stateId, idx) => {
      mapping[stateId] = colors[idx % colors.length];
    });
    return mapping;
  }, [selectedStates]);

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

  const toggleState = (stateId) => {
    setSelectedStates((prev) =>
      prev.includes(stateId) ? prev.filter((id) => id !== stateId) : [...prev, stateId]
    );
  };

  const toggleResidue = (key) => {
    setSelectedResidues((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  const loadAngles = useCallback(async () => {
    if (!selectedStates.length) {
      setAnglesByState({});
      setMetaByState({});
      return;
    }
    setLoadingAngles(true);
    setAnglesError(null);
    try {
      const qs = { max_points: maxPoints };

      const responses = await Promise.all(
        selectedStates.map(async (stateId) => {
          const data = await fetchStateDescriptors(projectId, systemId, stateId, qs);
          return { stateId, data };
        })
      );

      const newAngles = {};
      const newMeta = {};
      const unionResidues = new Set();

      responses.forEach(({ stateId, data }) => {
        newAngles[stateId] = data.angles || {};
        newMeta[stateId] = {
          residue_keys: data.residue_keys || [],
          residue_mapping: data.residue_mapping || {},
          residue_labels: data.residue_labels || {},
          n_frames: data.n_frames,
          sample_stride: data.sample_stride,
        };
        (data.residue_keys || []).forEach((key) => unionResidues.add(key));
      });

      setAnglesByState(newAngles);
      setMetaByState(newMeta);

      if (unionResidues.size) {
        if (!selectedResidues.length) {
          const defaultKey = Array.from(unionResidues).sort()[0];
          setSelectedResidues([defaultKey]);
        } else {
          const validResidues = selectedResidues.filter((key) => unionResidues.has(key));
          if (validResidues.length !== selectedResidues.length) {
            setSelectedResidues(validResidues);
          }
        }
      } else {
        setSelectedResidues([]);
      }
    } catch (err) {
      setAnglesError(err.message);
    } finally {
      setLoadingAngles(false);
    }
  }, [maxPoints, projectId, selectedStates, systemId]);

  useEffect(() => {
    // Auto-load when state selection changes
    setAnglesByState({});
    setMetaByState({});
    if (selectedStates.length) {
      loadAngles();
    } else {
      setSelectedResidues([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStates]);

  const traces3d = useMemo(() => {
    const traces = [];
    selectedStates.forEach((stateId) => {
      const perState = anglesByState[stateId] || {};
      selectedResidues.forEach((key) => {
        const data = perState[key];
        if (!data) return;
        traces.push({
          type: 'scatter3d',
          mode: 'markers',
          x: data.phi,
          y: data.psi,
          z: data.chi1,
          name: `${residueLabel(key)} — ${stateName(stateId)}`,
          legendgroup: residueLabel(key),
          marker: {
            size: 3,
            opacity: 0.75,
            color: stateColors[stateId],
            symbol: residueSymbols[key] || 'circle',
          },
        });
      });
    });
    return traces;
  }, [
    anglesByState,
    residueLabel,
    residueSymbols,
    selectedResidues,
    selectedStates,
    stateColors,
    stateName,
  ]);

  const make2DTraces = useCallback(
    (axisX, axisY) =>
      selectedStates
        .map((stateId) => {
          const perState = anglesByState[stateId] || {};
          return selectedResidues.map((key) => {
            const data = perState[key];
            if (!data) return null;
            return {
              type: 'scattergl',
              mode: 'markers',
              x: data[axisX],
              y: data[axisY],
              name: `${residueLabel(key)} — ${stateName(stateId)}`,
              legendgroup: residueLabel(key),
              marker: {
                size: 4,
                opacity: 0.7,
                color: stateColors[stateId],
                symbol: residueSymbols[key] || 'circle',
              },
            };
          });
        })
        .flat()
        .filter(Boolean),
    [
      anglesByState,
      residueLabel,
      residueSymbols,
      selectedResidues,
      selectedStates,
      stateColors,
      stateName,
    ]
  );

  const hasAngles = useMemo(
    () => Object.values(anglesByState).some((residues) => Object.keys(residues || {}).length > 0),
    [anglesByState]
  );

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
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
          <div className="grid md:grid-cols-4 gap-3">
            <div className="md:col-span-2">
              <label className="block text-xs text-gray-400 mb-1">States</label>
              <div className="grid sm:grid-cols-2 gap-2 max-h-28 overflow-y-auto border border-gray-700 rounded-md p-2 bg-gray-900">
                {descriptorStates.map((state) => (
                  <label key={state.state_id} className="flex items-center space-x-2 text-sm text-gray-200">
                    <input
                      type="checkbox"
                      checked={selectedStates.includes(state.state_id)}
                      onChange={() => toggleState(state.state_id)}
                      className="accent-cyan-500"
                    />
                    <span>{state.name}</span>
                  </label>
                ))}
              </div>
              <p className="text-[11px] text-gray-500 mt-1">
                Select one or more states to compare residue distributions.
              </p>
            </div>
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
            <div className="flex items-end">
              <button
                onClick={loadAngles}
                disabled={loadingAngles || !selectedStates.length}
                className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-semibold py-2 rounded-md disabled:opacity-50"
              >
                {loadingAngles ? 'Loading…' : selectedStates.length ? 'Refresh data' : 'Select states'}
              </button>
            </div>
          </div>

          <div className="mt-2">
            <label className="block text-xs text-gray-400 mb-1">Filter residues</label>
            <input
              type="text"
              value={residueFilter}
              onChange={(e) => setResidueFilter(e.target.value)}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
              placeholder="Search residue keys"
            />
          </div>

          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-2 max-h-48 overflow-y-auto border border-gray-700 rounded-md p-2 bg-gray-900">
            {filteredResidues.length === 0 && (
              <p className="text-sm text-gray-500 col-span-full">No residues match this filter.</p>
            )}
            {filteredResidues.map((key) => (
              <label key={key} className="flex items-center space-x-2 text-sm text-gray-200">
                <input
                  type="checkbox"
                  checked={selectedResidues.includes(key)}
                  onChange={() => toggleResidue(key)}
                  className="accent-cyan-500"
                />
                <span>{residueLabel(key)}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {anglesError && <ErrorMessage message={anglesError} />}
      {loadingAngles && <Loader message="Loading angles..." />}

      {!loadingAngles && !hasAngles && (
        <p className="text-sm text-gray-400">
          {selectedStates.length
            ? 'Select residues to visualize.'
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
                  xaxis: { title: 'Phi (°)' },
                  yaxis: { title: 'Psi (°)' },
                  zaxis: { title: 'Chi1 (°)' },
                },
                margin: { l: 0, r: 0, t: 10, b: 0 },
                legend: { bgcolor: 'rgba(0,0,0,0)' },
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
                    xaxis: { title: `${axes.x.toUpperCase()} (°)` },
                    yaxis: { title: `${axes.y.toUpperCase()} (°)` },
                    legend: { bgcolor: 'rgba(0,0,0,0)' },
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
    </div>
  );
}
