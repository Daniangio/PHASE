import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchResult } from '../api/jobs';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { Asset } from 'molstar/lib/mol-util/assets';
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Color } from 'molstar/lib/mol-util/color';
import 'molstar/build/viewer/molstar.css';

const parseResidueIds = (selection) => {
  if (!selection) return [];
  const cleaned = selection.replace(/resid/gi, '');
  const matches = cleaned.match(/-?\d+/g);
  if (!matches) return [];
  return matches.map((val) => Number(val)).filter((id) => Number.isFinite(id));
};

const convertKeysToResidues = (keys, mapping) => {
  if (!keys || !mapping) return [];
  const residues = new Set();
  keys.forEach((key) => {
    parseResidueIds(mapping[key]).forEach((id) => residues.add(id));
  });
  return Array.from(residues);
};

export default function VisualizePage() {
  const { jobId } = useParams();
  const navigate = useNavigate();

  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  const highlightComponentsRef = useRef([]);

  const [resultData, setResultData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const [status, setStatus] = useState('initializing'); // initializing | ready | loading-structure | error
  const [structureError, setStructureError] = useState(null);
  const [staticThreshold, setStaticThreshold] = useState(0.8);
  const [quboSolutionIdx, setQuboSolutionIdx] = useState({ active: 0, inactive: 0 });
  const [loadedStateId, setLoadedStateId] = useState(null);
  const [structureVersion, setStructureVersion] = useState(0);
  const availableStates = Object.values(resultData?.system_reference?.states || {});
  const structureKeys = Object.keys(resultData?.system_reference?.structures || {});
  const stateButtons =
    availableStates.length > 0
      ? availableStates
      : structureKeys.map((key) => ({ id: key, name: key }));

  // Fetch job/result metadata to locate the system and structures
  useEffect(() => {
    const loadResult = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchResult(jobId);
        if (!data.system_reference?.project_id || !data.system_reference?.system_id) {
          throw new Error('Result does not reference a stored system.');
        }
        setResultData(data);
      } catch (err) {
        setError(err.message || 'Failed to load result metadata.');
      } finally {
        setIsLoading(false);
      }
    };
    loadResult();
  }, [jobId]);

  // Initialize Mol* viewer
  useEffect(() => {
    let disposed = false;
    const initViewer = async () => {
      if (!containerRef.current || pluginRef.current) return;
      setStatus((prev) => (prev === 'initializing' ? prev : 'initializing'));
      try {
        const plugin = await createPluginUI({
          target: containerRef.current,
          render: renderReact18,
        });
        if (disposed) {
          plugin.dispose?.();
          return;
        }
        pluginRef.current = plugin;
        setStatus('ready');
      } catch (viewerErr) {
        console.error('Failed to initialize Mol* viewer', viewerErr);
        setError('3D viewer initialization failed.');
        setStatus('error');
      }
    };
    initViewer();
    return () => {
      disposed = true;
      if (pluginRef.current) {
        try {
          pluginRef.current.dispose?.();
        } catch (err) {
          console.warn('Failed to dispose Mol* viewer', err);
        }
        pluginRef.current = null;
      }
    };
  }, [resultData]);

  const clearHighlights = useCallback(async () => {
    const plugin = pluginRef.current;
    if (!plugin || highlightComponentsRef.current.length === 0) return;
    const manager = plugin.managers?.structure?.componentManager;
    if (!manager) return;
    try {
      await manager.removeComponents(highlightComponentsRef.current);
    } catch (err) {
      console.warn('Failed to clear highlights', err);
    }
    highlightComponentsRef.current = [];
  }, []);

  const createHighlightComponent = useCallback(async (residueIds, colorHex) => {
    const plugin = pluginRef.current;
    if (!plugin || residueIds.length === 0) return;
    const structureCell = plugin.managers.structure.hierarchy.current.structures[0]?.cell;
    if (!structureCell) return;
    try {
      const residueTests =
        residueIds.length === 1
          ? MS.core.rel.eq([MS.struct.atomProperty.macromolecular.label_seq_id(), residueIds[0]])
          : MS.core.set.has([
              MS.set(...residueIds),
              MS.struct.atomProperty.macromolecular.label_seq_id(),
            ]);

      const expression = MS.struct.generator.atomGroups({
        'residue-test': residueTests,
      });
      const component = await plugin.builders.structure.tryCreateComponentFromExpression(
        structureCell,
        expression,
        'alloskin-selection'
      );
      if (!component) return;
      highlightComponentsRef.current.push(component);
      await plugin.builders.structure.representation.addRepresentation(component, {
        type: 'ball-and-stick',
        color: 'illustrative',
      });
    } catch (err) {
      console.warn('Highlight failed', err);
    }
  }, []);

  const loadStructure = async (stateId) => {
    if (!pluginRef.current || !resultData) return;
    const { project_id, system_id } = resultData.system_reference || {};
    if (!project_id || !system_id) {
      setError('Result does not reference a stored system.');
      return;
    }

    setStatus('loading-structure');
    setStructureError(null);
    setError(null);

    try {
      await pluginRef.current.clear(); // reset any previous state tree
      await pluginRef.current.dataTransaction(async () => {
        const url = `/api/v1/projects/${project_id}/systems/${system_id}/structures/${stateId}`;
        const data = await pluginRef.current.builders.data.download(
          { url: Asset.Url(url), isBinary: false },
          { state: { isGhost: true } }
        );
        const trajectory = await pluginRef.current.builders.structure.parseTrajectory(data, 'pdb');
        await pluginRef.current.builders.structure.hierarchy.applyPreset(trajectory, 'default');
      });
      // Add a translucent cartoon overlay for context
      const structureCell = pluginRef.current.managers.structure.hierarchy.current.structures[0]?.cell;
      if (structureCell) {
        const allExpr = MS.struct.generator.all();
        const baseComponent = await pluginRef.current.builders.structure.tryCreateComponentFromExpression(
          structureCell,
          allExpr,
          'alloskin-base'
        );
        if (baseComponent) {
          await pluginRef.current.builders.structure.representation.addRepresentation(baseComponent, {
            type: 'cartoon',
            color: { name: 'uniform', params: { value: Color.fromHexString('#9ca3af') } },
            transparency: { name: 'uniform', params: { value: 0.6 } },
          });
        }
      }
      await clearHighlights();
      setStructureVersion((prev) => prev + 1);
      setLoadedStateId(stateId);
      setStatus('ready');
    } catch (err) {
      console.error('Structure load failed', err);
      setStructureError(err.message || 'Failed to load structure.');
      setStatus('ready');
    }
  };

  useEffect(() => {
    const applyStaticHighlights = async () => {
      if (!pluginRef.current || !pluginRef.current.managers?.structure || !resultData || structureVersion === 0) return;
      if (resultData.analysis_type !== 'static') return;
      const { results, residue_selections_mapping } = resultData;
      if (!results || !residue_selections_mapping) return;
      await clearHighlights();
      const selected = Object.keys(results).filter((key) => results[key]?.state_score >= staticThreshold);
      const residues = convertKeysToResidues(selected, residue_selections_mapping);
      await createHighlightComponent(residues, '#ef4444');
    };
    applyStaticHighlights();
  }, [resultData, structureVersion, staticThreshold, clearHighlights, createHighlightComponent]);

  useEffect(() => {
    const applyQuboHighlights = async () => {
      if (!pluginRef.current || !pluginRef.current.managers?.structure || !resultData || structureVersion === 0) return;
      if (resultData.analysis_type !== 'qubo') return;
      if (!loadedStateId) return;
      const mapping = resultData.results?.mapping || resultData.residue_selections_mapping;
      if (!mapping) return;

      const states = resultData.system_reference?.states || {};
      const isActive = loadedStateId === states.state_a?.id;
      const solIdx = isActive ? quboSolutionIdx.active : quboSolutionIdx.inactive;
      const quboKey = isActive ? 'qubo_active' : 'qubo_inactive';
      const solutions = resultData.results?.[quboKey]?.solutions || [];
      const solution = solutions[solIdx] || solutions[0];
      if (!solution) return;

      const residues = convertKeysToResidues(solution.selected || [], mapping);
      await clearHighlights();
      await createHighlightComponent(residues, '#ef4444');
    };
    applyQuboHighlights();
  }, [
    resultData,
    structureVersion,
    quboSolutionIdx.active,
    quboSolutionIdx.inactive,
    loadedStateId,
    clearHighlights,
    createHighlightComponent,
  ]);

  if (isLoading) return <Loader message="Preparing visualization..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!resultData) return null;

  return (
    <div className="space-y-4">
      <button onClick={() => navigate(`/results/${jobId}`)} className="text-cyan-400 hover:text-cyan-300 text-sm">
        ← Back to result
      </button>
      <h1 className="text-2xl font-bold text-white">Visualization: {jobId}</h1>
      <p className="text-sm text-gray-400">
        Use the buttons below to load the stored structures for this system. Once working, we will add the highlighting
        controls back in.
      </p>

      {(structureError || status === 'error') && (
        <ErrorMessage message={structureError || error || 'Viewer error'} />
      )}

      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-3">
        <h2 className="text-lg font-semibold text-white">Load structures</h2>
        {stateButtons.length === 0 ? (
          <p className="text-sm text-gray-400">No states referenced in this result.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {stateButtons.map((state) => (
              <button
                key={state.id || state.state_id}
                type="button"
                onClick={() => loadStructure(state.id || state.state_id)}
                disabled={status !== 'ready'}
                className="px-4 py-2 rounded-md bg-cyan-600 text-white text-sm disabled:opacity-50"
              >
                {status === 'loading-structure' ? 'Loading...' : `Load ${state.name || state.id?.slice(0, 8)}`}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-2">
        {status === 'initializing' && <Loader message="Initializing Mol* plugin..." />}
        {status === 'loading-structure' && <Loader message="Loading structure into viewer..." />}
        <div
          ref={containerRef}
          className="w-full h-[500px] rounded-lg bg-black overflow-hidden border border-gray-700 relative"
        />
      </div>

      {resultData.analysis_type === 'static' && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <label className="block text-sm text-gray-300 mb-1">State Score Threshold</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={staticThreshold}
            onChange={(e) => setStaticThreshold(Number(e.target.value))}
            className="w-full"
          />
          <p className="text-sm text-gray-400 mt-1">{staticThreshold.toFixed(2)}</p>
        </div>
      )}

      {resultData.analysis_type === 'qubo' && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 space-y-4">
          <div>
            <label className="block text-sm text-gray-300 mb-1">Active solutions</label>
            <select
              value={quboSolutionIdx.active}
              onChange={(e) => setQuboSolutionIdx((prev) => ({ ...prev, active: Number(e.target.value) }))}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
            >
              {(resultData.results?.qubo_active?.solutions || []).map((sol, idx) => (
                <option key={`act-${idx}`} value={idx}>
                  Solution {idx + 1} • E={sol.energy?.toFixed?.(2) ?? sol.energy}
                  {sol.union_coverage !== undefined ? ` • cov=${sol.union_coverage.toFixed?.(1)}` : ''}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-gray-300 mb-1">Inactive solutions</label>
            <select
              value={quboSolutionIdx.inactive}
              onChange={(e) => setQuboSolutionIdx((prev) => ({ ...prev, inactive: Number(e.target.value) }))}
              className="w-full bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
            >
              {(resultData.results?.qubo_inactive?.solutions || []).map((sol, idx) => (
                <option key={`inact-${idx}`} value={idx}>
                  Solution {idx + 1} • E={sol.energy?.toFixed?.(2) ?? sol.energy}
                  {sol.union_coverage !== undefined ? ` • cov=${sol.union_coverage.toFixed?.(1)}` : ''}
                </option>
              ))}
            </select>
          </div>
          <p className="text-xs text-gray-400">
            Load a structure above, then choose a solution. Active solutions highlight when the active state is loaded,
            and inactive solutions when the inactive state is loaded.
          </p>
        </div>
      )}

    </div>
  );
}
