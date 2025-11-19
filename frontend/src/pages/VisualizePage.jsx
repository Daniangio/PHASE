import { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import Loader from '../components/common/Loader';
import ErrorMessage from '../components/common/ErrorMessage';
import { fetchResult } from '../api/jobs';
import { downloadStructure } from '../api/projects';

const buildSelection = (keys, mapping) => {
  if (!keys || !mapping) return 'none';
  const residues = keys
    .map((key) => mapping[key])
    .filter(Boolean)
    .map((sel) => sel.replace(/resid /gi, ''))
    .join(' or ')
    .replace(/\s+/g, ' or ');
  return residues || 'none';
};

export default function VisualizePage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [resultData, setResultData] = useState(null);
  const [structureFile, setStructureFile] = useState(null);
  const [structureLoading, setStructureLoading] = useState(false);
  const [structureError, setStructureError] = useState(null);
  const stageRef = useRef(null);
  const [stageReady, setStageReady] = useState(false);
  const [stageKey, setStageKey] = useState(0);
  const componentRef = useRef(null);
  const [nglReady, setNglReady] = useState(() => typeof window !== 'undefined' && !!window.NGL);
  const [staticThreshold, setStaticThreshold] = useState(0.8);
  const [quboSolutionIndex, setQuboSolutionIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const nglViewportRef = useRef(null);

  useEffect(() => {
    const scriptId = 'ngl-script';
    if (window.NGL) {
      setNglReady(true);
      return;
    }
    if (!document.getElementById(scriptId)) {
      const script = document.createElement('script');
      script.id = scriptId;
      script.src = 'https://cdn.jsdelivr.net/npm/ngl/dist/ngl.js';
      script.async = true;
      script.onload = () => setNglReady(true);
      document.body.appendChild(script);
    } else {
      const existing = document.getElementById(scriptId);
      existing.addEventListener('load', () => setNglReady(true), { once: true });
    }
  }, []);

  useEffect(() => {
    const loadResult = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchResult(jobId);
        if (!data.residue_selections_mapping || !data.results) {
          throw new Error('Result is missing selections or outputs.');
        }
        if (!data.system_reference?.project_id || !data.system_reference?.system_id) {
          throw new Error('Result does not reference a stored system.');
        }
        setResultData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    loadResult();
  }, [jobId]);

  useEffect(() => {
    let cancelled = false;
    const fetchStructure = async () => {
      if (!resultData) return;
      const { project_id, system_id } = resultData.system_reference || {};
      if (!project_id || !system_id) return;
      setStructureLoading(true);
      setStructureError(null);
      setStructureFile(null);
      try {
        const blob = await downloadStructure(project_id, system_id, 'active');
        if (cancelled) return;
        const file = new File([blob], `active.pdb`, { type: 'chemical/x-pdb' });
        setStructureFile(file);
      } catch (err) {
        if (!cancelled) {
          setStructureError(err.message);
        }
      } finally {
        if (!cancelled) {
          setStructureLoading(false);
        }
      }
    };
    fetchStructure();
    return () => {
      cancelled = true;
    };
  }, [resultData]);

  useEffect(() => {
    if (!nglReady || !window.NGL || !nglViewportRef.current) return undefined;
    const stage = new window.NGL.Stage(nglViewportRef.current);
    stageRef.current = stage;
    setStageReady(true);
    setStageKey((key) => key + 1);
    return () => {
      setStageReady(false);
      if (stageRef.current) {
        try {
          stageRef.current.dispose();
        } catch (err) {
          console.warn('Failed to dispose NGL stage', err);
        }
        stageRef.current = null;
        componentRef.current = null;
      }
    };
  }, [nglReady]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!nglReady || !stageReady || !stage || !structureFile || !resultData) return;
    stage.removeAllComponents();
    componentRef.current = null;
    const ext = structureFile.name.split('.').pop();
    stage
      .loadFile(structureFile, { ext })
      .then((component) => {
        component.autoView();
        componentRef.current = component;
        renderHighlights(component, resultData, staticThreshold, quboSolutionIndex);
      })
      .catch((err) => {
        console.error('Failed to load structure', err);
        setStructureError(err.message || 'Failed to load structure.');
      });
  }, [structureFile, resultData, nglReady, stageReady]);

  useEffect(() => {
    const component = componentRef.current;
    if (!component || !resultData) return;
    renderHighlights(component, resultData, staticThreshold, quboSolutionIndex);
  }, [resultData, staticThreshold, quboSolutionIndex, stageReady, stageKey]);

function renderHighlights(component, resultData, staticThreshold, quboSolutionIndex) {
  if (!component || !resultData) return;
  const { analysis_type, results, residue_selections_mapping } = resultData;
  component.removeAllRepresentations();
  component.addRepresentation('cartoon', { color: '#555555', opacity: 0.3 });

  if (analysis_type === 'static') {
    const selected = Object.keys(results).filter(
      (key) => results[key]?.state_score >= staticThreshold
    );
    const selection = buildSelection(selected, residue_selections_mapping);
    if (selection !== 'none') {
      component.addRepresentation('ball+stick', { sele: selection, color: '#ef4444' });
    }
  } else if (analysis_type === 'qubo' && results.classification) {
    const solution = results.solutions?.[quboSolutionIndex];
    const selectedSet = new Set(solution?.residues || []);
    const classification = results.classification;
    const switches = [];
    const silentOps = [];
    const decoys = [];

    Object.keys(classification).forEach((key) => {
      const entry = classification[key];
      if (selectedSet.has(key)) {
        if (entry.jsd > (results.classification_threshold || 0.3)) {
          switches.push(key);
        } else {
          silentOps.push(key);
        }
      } else {
        decoys.push(key);
      }
    });

    const addRep = (keys, color, style = 'ball+stick') => {
      const selection = buildSelection(keys, residue_selections_mapping);
      if (selection !== 'none') {
        component.addRepresentation(style, { sele: selection, color });
      }
    };

    addRep(switches, '#ef4444');
    addRep(silentOps, '#f59e0b');
    addRep(decoys, '#9ca3af', 'licorice');
  }
}

  if (isLoading) return <Loader message="Preparing visualization..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!resultData) return null;

  return (
    <div className="space-y-4">
      <button onClick={() => navigate(`/results/${jobId}`)} className="text-cyan-400 hover:text-cyan-300 text-sm">
        ← Back to result
      </button>
      <h1 className="text-2xl font-bold text-white">Visualization: {jobId}</h1>

      {structureError && <ErrorMessage message={structureError} />}

      <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
        {structureLoading && <p className="text-sm text-gray-400 mb-2">Loading structure...</p>}
        <div ref={nglViewportRef} className="w-full h-[500px] rounded-lg bg-black" />
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
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <label className="block text-sm text-gray-300 mb-1">Solution</label>
          <select
            value={quboSolutionIndex}
            onChange={(e) => setQuboSolutionIndex(Number(e.target.value))}
            className="bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-white"
          >
            {resultData.results?.solutions?.map((solution, idx) => (
              <option key={solution.energy} value={idx}>
                Solution {idx + 1} – energy {solution.energy.toFixed(3)}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
