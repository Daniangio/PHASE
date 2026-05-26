import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui/index';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { StateTransforms } from 'molstar/lib/mol-plugin-state/transforms';
import 'molstar/build/viewer/molstar.css';

function ext(name = '') {
  const clean = String(name || '').split('?')[0].toLowerCase();
  const m = clean.match(/\.([a-z0-9]+)$/);
  return m ? m[1] : '';
}

function inferTopology(fileName = '') {
  const e = ext(fileName);
  if (e === 'cif' || e === 'mmcif') return { format: 'mmcif', isBinary: false };
  if (e === 'bcif') return { format: 'bcif', isBinary: true };
  if (e === 'gro') return { format: 'gro', isBinary: false };
  if (e === 'pdb' || e === 'ent') return { format: 'pdb', isBinary: false };
  return { format: 'pdb', isBinary: false };
}

function inferCoordinates(fileName = '') {
  const e = ext(fileName);
  if (e === 'dcd') return 'dcd';
  if (e === 'xtc') return 'xtc';
  if (e === 'trr') return 'trr';
  if (e === 'nc' || e === 'nctraj') return 'nctraj';
  return 'xtc';
}

function isStructureFile(fileName = '') {
  return ['pdb', 'ent', 'cif', 'mmcif', 'bcif', 'gro'].includes(ext(fileName));
}

const MolstarTrajectoryViewer = forwardRef(function MolstarTrajectoryViewer(
  { height = 760, onStatusChange, onError },
  ref
) {
  const containerRef = useRef(null);
  const pluginRef = useRef(null);
  const [status, setStatus] = useState('initializing');

  const setViewerStatus = useCallback((next) => {
    setStatus(next);
    onStatusChange?.(next);
  }, [onStatusChange]);

  useEffect(() => {
    let disposed = false;
    const init = async () => {
      if (!containerRef.current) return;
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
        setViewerStatus('ready');
      } catch (err) {
        console.error('Mol* trajectory viewer initialization failed', err);
        onError?.(err.message || 'Failed to initialize Mol*.');
        setViewerStatus('error');
      }
    };
    init();
    return () => {
      disposed = true;
      if (pluginRef.current) pluginRef.current.dispose?.();
      pluginRef.current = null;
    };
  }, [onError, setViewerStatus]);

  const loadStructure = useCallback(async ({ structureUrl, structureData, structureName }) => {
    const plugin = pluginRef.current;
    if (!plugin) throw new Error('Mol* is not ready yet.');
    const topology = inferTopology(structureName);
    setViewerStatus('loading');
    try {
      await plugin.clear();
      await plugin.dataTransaction(async () => {
        const data = structureData !== undefined
          ? await plugin.builders.data.rawData({ data: structureData, label: structureName })
          : await plugin.builders.data.download({ url: structureUrl, isBinary: topology.isBinary, label: structureName }, { state: { isGhost: true } });
        const trajectory = await plugin.builders.structure.parseTrajectory(data, topology.format);
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
      });
      setViewerStatus('ready');
    } catch (err) {
      console.error('Mol* structure load failed', err);
      setViewerStatus('ready');
      throw err;
    }
  }, [setViewerStatus]);

  const loadTrajectory = useCallback(async ({ topologyUrl, topologyData, topologyName, coordinatesUrl, coordinatesData, coordinatesName }) => {
      const plugin = pluginRef.current;
      if (!plugin) throw new Error('Mol* is not ready yet.');
      if (isStructureFile(coordinatesName)) {
        return loadStructure({
          structureUrl: coordinatesUrl || topologyUrl,
          structureData: coordinatesData !== undefined ? coordinatesData : topologyData,
          structureName: coordinatesName || topologyName,
        });
      }
      const topology = inferTopology(topologyName);
      const coordinatesFormat = inferCoordinates(coordinatesName);
      setViewerStatus('loading');
      try {
        await plugin.clear();
        await plugin.dataTransaction(async () => {
          const modelData = topologyData !== undefined
            ? await plugin.builders.data.rawData({ data: topologyData, label: topologyName })
            : await plugin.builders.data.download({ url: topologyUrl, isBinary: topology.isBinary, label: topologyName }, { state: { isGhost: true } });
          const modelTrajectory = await plugin.builders.structure.parseTrajectory(modelData, topology.format);
          const model = await plugin.builders.structure.createModel(modelTrajectory);

          const coordData = coordinatesData !== undefined
            ? await plugin.builders.data.rawData({ data: coordinatesData, label: coordinatesName })
            : await plugin.builders.data.download({ url: coordinatesUrl, isBinary: true, label: coordinatesName }, { state: { isGhost: true } });
          const provider = plugin.dataFormats.get(coordinatesFormat);
          if (!provider) throw new Error(`Unsupported coordinate format '${coordinatesFormat}'.`);
          const coords = await provider.parse(plugin, coordData);
          const trajectory = await plugin.build().toRoot()
            .apply(StateTransforms.Model.TrajectoryFromModelAndCoordinates, {
              modelRef: model.ref,
              coordinatesRef: coords.ref,
            }, { dependsOn: [model.ref, coords.ref] })
            .commit();
          await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
        });
        setViewerStatus('ready');
      } catch (err) {
        console.error('Mol* raw trajectory load failed', err);
        setViewerStatus('ready');
        throw err;
      }
  }, [loadStructure, setViewerStatus]);

  useImperativeHandle(ref, () => ({
    loadStructure,
    loadTrajectory,
  }), [loadStructure, loadTrajectory]);

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900/40 p-3 min-h-0">
      <div
        className="rounded-md border border-gray-800 bg-black/20 overflow-hidden relative"
        style={{ height }}
      >
        {status === 'initializing' && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/60 text-sm text-gray-300">
            Initializing Mol*...
          </div>
        )}
        {status === 'loading' && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50 text-sm text-gray-300">
            Loading trajectory...
          </div>
        )}
        <div ref={containerRef} className="w-full h-full relative" />
      </div>
    </section>
  );
});

export default MolstarTrajectoryViewer;
