import { useEffect, useRef, useState } from 'react';
import { Pencil, X } from 'lucide-react';
import ErrorMessage from '../common/ErrorMessage';
import IconButton from '../common/IconButton';
import { formatClusterAlgorithm, getClusterDisplayName } from './systemDetailUtils';

const DOC_FILES = {
  metastable_states: '/docs/metastable_states.md',
  vamp_tica: '/docs/vamp_tica.md',
  msm: '/docs/msm.md',
  pcca: '/docs/pcca.md',
  potts_overview: '/docs/potts_overview.md',
  potts_model: '/docs/potts_model.md',
  potts_pmi_plm: '/docs/potts_pmi_plm.md',
  potts_gibbs: '/docs/potts_gibbs.md',
  potts_sa_qubo: '/docs/potts_sa_qubo.md',
  potts_beta_eff: '/docs/potts_beta_eff.md',
};

function parseMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  const blocks = [];
  let title = 'Documentation';
  let paragraph = [];
  let list = null;
  let listType = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push({ type: 'p', text: paragraph.join(' ') });
      paragraph = [];
    }
  };

  const flushList = () => {
    if (list && list.length) {
      blocks.push({ type: listType, items: list });
    }
    list = null;
    listType = null;
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = headingMatch[1].length;
      const text = headingMatch[2].trim();
      if (level === 1 && title === 'Documentation') {
        title = text;
      } else {
        blocks.push({ type: `h${level}`, text });
      }
      continue;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.*)$/);
    const olMatch = trimmed.match(/^\d+[.)]\s+(.*)$/);
    if (ulMatch || olMatch) {
      flushParagraph();
      const nextType = olMatch ? 'ol' : 'ul';
      if (listType && listType !== nextType) {
        flushList();
      }
      listType = nextType;
      if (!list) list = [];
      list.push((ulMatch ? ulMatch[1] : olMatch[1]).trim());
      continue;
    }

    if (listType) {
      flushList();
    }
    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();

  return { title, blocks };
}

function renderInline(text, onNavigate) {
  const parts = [];
  const regex = /\[([^\]]+)\]\(([^)]+)\)/g;
  let lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', value: text.slice(lastIndex, match.index) });
    }
    parts.push({ type: 'link', label: match[1], href: match[2] });
    lastIndex = regex.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push({ type: 'text', value: text.slice(lastIndex) });
  }

  return parts.map((part, index) => {
    if (part.type === 'text') {
      const segments = part.value.split('`');
      return (
        <span key={`text-${index}`}>
          {segments.map((segment, segIdx) =>
            segIdx % 2 === 1 ? (
              <code key={`code-${index}-${segIdx}`} className="rounded bg-gray-800 px-1 text-xs text-gray-100">
                {segment}
              </code>
            ) : (
              <span key={`seg-${index}-${segIdx}`}>{segment}</span>
            )
          )}
        </span>
      );
    }
    if (part.href.startsWith('doc:')) {
      const target = part.href.replace('doc:', '').trim();
      return (
        <button
          key={`doc-${index}`}
          type="button"
          onClick={() => onNavigate(target)}
          className="text-cyan-300 hover:text-cyan-200 underline underline-offset-2"
        >
          {part.label}
        </button>
      );
    }
    return (
      <a
        key={`link-${index}`}
        href={part.href}
        target="_blank"
        rel="noreferrer"
        className="text-cyan-300 hover:text-cyan-200 underline underline-offset-2"
      >
        {part.label}
      </a>
    );
  });
}

function renderDocBlock(block, idx, onNavigate) {
  if (block.type === 'p') {
    return (
      <p key={`p-${idx}`} className="text-gray-300 leading-relaxed">
        {renderInline(block.text, onNavigate)}
      </p>
    );
  }
  if (block.type === 'ul' || block.type === 'ol') {
    const Tag = block.type === 'ol' ? 'ol' : 'ul';
    return (
      <Tag
        key={`${block.type}-${idx}`}
        className={`pl-5 space-y-1 text-gray-300 ${block.type === 'ol' ? 'list-decimal' : 'list-disc'}`}
      >
        {block.items.map((item, itemIdx) => (
          <li key={`${block.type}-${idx}-${itemIdx}`}>{renderInline(item, onNavigate)}</li>
        ))}
      </Tag>
    );
  }
  if (block.type === 'h2' || block.type === 'h3') {
    return (
      <h4 key={`${block.type}-${idx}`} className="text-base font-semibold text-white">
        {block.text}
      </h4>
    );
  }
  return null;
}

export function DocOverlay({ docId, onClose, onNavigate }) {
  const cacheRef = useRef({});
  const [docState, setDocState] = useState({
    title: 'Documentation',
    blocks: [],
    loading: true,
    error: null,
  });

  useEffect(() => {
    let isMounted = true;
    const targetId = DOC_FILES[docId] ? docId : 'metastable_states';
    const cached = cacheRef.current[targetId];
    if (cached) {
      setDocState({ ...cached, loading: false, error: null });
      return () => {};
    }
    setDocState((prev) => ({ ...prev, loading: true, error: null }));
    fetch(DOC_FILES[targetId])
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load ${targetId} documentation.`);
        return res.text();
      })
      .then((text) => {
        const parsed = parseMarkdown(text);
        if (!isMounted) return;
        cacheRef.current[targetId] = parsed;
        setDocState({ ...parsed, loading: false, error: null });
      })
      .catch((err) => {
        if (!isMounted) return;
        setDocState((prev) => ({
          ...prev,
          loading: false,
          error: err.message || 'Failed to load documentation.',
        }));
      });
    return () => {
      isMounted = false;
    };
  }, [docId]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
      <div className="w-full max-w-3xl bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-lg font-semibold text-white">{docState.title}</h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-5 max-h-[75vh] overflow-y-auto space-y-4 text-sm text-gray-200">
          {docState.loading && <p className="text-gray-400">Loading documentation...</p>}
          {docState.error && <ErrorMessage message={docState.error} />}
          {!docState.loading &&
            !docState.error &&
            docState.blocks.map((block, idx) => renderDocBlock(block, idx, onNavigate))}
        </div>
      </div>
    </div>
  );
}

export function ClusterBuildOverlay({
  metastableStates,
  selectedMetastableIds,
  onToggleMetastable,
  clusterName,
  setClusterName,
  clusterMode,
  setClusterMode,
  uploadClusterName,
  setUploadClusterName,
  uploadClusterFile,
  setUploadClusterFile,
  uploadClusterStateIds,
  uploadStateOptions,
  onToggleUploadState,
  uploadClusterError,
  uploadClusterLoading,
  densityZMode,
  setDensityZMode,
  densityZValue,
  setDensityZValue,
  densityMaxk,
  setDensityMaxk,
  maxClusterFrames,
  setMaxClusterFrames,
  clusterError,
  clusterLoading,
  onClose,
  onSubmit,
  onUpload,
}) {
  const hasMetastable = metastableStates.length > 0;
  const isUpload = clusterMode === 'upload';
  const canSubmit = !isUpload && selectedMetastableIds.length > 0 && !clusterLoading;
  const canUpload =
    isUpload && uploadClusterFile && uploadClusterStateIds.length > 0 && !uploadClusterLoading;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
      <div className="w-full max-w-5xl bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-lg font-semibold text-white">New Cluster NPZ</h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-5 max-h-[75vh] overflow-y-auto space-y-4 text-sm text-gray-200">
          <div className="flex items-center gap-2 text-xs">
            <button
              type="button"
              onClick={() => setClusterMode('generate')}
              className={`px-3 py-1 rounded-full border ${
                !isUpload
                  ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10'
                  : 'border-gray-700 text-gray-400 hover:border-gray-500'
              }`}
            >
              Generate
            </button>
            <button
              type="button"
              onClick={() => setClusterMode('upload')}
              className={`px-3 py-1 rounded-full border ${
                isUpload
                  ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10'
                  : 'border-gray-700 text-gray-400 hover:border-gray-500'
              }`}
            >
              Upload local NPZ
            </button>
          </div>

          {!isUpload && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Cluster name</label>
                <input
                  type="text"
                  value={clusterName}
                  onChange={(e) => setClusterName(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white"
                />
              </div>

              <div className="flex flex-wrap gap-2 text-xs">
                {metastableStates.map((meta) => {
                  const key = meta.state_id || meta.metastable_id || `${meta.macro_state}-${meta.metastable_index}`;
                  const active = selectedMetastableIds.includes(key);
                  const label = meta.name || meta.default_name || meta.macro_state || key;
                  const kind = meta.kind || (meta.metastable_id ? 'metastable' : 'macro');
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => onToggleMetastable(key)}
                      className={`px-3 py-1 rounded-full border ${
                        active
                          ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10'
                          : 'border-gray-700 text-gray-400 hover:border-gray-500'
                      }`}
                    >
                      {kind === 'metastable' ? '[Meta] ' : '[Macro] '}
                      {label}
                    </button>
                  );
                })}
                {!hasMetastable && <p className="text-xs text-gray-400">No states available.</p>}
              </div>

              <div className="grid md:grid-cols-3 gap-3">
                <label className="space-y-1">
                  <span className="block text-gray-400">Max frames</span>
                  <input
                    type="number"
                    min={0}
                    value={maxClusterFrames}
                    onChange={(e) => setMaxClusterFrames(Math.max(0, Number(e.target.value) || 0))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  />
                </label>
              </div>

              <div className="grid md:grid-cols-3 gap-3">
                <label className="space-y-1">
                  <span className="block text-gray-400">Z threshold</span>
                  <select
                    value={densityZMode}
                    onChange={(e) => setDensityZMode(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  >
                    <option value="auto">Auto</option>
                    <option value="manual">Manual</option>
                  </select>
                </label>
                {densityZMode === 'manual' && (
                  <label className="space-y-1">
                    <span className="block text-gray-400">Z value</span>
                    <input
                      type="number"
                      step="0.05"
                      min={0}
                      value={densityZValue}
                      onChange={(e) => setDensityZValue(Math.max(0.1, Number(e.target.value) || 0))}
                      className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                    />
                  </label>
                )}
                <label className="space-y-1">
                  <span className="block text-gray-400">Max k</span>
                  <input
                    type="number"
                    min={5}
                    value={densityMaxk}
                    onChange={(e) => setDensityMaxk(Math.max(5, Number(e.target.value) || 5))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white"
                  />
                </label>
              </div>
              <div className="flex items-center">
                <p className="text-gray-300">
                  Selected: <span className="text-white font-semibold">{selectedMetastableIds.length}</span> /{' '}
                  {metastableStates.length}
                </p>
              </div>
            </div>
          )}

          {isUpload && (
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Cluster name (optional)</label>
                <input
                  type="text"
                  value={uploadClusterName}
                  onChange={(e) => setUploadClusterName(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">Upload cluster NPZ</label>
                <input
                  type="file"
                  accept=".npz"
                  onChange={(e) => setUploadClusterFile(e.target.files?.[0] || null)}
                  className="w-full text-sm text-gray-300"
                />
              </div>
              <div className="space-y-2">
                <p className="text-xs text-gray-400">
                  Select the macro-states used for local clustering (order inferred from NPZ metadata).
                </p>
                <div className="flex flex-wrap gap-2 text-xs">
                  {uploadStateOptions.map((opt) => {
                    const active = uploadClusterStateIds.includes(opt.id);
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => onToggleUploadState(opt.id)}
                        className={`px-3 py-1 rounded-full border ${
                          active
                            ? 'border-cyan-400 text-cyan-200 bg-cyan-500/10'
                            : 'border-gray-700 text-gray-400 hover:border-gray-500'
                        }`}
                      >
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
                {uploadClusterStateIds.length > 0 && (
                  <div className="space-y-1 text-xs text-gray-300">
                    {uploadClusterStateIds.map((sid) => {
                      const label = uploadStateOptions.find((opt) => opt.id === sid)?.label || sid;
                      return (
                        <div key={sid} className="flex items-center gap-2">
                          <span className="w-5 text-gray-500">•</span>
                          <span className="flex-1">{label}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
              {uploadClusterError && <ErrorMessage message={uploadClusterError} />}
            </div>
          )}

          {!isUpload && (
            <>
              <p className="text-gray-400 text-xs">
                NPZ includes merged cluster vectors, contact map edge_index (pyg format), and metadata JSON.
              </p>
              {clusterError && <ErrorMessage message={clusterError} />}
            </>
          )}
        </div>
        <div className="flex items-center justify-end gap-2 border-t border-gray-800 px-4 py-3">
          <button
            type="button"
            onClick={onClose}
            className="text-xs px-3 py-2 rounded-md border border-gray-600 text-gray-200 hover:bg-gray-700/60"
          >
            Cancel
          </button>
          {!isUpload ? (
            <button
              type="button"
              onClick={onSubmit}
              disabled={!canSubmit}
              className="text-xs px-3 py-2 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50"
            >
              {clusterLoading ? 'Generating...' : 'Run clustering'}
            </button>
          ) : (
            <button
              type="button"
              onClick={onUpload}
              disabled={!canUpload}
              className="text-xs px-3 py-2 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10 disabled:opacity-50"
            >
              {uploadClusterLoading ? 'Uploading...' : 'Upload cluster'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function ClusterDetailOverlay({
  cluster,
  analysisMode,
  onClose,
  onRename,
  onDownload,
  onDownloadBackmapping,
  backmappingProgress,
  backmappingJob,
  onDelete,
  onVisualize,
}) {
  const [name, setName] = useState(getClusterDisplayName(cluster));
  const [isSaving, setIsSaving] = useState(false);
  const stateLabel = analysisMode === 'macro' ? 'States' : 'Metastable';

  useEffect(() => {
    setName(getClusterDisplayName(cluster));
  }, [cluster]);

  const handleSave = async () => {
    if (!name.trim()) return;
    setIsSaving(true);
    try {
      await onRename(cluster.cluster_id, name.trim());
    } finally {
      setIsSaving(false);
    }
  };

  const algoSummary = formatClusterAlgorithm(cluster);
  const filename = cluster.path?.split('/').pop();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
      <div className="w-full max-w-2xl bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-lg font-semibold text-white">Cluster NPZ Details</h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-5 space-y-4 text-sm text-gray-200">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Cluster name</label>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="flex-1 bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-white"
              />
              <button
                type="button"
                onClick={handleSave}
                disabled={isSaving || !name.trim()}
                className="text-xs px-3 py-2 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10 disabled:opacity-50"
              >
                {isSaving ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>

          <div className="space-y-1 text-xs text-gray-400">
            <p>
              <span className="text-gray-300 font-semibold">ID:</span> {cluster.cluster_id}
            </p>
            <p>
              <span className="text-gray-300 font-semibold">Algorithm:</span> {algoSummary || '—'}
            </p>
            <p>
              <span className="text-gray-300 font-semibold">{stateLabel}:</span>{' '}
              {Array.isArray(cluster.state_ids || cluster.metastable_ids)
                ? (cluster.state_ids || cluster.metastable_ids).join(', ')
                : '—'}
            </p>
            <p>
              <span className="text-gray-300 font-semibold">Max frames:</span> {cluster.max_cluster_frames ?? 'all'}
            </p>
            <p>
              <span className="text-gray-300 font-semibold">Generated:</span> {cluster.generated_at || '—'}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center justify-end gap-2 border-t border-gray-800 px-4 py-3">
          <button
            type="button"
            onClick={() => onDownload(cluster.cluster_id, filename)}
            className="text-xs px-3 py-2 rounded-md border border-emerald-500 text-emerald-300 hover:bg-emerald-500/10"
          >
            Download
          </button>
          <button
            type="button"
            onClick={() => onDownloadBackmapping?.(cluster.cluster_id, `backmapping_${cluster.cluster_id}.npz`)}
            disabled={
              (backmappingJob && backmappingJob.status && backmappingJob.status !== 'failed' && backmappingJob.status !== 'finished') ||
              (backmappingProgress !== null && backmappingProgress !== undefined)
            }
            className="text-xs px-3 py-2 rounded-md border border-sky-500 text-sky-300 hover:bg-sky-500/10 disabled:opacity-60"
          >
            {backmappingJob?.status === 'finished'
              ? 'Download backmapping NPZ'
              : backmappingJob?.status && backmappingJob.status !== 'failed'
              ? `Preparing... ${backmappingJob?.meta?.progress ?? 0}%`
              : backmappingProgress !== null && backmappingProgress !== undefined
              ? `Downloading... ${backmappingProgress ?? 0}%`
              : 'Backmapping NPZ'}
          </button>
          <button
            type="button"
            onClick={() => onVisualize(cluster.cluster_id)}
            className="text-xs px-3 py-2 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10"
          >
            Visualize
          </button>
          <button
            type="button"
            onClick={() => onDelete(cluster.cluster_id)}
            className="text-xs px-3 py-2 rounded-md border border-red-500 text-red-300 hover:bg-red-500/10"
          >
            Delete
          </button>
        </div>
        {(backmappingJob?.status && backmappingJob.status !== 'finished' && backmappingJob.status !== 'failed') ||
        (backmappingProgress !== null && backmappingProgress !== undefined) ? (
          <div className="px-4 pb-4">
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-sky-500 transition-all duration-200"
                style={{
                  width: `${Math.max(
                    2,
                    backmappingJob?.status && backmappingJob.status !== 'finished' && backmappingJob.status !== 'failed'
                      ? backmappingJob?.meta?.progress ?? 0
                      : backmappingProgress || 0
                  )}%`,
                }}
              />
            </div>
            <p className="text-[11px] text-gray-400 mt-1">
              {backmappingProgress !== null && backmappingProgress !== undefined
                ? 'Downloading backmapping NPZ…'
                : 'Building backmapping NPZ…'}
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function InfoOverlay({ state, type, onClose, onRenameMacro, onRenameMeta, onDownloadMacroNpz }) {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(state.name || state.default_name || '');
  const isMetastable = type === 'meta';

  const handleSave = async (nextName = name) => {
    const trimmed = (nextName || '').trim();
    if (!trimmed) return;
    if (isMetastable) {
      await onRenameMeta(state.metastable_id, trimmed);
    } else {
      await onRenameMacro(state.state_id, trimmed);
    }
    setName(trimmed);
    setIsEditing(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
      <div className="w-full max-w-xl bg-gray-900 border border-gray-700 rounded-lg shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-4 py-3">
          <h3 className="text-lg font-semibold text-white">
            {isMetastable ? 'Metastable State Info' : 'Macro-state Info'}
          </h3>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200"
            aria-label="Close"
          >
            <X className="h-5 w-5" />
          </button>
        </div>
        <div className="p-5 max-h-[75vh] overflow-y-auto space-y-4 text-sm text-gray-200">
          <div className="flex items-center gap-2">
            <h4 className="text-base font-semibold text-white">Name:</h4>
            <RenameableText
              value={name}
              isEditing={isEditing}
              setIsEditing={setIsEditing}
              handleSave={handleSave}
            />
          </div>

          <p>
            <span className="font-semibold">ID:</span>{' '}
            {isMetastable ? state.metastable_id : state.state_id}
          </p>
          {!isMetastable && state.pdb_file && (
            <p>
              <span className="font-semibold">PDB File:</span> {state.pdb_file}
            </p>
          )}
          {!isMetastable && state.trajectory_file && (
            <p>
              <span className="font-semibold">Trajectory File:</span> {state.trajectory_file}
            </p>
          )}
          {isMetastable && state.macro_state && (
            <p>
              <span className="font-semibold">Macro State:</span> {state.macro_state}
            </p>
          )}
          {isMetastable && (
            <p>
              <span className="font-semibold">Metastable Index:</span> {state.metastable_index}
            </p>
          )}
          <p>
            <span className="font-semibold">Number of Frames:</span> {state.n_frames ?? 'N/A'}
          </p>
          <p>
            <span className="font-semibold">Stride:</span> {state.stride ?? 'N/A'}
          </p>
          <p>
            <span className="font-semibold">Descriptors Ready:</span>{' '}
            {state.descriptor_file ? 'Yes' : 'No'}
          </p>
          {!isMetastable && state.descriptor_file && (
            <button
              type="button"
              onClick={() => onDownloadMacroNpz?.(state.state_id, state.name)}
              className="text-xs px-3 py-2 rounded-md border border-cyan-500 text-cyan-300 hover:bg-cyan-500/10"
            >
              Download descriptors (NPZ)
            </button>
          )}
          {isMetastable && state.representative_pdb && (
            <p>
              <span className="font-semibold">Representative PDB:</span>{' '}
              <a
                href={state.representative_pdb}
                target="_blank"
                rel="noreferrer"
                className="text-cyan-400 hover:underline break-all"
              >
                {state.representative_pdb}
              </a>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function RenameableText({ value, isEditing, setIsEditing, handleSave }) {
  const [currentValue, setCurrentValue] = useState(value);
  const inputRef = useRef(null);

  useEffect(() => {
    if (isEditing) {
      inputRef.current?.focus();
    }
  }, [isEditing]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSave(currentValue);
    }
    if (e.key === 'Escape') {
      setIsEditing(false);
      setCurrentValue(value);
    }
  };

  return (
    <div className="flex items-center gap-2">
      {isEditing ? (
        <input
          ref={inputRef}
          type="text"
          value={currentValue}
          onChange={(e) => setCurrentValue(e.target.value)}
          onBlur={() => handleSave(currentValue)}
          onKeyDown={handleKeyDown}
          className="bg-gray-800 border border-gray-700 rounded-md px-2 py-1 text-white text-sm"
        />
      ) : (
        <span className="text-gray-200">{value}</span>
      )}
      <IconButton
        icon={Pencil}
        label={isEditing ? 'Save name' : 'Edit name'}
        onClick={() => (isEditing ? handleSave(currentValue) : setIsEditing(true))}
        className="text-gray-400 hover:text-cyan-300"
        iconClassName="h-4 w-4"
      />
    </div>
  );
}
