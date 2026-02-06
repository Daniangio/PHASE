import { useEffect, useState } from 'react';
import { X } from 'lucide-react';

export default function HelpDrawer({ open, title, docPath, onClose }) {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!open || !docPath) return;
      setLoading(true);
      setError('');
      try {
        const res = await fetch(docPath, { cache: 'no-cache' });
        if (!res.ok) throw new Error(`Failed to load doc (${res.status})`);
        const t = await res.text();
        if (!cancelled) setText(t || '');
      } catch (err) {
        if (!cancelled) setError(err.message || 'Failed to load documentation.');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => {
      cancelled = true;
    };
  }, [open, docPath]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      <button
        type="button"
        className="absolute inset-0 bg-black/60"
        aria-label="Close help"
        onClick={onClose}
      />
      <div className="absolute right-0 top-0 h-full w-full max-w-[720px] bg-gray-950 border-l border-gray-800 shadow-xl flex flex-col">
        <div className="flex items-center justify-between gap-3 px-4 py-3 border-b border-gray-800">
          <div className="min-w-0">
            <p className="text-sm font-semibold text-white truncate">{title || 'Help'}</p>
            {docPath && <p className="text-[11px] text-gray-500 truncate">{docPath}</p>}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="inline-flex items-center gap-2 px-2 py-2 rounded-md border border-gray-800 text-gray-200 hover:border-gray-600"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="flex-1 overflow-auto p-4">
          {loading && <p className="text-sm text-gray-400">Loading…</p>}
          {error && <p className="text-sm text-red-300">{error}</p>}
          {!loading && !error && (
            <pre className="whitespace-pre-wrap text-[12px] leading-relaxed text-gray-200">{text}</pre>
          )}
        </div>
      </div>
    </div>
  );
}

