import React from 'react';

function clamp01(x) {
  if (!Number.isFinite(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function normRule(rule) {
  const aMin = clamp01(Number(rule?.aMin));
  const aMax = clamp01(Number(rule?.aMax));
  const bMin = clamp01(Number(rule?.bMin));
  const bMax = clamp01(Number(rule?.bMax));
  return {
    aMin: Math.min(aMin, aMax),
    aMax: Math.max(aMin, aMax),
    bMin: Math.min(bMin, bMax),
    bMax: Math.max(bMin, bMax),
  };
}

export function passesAnyJsFilter(dA, dB, rules) {
  if (!Number.isFinite(dA) || !Number.isFinite(dB)) return false;
  const arr = Array.isArray(rules) ? rules : [];
  if (!arr.length) return true;
  for (const raw of arr) {
    const r = normRule(raw);
    if (dA >= r.aMin && dA <= r.aMax && dB >= r.bMin && dB <= r.bMax) return true;
  }
  return false;
}

function RangePair({ label, minValue, maxValue, onChange }) {
  const minV = clamp01(Number(minValue));
  const maxV = clamp01(Number(maxValue));
  const lo = Math.min(minV, maxV);
  const hi = Math.max(minV, maxV);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span>{label}</span>
        <span className="font-mono text-gray-300">
          {lo.toFixed(2)} - {hi.toFixed(2)}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={lo}
          onChange={(e) => {
            const v = clamp01(Number(e.target.value));
            onChange(Math.min(v, hi), hi);
          }}
          className="w-full"
        />
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={hi}
          onChange={(e) => {
            const v = clamp01(Number(e.target.value));
            onChange(lo, Math.max(v, lo));
          }}
          className="w-full"
        />
      </div>
    </div>
  );
}

export default function JsRangeFilterBuilder({ rules, onChange, className = '' }) {
  const safeRules = (Array.isArray(rules) && rules.length ? rules : [{ aMin: 0, aMax: 1, bMin: 0, bMax: 1 }]).map(normRule);

  const updateRule = (idx, patch) => {
    const next = safeRules.map((r, i) => (i === idx ? normRule({ ...r, ...patch }) : r));
    onChange(next);
  };

  const removeRule = (idx) => {
    if (safeRules.length <= 1) return;
    const next = safeRules.filter((_, i) => i !== idx);
    onChange(next.length ? next : [{ aMin: 0, aMax: 1, bMin: 0, bMax: 1 }]);
  };

  const addRule = () => {
    onChange([...safeRules, { aMin: 0, aMax: 1, bMin: 0, bMax: 1 }]);
  };

  return (
    <div className={`rounded-md border border-gray-800 bg-gray-950/30 p-3 space-y-3 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="text-xs text-gray-300">Display filters (OR-combined)</div>
        <button
          type="button"
          onClick={addRule}
          className="text-xs px-2 py-1 rounded-md border border-gray-700 text-gray-200 hover:border-gray-500"
        >
          + Add OR rule
        </button>
      </div>
      <div className="space-y-2">
        {safeRules.map((rule, idx) => (
          <div key={`js-rule-${idx}`} className="rounded-md border border-gray-800 p-2 space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-xs text-gray-400">Rule {idx + 1}</div>
              {safeRules.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeRule(idx)}
                  className="text-xs px-2 py-1 rounded-md border border-gray-700 text-gray-300 hover:border-red-500 hover:text-red-300"
                >
                  Remove
                </button>
              )}
            </div>
            <RangePair
              label="JS(A)"
              minValue={rule.aMin}
              maxValue={rule.aMax}
              onChange={(aMin, aMax) => updateRule(idx, { aMin, aMax })}
            />
            <RangePair
              label="JS(B)"
              minValue={rule.bMin}
              maxValue={rule.bMax}
              onChange={(bMin, bMax) => updateRule(idx, { bMin, bMax })}
            />
          </div>
        ))}
      </div>
      <div className="text-[11px] text-gray-500">
        A point/residue passes if it matches at least one rule:
        <span className="font-mono"> (A in [min,max] and B in [min,max]) OR ...</span>
      </div>
    </div>
  );
}

