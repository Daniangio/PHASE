const CLUSTER_COLORS = [
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

export function clusterPieColor(clusterId) {
  const value = Number(clusterId);
  const index = Number.isFinite(value) ? Math.abs(Math.trunc(value)) : 0;
  return CLUSTER_COLORS[index % CLUSTER_COLORS.length];
}

function piePath(cx, cy, radius, start, end) {
  const x0 = cx + radius * Math.cos(start);
  const y0 = cy + radius * Math.sin(start);
  const x1 = cx + radius * Math.cos(end);
  const y1 = cy + radius * Math.sin(end);
  const large = end - start > Math.PI ? 1 : 0;
  return `M ${cx} ${cy} L ${x0} ${y0} A ${radius} ${radius} 0 ${large} 1 ${x1} ${y1} Z`;
}

export default function ClusterPieChart({ slices, size = 120, onClick = null }) {
  const radius = size * 0.45;
  const center = size / 2;
  const positiveSlices = (Array.isArray(slices) ? slices : []).filter((slice) => Number(slice?.value) > 0);
  let acc = -Math.PI / 2;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      <circle cx={center} cy={center} r={radius} fill="#111827" />
      {positiveSlices.map((slice, idx) => {
        const value = Number(slice.value) || 0;
        const color = slice.color || clusterPieColor(slice.clusterId ?? idx);
        const percentage = `${(100 * value).toFixed(2)}%`;
        const handleClick = onClick ? () => onClick(slice) : undefined;

        // A 2π SVG arc has coincident endpoints and renders as an empty path.
        // Use a circle for a single-cluster distribution instead.
        if (positiveSlices.length === 1 || value >= 1 - 1e-9) {
          return (
            <circle
              key={`${slice.label}:${idx}`}
              cx={center}
              cy={center}
              r={radius}
              fill={color}
              onClick={handleClick}
              className={onClick ? 'cursor-pointer' : undefined}
            >
              <title>{`${slice.tooltip || slice.label}: ${percentage}`}</title>
            </circle>
          );
        }

        const next = acc + value * Math.PI * 2;
        const path = piePath(center, center, radius, acc, next);
        acc = next;
        return (
          <path
            key={`${slice.label}:${idx}`}
            d={path}
            fill={color}
            onClick={handleClick}
            className={onClick ? 'cursor-pointer' : undefined}
          >
            <title>{`${slice.tooltip || slice.label}: ${percentage}`}</title>
          </path>
        );
      })}
      <circle cx={center} cy={center} r={radius * 0.35} fill="#0b1220" />
    </svg>
  );
}
