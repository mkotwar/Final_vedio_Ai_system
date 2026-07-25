interface ConfidenceBadgeProps {
  value?: number | null
}

export default function ConfidenceBadge({ value }: ConfidenceBadgeProps) {
  if (value === undefined || value === null) {
    return <span className="badge badge--neutral">N/A</span>
  }

  const variant = value >= 0.9 ? 'success' : value >= 0.7 ? 'warning' : 'danger'
  return <span className={`badge badge--${variant}`}>{Math.round(value * 100)}%</span>
}
