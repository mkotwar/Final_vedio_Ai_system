interface StatusBadgeProps {
  value?: string | null
}

export default function StatusBadge({ value }: StatusBadgeProps) {
  const normalized = (value || 'UNKNOWN').toUpperCase()
  const variant = [
    'COMPLETED',
    'CONFIRMED',
    'ACTIVE',
    'CURRENT',
    'VERIFIED',
  ].includes(normalized)
    ? 'success'
    : ['FAILED', 'REJECTED', 'CANCELLED'].includes(normalized)
      ? 'danger'
      : ['RUNNING', 'POSSIBLE', 'REVIEW_REQUIRED', 'WARNING'].includes(normalized)
        ? 'warning'
        : 'neutral'

  return (
    <span className={`badge badge--${variant}`}>
      <span aria-hidden="true" className="badge__dot" />
      {normalized.replaceAll('_', ' ')}
    </span>
  )
}
