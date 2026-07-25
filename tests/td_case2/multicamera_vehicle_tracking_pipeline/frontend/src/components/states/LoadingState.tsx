interface LoadingStateProps {
  label?: string
}

export default function LoadingState({ label = 'Loading data...' }: LoadingStateProps) {
  return (
    <div className="state-card" role="status" aria-live="polite">
      <div className="loading-shimmer" />
      <p>{label}</p>
    </div>
  )
}
