import RetryButton from './RetryButton'

interface ErrorStateProps {
  title: string
  message: string
  onRetry?: () => void
}

export default function ErrorState({ title, message, onRetry }: ErrorStateProps) {
  return (
    <div className="state-card state-card--error" role="alert">
      <h3>{title}</h3>
      <p>{message}</p>
      {onRetry ? <RetryButton onClick={onRetry} /> : null}
    </div>
  )
}
