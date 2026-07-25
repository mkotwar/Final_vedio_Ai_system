interface RetryButtonProps {
  onClick: () => void
  label?: string
}

export default function RetryButton({ onClick, label = 'Retry' }: RetryButtonProps) {
  return (
    <button className="button button--secondary" onClick={onClick} type="button">
      {label}
    </button>
  )
}
