interface EmptyStateProps {
  title: string
  description: string
}

export default function EmptyState({ title, description }: EmptyStateProps) {
  return (
    <div className="state-card state-card--empty">
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  )
}
