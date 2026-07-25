import type { MediaReference } from '../../types/media'

interface EvidenceCardProps {
  media?: MediaReference | null
}

export default function EvidenceCard({ media }: EvidenceCardProps) {
  if (!media) {
    return (
      <article className="evidence-card">
        <h4>Evidence</h4>
        <p>No media reference available.</p>
      </article>
    )
  }

  return (
    <article className="evidence-card">
      <div className="evidence-card__header">
        <h4>{media.media_type || 'Reference only'}</h4>
        <span className="badge badge--neutral">REFERENCE ONLY</span>
      </div>
      <dl className="meta-grid">
        <div>
          <dt>Frame</dt>
          <dd>{media.frame_number ?? 'N/A'}</dd>
        </div>
        <div>
          <dt>Quality</dt>
          <dd>{media.quality_score ?? 'N/A'}</dd>
        </div>
        <div>
          <dt>Selection rank</dt>
          <dd>{media.selection_rank ?? 'N/A'}</dd>
        </div>
        <div>
          <dt>Reference</dt>
          <dd className="break-text">{media.storage_uri || 'Unavailable'}</dd>
        </div>
      </dl>
    </article>
  )
}
