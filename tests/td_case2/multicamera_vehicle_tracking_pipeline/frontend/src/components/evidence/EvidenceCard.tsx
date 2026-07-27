import type { MediaReference } from '../../types/media'
import EvidencePreview from './EvidencePreview'

interface EvidenceCardProps {
  media?: MediaReference | null
}

export default function EvidenceCard({ media }: EvidenceCardProps) {
  return (
    <article className="evidence-card">
      <div className="evidence-card__header">
        <div>
          <h4>{media?.media_type || 'Evidence'}</h4>
          <p className="table-subtext">{media?.availability || 'REFERENCE_ONLY'}</p>
        </div>
        <div className="evidence-card__badges">
          {media?.is_primary ? <span className="badge badge--success">PRIMARY</span> : null}
          <span className="badge badge--neutral">{media?.availability || 'REFERENCE_ONLY'}</span>
        </div>
      </div>

      <div className="evidence-card__preview">
        <EvidencePreview media={media} title={media?.media_type || 'Evidence image'} />
      </div>

      <dl className="meta-grid">
        <div>
          <dt>Frame</dt>
          <dd>{media?.frame_number ?? 'N/A'}</dd>
        </div>
        <div>
          <dt>Quality</dt>
          <dd>{typeof media?.quality_score === 'number' ? media.quality_score.toFixed(2) : 'N/A'}</dd>
        </div>
        <div>
          <dt>Selection rank</dt>
          <dd>{media?.selection_rank ?? 'N/A'}</dd>
        </div>
        <div>
          <dt>Size</dt>
          <dd>{media?.width && media?.height ? `${media.width} × ${media.height}` : 'N/A'}</dd>
        </div>
      </dl>
    </article>
  )
}
