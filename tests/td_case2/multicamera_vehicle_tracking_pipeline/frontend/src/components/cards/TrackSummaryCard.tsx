import type { TrackListItem } from '../../types/track'
import ConfidenceBadge from '../states/ConfidenceBadge'
import StatusBadge from '../states/StatusBadge'

interface TrackSummaryCardProps {
  track: TrackListItem
}

export default function TrackSummaryCard({ track }: TrackSummaryCardProps) {
  return (
    <article className="summary-card">
      <div className="summary-card__header">
        <div>
          <p className="summary-card__eyebrow">{track.camera_code || 'Camera'}</p>
          <h3>{track.track_uuid}</h3>
        </div>
        <StatusBadge value={track.lifecycle_state} />
      </div>
      <div className="summary-card__metrics">
        <div>
          <span>Class</span>
          <strong>{track.vehicle_class || 'N/A'}</strong>
        </div>
        <div>
          <span>Colour</span>
          <strong>{track.primary_colour || 'N/A'}</strong>
        </div>
        <div>
          <span>Plate</span>
          <strong>{track.canonical_plate || 'N/A'}</strong>
        </div>
        <div>
          <span>Confidence</span>
          <ConfidenceBadge value={track.best_detection_confidence} />
        </div>
      </div>
    </article>
  )
}
