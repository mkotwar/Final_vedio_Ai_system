import type { GlobalVehicleListItem } from '../../types/globalVehicle'
import ConfidenceBadge from '../states/ConfidenceBadge'
import StatusBadge from '../states/StatusBadge'

interface GlobalVehicleCardProps {
  vehicle: GlobalVehicleListItem
}

export default function GlobalVehicleCard({ vehicle }: GlobalVehicleCardProps) {
  return (
    <article className="summary-card">
      <div className="summary-card__header">
        <div>
          <p className="summary-card__eyebrow">{vehicle.run_code || 'Run'}</p>
          <h3>{vehicle.global_vehicle_code}</h3>
        </div>
        <StatusBadge value={vehicle.status} />
      </div>
      <div className="summary-card__metrics">
        <div>
          <span>Plate</span>
          <strong>{vehicle.canonical_plate || 'N/A'}</strong>
        </div>
        <div>
          <span>Class</span>
          <strong>{vehicle.canonical_vehicle_class || 'N/A'}</strong>
        </div>
        <div>
          <span>Colour</span>
          <strong>{vehicle.canonical_colour || 'N/A'}</strong>
        </div>
        <div>
          <span>Confidence</span>
          <ConfidenceBadge value={vehicle.confidence} />
        </div>
      </div>
    </article>
  )
}
