import type { VehicleSearchResultItem } from '../../types/search'
import VehicleIdentityCard from '../vehicle/VehicleIdentityCard'

interface VehicleSearchResultCardProps {
  result: VehicleSearchResultItem
}

export default function VehicleSearchResultCard({ result }: VehicleSearchResultCardProps) {
  const detailHref =
    result.result_type === 'GLOBAL_VEHICLE' && result.global_vehicle_code
      ? `/global-vehicles/${encodeURIComponent(result.global_vehicle_code)}`
      : result.track_uuid
        ? `/tracks/${encodeURIComponent(result.track_uuid)}`
        : '#'

  return (
    <article className="search-result-card">
      <div className="search-result-card__body">
        <VehicleIdentityCard
          eyebrow={result.result_type.replace('_', ' ')}
          title={result.result_type === 'GLOBAL_VEHICLE' ? 'Global vehicle' : 'Local track'}
          identifier={result.global_vehicle_code || result.track_uuid || 'Search result'}
          status={result.plate_status}
          vehicleClass={result.class_name}
          colour={result.colour}
          plateResult={result.plate_result}
          plate={result.plate}
          plateStatus={result.plate_status}
          plateConfidence={null}
          cameraCodes={result.camera_codes}
          firstSeenAt={result.first_seen_at}
          lastSeenAt={result.last_seen_at}
          confidence={result.confidence}
          vehicleMedia={result.primary_vehicle_media || result.primary_media}
          plateMedia={result.primary_plate_media}
          memberCount={result.member_track_count}
          showPlateMedia={false}
          detailHref={detailHref}
        />

        <div className="search-result-card__reasons">
          {result.match_reasons.map((reason) => (
            <span key={`${result.global_vehicle_code || result.track_uuid}-${reason}`} className="badge badge--neutral">
              {reason}
            </span>
          ))}
        </div>

        <p className="search-result-card__relevance">
          Relevance {typeof result.relevance_score === 'number' ? result.relevance_score.toFixed(2) : 'N/A'}
        </p>
      </div>
    </article>
  )
}
