import { Link } from 'react-router-dom'
import type { ReactNode } from 'react'
import type { MediaReference } from '../../types/media'
import type { PlateResult } from '../../types/plate'
import EvidencePreview from '../evidence/EvidencePreview'
import StatusBadge from '../states/StatusBadge'
import { pickMediaPair } from './mediaGroups'
import { formatPlateDisplay, formatPlateStatus, formatVehicleTimestamp } from './vehiclePresentation'

interface VehicleIdentityCardProps {
  eyebrow?: string
  title: string
  identifier?: string | null
  subtitle?: string | null
  status?: string | null
  vehicleClass?: string | null
  colour?: string | null
  plateResult?: PlateResult | null
  plate?: string | null
  plateStatus?: string | null
  plateConfidence?: number | null
  cameraCodes?: string[]
  firstSeenAt?: string | null
  lastSeenAt?: string | null
  confidence?: number | null
  vehicleMedia?: MediaReference | null
  plateMedia?: MediaReference | null
  globalMembership?: ReactNode
  detailHref?: string
  detailLabel?: string
  compact?: boolean
  memberCount?: number | null
  showPlateMedia?: boolean
}

export default function VehicleIdentityCard({
  eyebrow,
  title,
  identifier,
  subtitle,
  status: _status,
  vehicleClass,
  colour,
  plateResult,
  plate,
  plateStatus,
  plateConfidence,
  cameraCodes = [],
  firstSeenAt,
  lastSeenAt,
  confidence,
  vehicleMedia,
  plateMedia,
  globalMembership,
  detailHref,
  detailLabel = 'Open details',
  compact = false,
  memberCount,
  showPlateMedia = true,
}: VehicleIdentityCardProps) {
  const resolvedMedia = pickMediaPair([vehicleMedia], [plateMedia])
  const plateDisplay = formatPlateDisplay(plate, plateStatus, plateResult)
  const normalizedPlateStatus = formatPlateStatus(plateStatus, plateResult)
  const classHeading = vehicleClass || title || 'Vehicle'
  const detailItems = [
    { label: 'Plate', value: plateDisplay },
    { label: 'Camera', value: cameraCodes.length > 0 ? cameraCodes.join(', ') : 'N/A' },
    { label: 'First seen', value: formatVehicleTimestamp(firstSeenAt) },
    { label: 'Last seen', value: formatVehicleTimestamp(lastSeenAt) },
    { label: 'Confidence', value: typeof confidence === 'number' ? `${Math.round(confidence * 100)}%` : 'N/A' },
  ]

  if (typeof memberCount === 'number' && memberCount > 1) {
    detailItems.push({ label: 'Member tracks', value: String(memberCount) })
  }

  return (
    <article className={`vehicle-card${compact ? ' vehicle-card--compact' : ''}`}>
      <div className="vehicle-card__vehicle-media">
        <EvidencePreview
          media={resolvedMedia.vehicleMedia}
          title={resolvedMedia.vehicleMedia?.media_type || 'Vehicle image'}
          buttonLabel={`Open ${title} vehicle preview`}
          viewportClassName={`vehicle-card__vehicle-viewport${compact ? ' vehicle-card__vehicle-viewport--compact' : ''}`}
          imageClassName="vehicle-card__image"
          placeholderClassName={`vehicle-card__vehicle-placeholder${compact ? ' vehicle-card__vehicle-placeholder--compact' : ''}`}
          placeholderText={resolvedMedia.vehicleMedia?.error_detail || 'Vehicle image unavailable'}
        />
      </div>

      {showPlateMedia ? (
        <div className="vehicle-card__plate-media">
          <EvidencePreview
            media={resolvedMedia.plateMedia}
            title={resolvedMedia.plateMedia?.media_type || 'Plate image'}
            buttonLabel={`Open ${title} plate preview`}
            viewportClassName="vehicle-card__plate-viewport"
            imageClassName="vehicle-card__image"
            placeholderClassName="vehicle-card__plate-placeholder"
            placeholderText={resolvedMedia.plateMedia?.error_detail || 'No plate image available'}
          />
          <div className="vehicle-card__plate-caption">
            <span className="vehicle-card__plate-label">Plate evidence</span>
            <strong>{plateDisplay}</strong>
          </div>
        </div>
      ) : null}

      <div className="vehicle-card__content">
        <div className="vehicle-card__header">
          <div>
            {eyebrow ? <p className="vehicle-card__eyebrow">{eyebrow}</p> : null}
            <h3>{classHeading}</h3>
            {colour ? <p className="vehicle-card__colour">{colour}</p> : null}
            {subtitle ? <p className="vehicle-card__subtitle">{subtitle}</p> : null}
          </div>
          <div className="vehicle-card__header-badges">
            {plateStatus ? <StatusBadge value={normalizedPlateStatus} /> : null}
          </div>
        </div>

        <dl className="vehicle-card__details">
          {detailItems.map((item) => (
            <div key={item.label}>
              <dt>{item.label}</dt>
              <dd className={item.label.includes('seen') ? 'vehicle-card__timestamp' : undefined}>{item.value}</dd>
            </div>
          ))}
          {globalMembership ? (
            <div className="vehicle-card__details--wide">
              <dt>Global membership</dt>
              <dd>{globalMembership}</dd>
            </div>
          ) : null}
          {typeof (plateResult?.ocr_confidence ?? plateConfidence) === 'number' ? (
            <div>
              <dt>OCR confidence</dt>
              <dd>{`${Math.round(((plateResult?.ocr_confidence ?? plateConfidence) || 0) * 100)}%`}</dd>
            </div>
          ) : null}
        </dl>

        {identifier ? <p className="vehicle-card__id">{identifier}</p> : null}

        {detailHref ? (
          <div className="vehicle-card__actions">
            <Link className="button button--secondary" to={detailHref}>
              {detailLabel}
            </Link>
          </div>
        ) : null}
      </div>
    </article>
  )
}
