import type { PlateResult } from '../../types/plate'

export function formatPlateDisplay(plate?: string | null, plateStatus?: string | null, plateResult?: PlateResult | null): string {
  const normalizedStatus = String(plateResult?.status || plateStatus || '').trim().toUpperCase()
  const normalizedPlate = normalizePlateText(plateResult?.display_text || plateResult?.normalized_text || plateResult?.raw_text || plate)

  if (normalizedStatus === 'PARTIAL') {
    if (!normalizedPlate) {
      return 'Partial plate'
    }
    return normalizedPlate.length > 4 ? `...${normalizedPlate.slice(-4)}` : normalizedPlate
  }

  if (normalizedPlate) {
    return normalizedPlate
  }

  switch (normalizedStatus) {
    case 'UNREADABLE':
      return 'Unreadable'
    case 'NO_PLATE_DETECTED':
      return 'Not detected'
    case 'PARTIAL':
      return 'Partial plate'
    default:
      return 'No plate result'
  }
}

export function formatPlateStatus(plateStatus?: string | null, plateResult?: PlateResult | null): string {
  const normalizedStatus = String(plateResult?.status || plateStatus || '').trim().toUpperCase()
  if (!normalizedStatus) {
    return 'UNKNOWN'
  }
  return normalizedStatus
}

export function formatVehicleTimestamp(value?: string | null): string {
  if (!value) {
    return 'N/A'
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) {
    return value
  }

  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  }).format(parsed)
}

function normalizePlateText(plate?: string | null): string | null {
  const normalizedPlate = String(plate || '').trim().toUpperCase()
  return normalizedPlate || null
}
