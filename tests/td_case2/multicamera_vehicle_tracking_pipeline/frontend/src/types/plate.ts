export interface PlateResult {
  raw_text?: string | null
  normalized_text?: string | null
  display_text?: string | null
  status?: string | null
  verification_status?: string | null
  plate_pattern?: string | null
  ocr_confidence?: number | null
  detector_confidence?: number | null
  source_media_id?: string | null
}
