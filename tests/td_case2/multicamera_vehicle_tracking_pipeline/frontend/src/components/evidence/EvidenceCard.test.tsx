import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import EvidenceCard from './EvidenceCard'

describe('EvidenceCard', () => {
  it('renders an image for available local media', async () => {
    render(
      <EvidenceCard
        media={{
          media_id: 'media-1',
          media_type: 'BEST_VEHICLE_CROP',
          availability: 'LOCAL_FILE',
          content_url: '/api/v1/media/media-1/content',
          is_primary: true,
          frame_number: 12,
          quality_score: 0.92,
          selection_rank: 1,
        }}
      />,
    )

    const image = await screen.findByRole('img', { name: 'BEST_VEHICLE_CROP' })
    expect(image).toHaveAttribute('src', 'http://127.0.0.1:8000/api/v1/media/media-1/content')
    expect(screen.getByText('PRIMARY')).toBeInTheDocument()
  })

  it('shows a placeholder for reference-only media', () => {
    render(
      <EvidenceCard
        media={{
          media_id: 'media-2',
          media_type: 'PLATE_CROP',
          availability: 'REFERENCE_ONLY',
        }}
      />,
    )

    expect(screen.getByText('Image unavailable')).toBeInTheDocument()
    expect(screen.getByText('Reference only')).toBeInTheDocument()
  })

  it('shows a placeholder for missing media', () => {
    render(
      <EvidenceCard
        media={{
          media_id: 'media-3',
          media_type: 'BEST_VEHICLE_CROP',
          availability: 'MISSING',
        }}
      />,
    )

    expect(screen.getByText('Local evidence file not found')).toBeInTheDocument()
  })

  it('falls back safely when image loading fails', async () => {
    render(
      <EvidenceCard
        media={{
          media_id: 'media-1',
          media_type: 'BEST_VEHICLE_CROP',
          availability: 'LOCAL_FILE',
          content_url: '/api/v1/media/media-1/content',
        }}
      />,
    )

    const image = await screen.findByRole('img', { name: 'BEST_VEHICLE_CROP' })
    fireEvent.error(image)

    await waitFor(() => expect(screen.getByText('The evidence image could not be loaded.')).toBeInTheDocument())
  })

  it('does not render absolute local paths', () => {
    render(
      <EvidenceCard
        media={{
          media_id: 'media-4',
          media_type: 'BEST_VEHICLE_CROP',
          availability: 'UNSAFE_REFERENCE',
          error_detail: 'Unsafe media reference.',
        }}
      />,
    )

    expect(screen.queryByText(/C:\\/)).not.toBeInTheDocument()
  })
})
