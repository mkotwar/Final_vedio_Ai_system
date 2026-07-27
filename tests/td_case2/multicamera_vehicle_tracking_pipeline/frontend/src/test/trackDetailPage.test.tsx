import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import TrackDetailPage from '../pages/TrackDetailPage'
import { renderWithProviders } from './renderWithProviders'

describe('TrackDetailPage', () => {
  it('renders track detail, observations, and available media', async () => {
    renderWithProviders(<TrackDetailPage />, {
      route: '/tracks/RUN_20260724_151402%3ACAM_001%3ATRACK_4',
      path: '/tracks/:trackUuid',
    })
    expect((await screen.findAllByText('DL8CBF6268')).length).toBeGreaterThan(0)
    expect(screen.getByRole('img', { name: 'BEST_VEHICLE_CROP' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'PLATE_CROP' })).toBeInTheDocument()
  })
})
