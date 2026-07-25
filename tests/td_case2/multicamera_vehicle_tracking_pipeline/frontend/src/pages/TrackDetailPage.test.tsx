import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import TrackDetailPage from './TrackDetailPage'
import { renderWithProviders } from '../test/renderWithProviders'

describe('TrackDetailPage', () => {
  it('renders available evidence images and track details', async () => {
    renderWithProviders(<TrackDetailPage />, {
      route: '/tracks/RUN_20260724_151402%3ACAM_001%3ATRACK_4',
      path: '/tracks/:trackUuid',
    })

    expect(await screen.findByText('DL8CBF6268')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'BEST_VEHICLE_CROP' })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'PLATE_CROP' })).toBeInTheDocument()
    expect(screen.queryByText('Reference only')).not.toBeInTheDocument()
  })
})
